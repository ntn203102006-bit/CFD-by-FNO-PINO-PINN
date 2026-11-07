# -*- coding: utf-8 -*-
"""
Module dữ liệu dùng chung cho PINN / FNO / PINO — cylinder2d.vti.

Nhiệm vụ chung (giống nhau cho cả 3 thuật toán):
    Input  X = [u(t-k+1:t), v(t-k+1:t)]  (k frame quá khứ, 2 kênh)
    Output Y = [u(t+1:t+H), v(t+1:t+H)]  (H frame tương lai, 2 kênh)
Cửa sổ trượt stride=1, cùng split purged 70/15/15, cùng test set.

Mọi thông số lấy từ kết quả kiểm toán đã xác nhận (output/vti_validation_report.md):
  - lưới 640x80, 1501 frame, float32, 0 NaN/Inf
  - dx=0.01252, dy=0.01266, dt nhãn = 0.01, T = 15.0
  - trụ tâm (0,0), r ≈ 0.05 (D ≈ 0.1), 60 node exact-zero (cylinder_mask.npy)
  - inlet u = 1.000 (std 0.0003), tường trên/dưới v = 0 (no-penetration)
  - St = 0.157  =>  Re ≈ 82 (tương quan St–Re)  =>  nu = U*D/Re ≈ 1.22e-3
File gốc .vti CHỈ ĐỌC — loader cache ra .npz trong output/cache.
"""
import json
import os

import numpy as np

ROOT = r"D:\CFD_cylinder_2d"
VTI = os.path.join(ROOT, "cylinder2d_vti", "cylinder2d.vti")
OUT = os.path.join(ROOT, "output")
CACHE = os.path.join(OUT, "cache")
STATS_JSON = os.path.join(OUT, "train_stats.json")

# ------------------------- hằng số (đã kiểm toán) -------------------------
NX, NY, NT = 640, 80, 1501
DX = 0.012519561815
DY = 0.012658227848
DT = 0.01                    # nhãn thời gian mỗi frame (metadata VTI spacing)
T_TOTAL = NT * DT            # 15.01
X0, Y0 = -0.5, -0.5          # originDom
XW, YW = NX * DX, NY * DY    # rộng 8.0 x 1.0

K, H = 4, 10                 # user quyết định: k=4 frame quá khứ, H=10 frame dự đoán
TRANSIENT = 100              # bỏ frame < 100 (transient đã kiểm toán)
DS = 2                       # độ phân giải FNO/PINO: downsample x2 -> 320x40
NDSX, NDSY = NX // DS, NY // DS

# Thông số vật lý có bằng chứng: U_inf=1.0 (inlet đo được), D=0.1 (mask đo được),
# Re ≈ 82 (từ St=0.157 qua tương quan St-Re) => nu = U*D/Re
U_INF, D_CYL = 1.0, 0.1
RE_IMPLIED = 82.0
NU = U_INF * D_CYL / RE_IMPLIED   # ≈ 1.2195e-3

PURGE = K + H                # số frame purged giữa các khối (14)


def _ensure_cache():
    os.makedirs(CACHE, exist_ok=True)


def load_full():
    """Đọc (u, v) full-res [t,y,x] = (1501, 80, 640) float32. Cache npz."""
    fp = os.path.join(CACHE, "uv_full.npz")
    if os.path.exists(fp):
        z = np.load(fp)
        return z["u"], z["v"]
    import pyvista as pv
    mesh = pv.read(VTI)
    U = np.ascontiguousarray(np.asarray(mesh.point_data["u"], np.float32).reshape(NT, NY, NX))
    V = np.ascontiguousarray(np.asarray(mesh.point_data["v"], np.float32).reshape(NT, NY, NX))
    _ensure_cache()
    np.savez(fp, u=U, v=V)
    return U, V


def load_ds():
    """Downsample x2 cho FNO/PINO: [t,y,x] = (1501, 40, 320). Cache npz."""
    fp = os.path.join(CACHE, "uv_ds.npz")
    if os.path.exists(fp):
        z = np.load(fp)
        return z["u"], z["v"]
    U, V = load_full()
    Ud, Vd = U[:, ::DS, ::DS].copy(), V[:, ::DS, ::DS].copy()
    _ensure_cache()
    np.savez(fp, u=Ud, v=Vd)
    return Ud, Vd


def load_stacked_ds():
    """F = [t, 2, y, x] xuống lấy mẫu, kênh 0=u, 1=v."""
    Ud, Vd = load_ds()
    return np.stack([Ud, Vd], axis=1)


def mask_full():
    """Mask trụ full-res [y,x] bool (60 node, u=v=0 mọi frame)."""
    return np.load(os.path.join(OUT, "cylinder_mask.npy"))


def mask_ds():
    return mask_full()[::DS, ::DS]

def window_starts():
    """Cửa sổ hợp lệ: bắt đầu s >= TRANSIENT, s+K+H-1 <= NT-1 (stride 1)."""
    last = NT - 1 - K - H
    return np.arange(TRANSIENT, last + 1, dtype=np.int64)


def make_split(s=None, purge=PURGE):
    """Split khối thời gian purged 70/15/15 (deterministic, không random).

    Train: 70% cửa sổ đầu. Val/Test: của sổ sau, cách nhau purge cửa sổ
    => giữa các khối không có frame input/target chồng lấn.
    """
    if s is None:
        s = window_starts()
    n = len(s)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    i1, i2 = n_tr, n_tr + n_va
    return dict(
        train=s[:i1],
        val=s[i1 + purge: i2 + purge],
        test=s[i2 + 2 * purge:],
    )


def make_fold_split(fold, n_folds=5, s=None, purge=PURGE):
    """5-fold theo khối thời gian: fold i -> test là khối i; purge quanh test."""
    if s is None:
        s = window_starts()
    n = len(s)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    i0, i1 = edges[fold], edges[fold + 1]
    test = s[i0:i1]
    lo, hi = max(0, i0 - purge), min(n, i1 + purge)
    rest = np.concatenate([s[:lo], s[hi:]])
    n_va = int(0.15 * len(rest))
    return dict(train=rest[:-n_va] if n_va else rest, val=rest[-n_va:], test=test)


def compute_train_stats():
    """Mean/std theo kênh trên vùng frame của train (chống leakage), + thang PDE."""
    sp = make_split()
    fmax = int(sp["train"].max()) + K + H          # frame cuối vùng train (exclusive)
    Ud, Vd = load_ds()
    um, us = float(Ud[TRANSIENT:fmax].mean()), float(Ud[TRANSIENT:fmax].std())
    vm, vs = float(Vd[TRANSIENT:fmax].mean()), float(Vd[TRANSIENT:fmax].std())
    stats = dict(u_mean=um, u_std=us, v_mean=vm, v_std=vs, train_fmax=fmax)
    return stats


def get_train_stats():
    """Đọc/ tạo một lần, mọi script dùng đúng một bộ thông số."""
    if os.path.exists(STATS_JSON):
        return json.load(open(STATS_JSON))
    st = compute_train_stats()
    json.dump(st, open(STATS_JSON, "w"), indent=2)
    return st


# ------------------------- chuẩn hoá -------------------------
def norm(a, stats, ch):
    """Chuẩn hoá kênh: ch=0 -> u, ch=1 -> v. a áp dụng cho tensor [.., y, x]."""
    m = stats["u_mean"] if ch == 0 else stats["v_mean"]
    s = stats["u_std"] if ch == 0 else stats["v_std"]
    return (a - m) / s


def denorm(a, stats, ch):
    m = stats["u_mean"] if ch == 0 else stats["v_mean"]
    s = stats["u_std"] if ch == 0 else stats["v_std"]
    return a * s + m


# ------------------------- vật lý (numpy, tham chiếu/kiểm tra) -------------------------
def pde_residual_np(u, v, dt, nu, dx, dy):
    """Residual vận chuyển vorticity (không cần pressure):
        w_t + u w_x + v w_y = nu * lap(w),  w = dv/dx - du/dy
    u, v: (T, ny, nx) đơn vị vật lý, T>=3. Trả về r (T-2, ny-4, nx-4).
    Sai phân trung tâm, khớp cả FFT (PINO) lẫn FD stencil (PINN).
    """
    w = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * dx) \
        - (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dy)          # (T, ny-2, nx-2)
    wt = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dt)   # (T-2, ny-4, nx-4)
    wx = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dx)
    wy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2 * dy)
    wc = w[1:-1, 1:-1, 1:-1]
    lap = (w[1:-1, 1:-1, 2:] - 2 * wc + w[1:-1, 1:-1, :-2]) / dx ** 2 \
        + (w[1:-1, 2:, 1:-1] - 2 * wc + w[1:-1, :-2, 1:-1]) / dy ** 2
    uu = u[1:-1, 2:-2, 2:-2]
    vv = v[1:-1, 2:-2, 2:-2]
    return wt + uu * wx + vv * wy - nu * lap


def div_np(u, v, dx, dy):
    """Divergence trung tâm, chỉ phần nội miền: (ny-2, nx-2)."""
    return (u[..., 1:-1, 2:] - u[..., 1:-1, :-2]) / (2 * dx) \
        + (v[..., 2:, 1:-1] - v[..., :-2, 1:-1]) / (2 * dy)


if __name__ == "__main__":
    # tự kiểm nhanh (không chạy khi import)
    print("window count:", len(window_starts()))
    sp = make_split()
    print("train/val/test windows:", len(sp["train"]), len(sp["val"]), len(sp["test"]))
    st = get_train_stats()
    print("stats:", {k: round(v, 5) for k, v in st.items()})
