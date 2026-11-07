# -*- coding: utf-8 -*-
"""
PINO (Physics-Informed Neural Operator) — backbone FNO + loss vật lý.

Giai đoạn 1: pretrain supervised giống FNO (data loss).
Giai đoạn 2: fine-tune với PDE residual (vận chuyển vorticity, không cần p):
    w_t + u w_x + v w_y = nu * lap(w),  w = dv/dx - du/dy
Đạo hàm không gian qua FFT (spectral), đạo hàm thời gian bằng sai phân
trung tâm trên chuỗi H frame dự đoán (dt=0.01). Cộng thêm penalty divergence.

Thang residual chuẩn hoá theo residual của ground-truth cùng operator
(pde_scale lưu trong train_stats.json) -> loss vật lý vô hướng so sánh được.

Cùng dữ liệu/split/test-set với FNO & PINN (data_common.py).
Output: output/preds_pino.npz, pino_model.pt, pino_train_log.json
Chạy:  python pino.py [--fold N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_common as dc

OUT = dc.OUT
OUTM = os.path.join(dc.ROOT, "results", "pino")   # ket qua pino nam rieng
os.makedirs(OUTM, exist_ok=True)
SEED = 42

# ------------------------- spectral derivatives (torch) -------------------------
class SpectralDeriv:
    """d/dx, d/dy, laplacian qua FFT2 trên lưới đều (ny, nx), spacing dx, dy."""

    def __init__(self, ny, nx, dx, dy):
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)      # (nx,) trục -1
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)      # (ny,) trục -2
        self.kx = torch.tensor(kx, dtype=torch.float32).view(1, 1, 1, nx)
        self.ky = torch.tensor(ky, dtype=torch.float32).view(1, 1, ny, 1)
        self.ny, self.nx = ny, nx

    def ddx(self, f):
        f = f.contiguous()
        return torch.fft.ifft2(torch.fft.fft2(f) * self.kx.to(torch.complex64)).real

    def ddy(self, f):
        f = f.contiguous()
        return torch.fft.ifft2(torch.fft.fft2(f) * self.ky.to(torch.complex64)).real

    def lap(self, f):
        f = f.contiguous()
        k2 = (self.kx ** 2 + self.ky ** 2).to(torch.complex64)
        return torch.fft.ifft2(torch.fft.fft2(f) * k2).real

def pde_residual_torch(u, v, dt, nu, sd):
    """u, v: (T, ny, nx) đơn vị vật lý (T frames dự đoán). Trả về residual
    (T-2, ny, nx) — đạo hàm thời gian central diff, không gian spectral."""
    w = sd.ddx(v) - sd.ddy(u)                      # (T, ny, nx)
    wt = (w[2:] - w[:-2]) / (2 * dt)               # frames 1..T-2
    wx = sd.ddx(w[1:-1])
    wy = sd.ddy(w[1:-1])
    lap_w = sd.lap(w[1:-1])
    return wt + u[1:-1] * wx + v[1:-1] * wy - nu * lap_w


def compute_pde_scale(stats):
    """Thang residual của ground-truth trên vùng train (lưu cache)."""
    fp = os.path.join(OUT, "cache", "pde_scale_ds.json")
    if os.path.exists(fp):
        return json.load(open(fp))["scale"]
    F = dc.load_stacked_ds().astype(np.float32)
    U, V = F[:, 0], F[:, 1]
    sp = dc.make_split()
    lo = dc.TRANSIENT
    hi = int(sp["train"].max()) + dc.K + dc.H
    idx = np.linspace(lo, hi - 3, 8, dtype=int)     # 8 đoạn 3-frame
    rs = []
    for i in idx:
        r = dc.pde_residual_np(U[i:i + 3], V[i:i + 3], dc.DT, dc.NU,
                               dc.DX * dc.DS, dc.DY * dc.DS)
        rs.append(np.abs(r))
    r = np.concatenate([x.reshape(-1) for x in rs])
    scale = float(np.percentile(r, 90.0))
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    json.dump(dict(scale=scale, p90_abs=scale, n=len(r)), open(fp, "w"), indent=2)
    return scale


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--epochs-sup", type=int, default=35)     # giai đoạn 1
    ap.add_argument("--epochs-phys", type=int, default=12)    # giai đoạn 2
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch-phys", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--w-phys", type=float, default=1.0)
    ap.add_argument("--w-div", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    global stats
    stats = dc.get_train_stats()
    F_norm = fno_mod.get_norm_stack(stats)
    mesh = fno_mod.mesh_channels(dc.NDSY, dc.NDSX)
    raw = dc.load_stacked_ds().astype(np.float32)   # đơn vị vật lý (mask...)

    if args.fold >= 0:
        sp = dc.make_fold_split(args.fold); tag = f"_fold{args.fold}"
    else:
        sp = dc.make_split(); tag = ""
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"splits: train {len(tr)} / val {len(va)} / test {len(te)}")

    pde_scale = compute_pde_scale(stats)
    print(f"pde_scale (p90 |residual| GT) = {pde_scale:.4f}")

    sd = SpectralDeriv(dc.NDSY, dc.NDSX, dc.DX * dc.DS, dc.DY * dc.DS)
    stats_u = (stats["u_mean"], stats["u_std"])
    stats_v = (stats["v_mean"], stats["v_std"])

    model = fno_mod.FNO2d(in_ch=2 * dc.K + 2, out_ch=2 * dc.H).to("cpu")
    print(f"PINO backbone FNO width=24 params={sum(p.numel() for p in model.parameters())/1e3:.1f}K")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = dict(sup=[], phys=[]) 

    # ---------------- giai đoạn 1: supervised ----------------
    n_gd = max(1, len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_sup * n_gd, eta_min=args.lr * 0.05)
    rng = np.random.default_rng(SEED)
    best_val, best_state = float("inf"), None
    for ep in range(1, args.epochs_sup + 1):
        model.train()
        perm = rng.permutation(len(tr)); tot, nb = 0.0, 0
        for i in range(0, len(tr), args.batch):
            idx = tr[perm[i:i + args.batch]]
            X, Y = fno_mod.batch_windows(F_norm, mesh, idx)
            opt.zero_grad(); loss = F.mse_loss(model(X), Y)
            loss.backward(); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        vidx = va if len(va) <= 100 else va[rng.permutation(len(va))[:100]]
        model.eval(); vl = []
        with torch.no_grad():
            for i in range(0, len(vidx), 32):
                X, Y = fno_mod.batch_windows(F_norm, mesh, vidx[i:i + 32])
                vl.append(F.mse_loss(model(X), Y).item())
        v = float(np.mean(vl))
        log["sup"].append(dict(ep=ep, tr=tot / nb, val=v))
        if v < best_val:
            best_val = v; best_state = {k: x.clone() for k, x in model.state_dict().items()}
        print(f"[sup] ep {ep:3d} train {tot/nb:.5f} val {v:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if ep >= 15 and min(x["val"] for x in log["sup"][-8:]) > best_val * 0.98:
            print("[sup] early stop"); break
    model.load_state_dict(best_state)
    sup_name = os.path.join(OUTM, f"pino_sup{tag}.pt")
    torch.save(best_state, sup_name)

    # ---------------- giai đoạn 2: physics fine-tune ----------------
    for p in opt.param_groups:
        p["lr"] = 3e-4
    n_ph = max(1, len(tr) // args.batch_phys)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_phys * n_ph, eta_min=3e-5)
    ph = 0.0
    for ep in range(1, args.epochs_phys + 1):
        model.train()
        perm = rng.permutation(len(tr)); tot, nb = 0.0, 0
        for i in range(0, len(tr), args.batch_phys):
            idx = tr[perm[i:i + args.batch_phys]]
            X, Yn = fno_mod.batch_windows(F_norm, mesh, idx)
            out = model(X)                                    # (B, 2H, ny, nx) chuẩn hoá
            # denorm (differentiable)
            u = out[:, 0::2] * stats_u[1] + stats_u[0]
            v = out[:, 1::2] * stats_v[1] + stats_v[0]
            B = u.shape[0]
            r = pde_residual_torch(
                u.permute(1, 0, 2, 3), v.permute(1, 0, 2, 3), dc.DT, dc.NU, sd)
            loss_phys = (r / pde_scale).pow(2).mean()
            # divergence penalty (vô hướng hoá bằng thang vận tốc)
            div = sd.ddx(u.permute(1, 0, 2, 3)) + sd.ddy(v.permute(1, 0, 2, 3))
            loss_div = (div / stats_u[1]).pow(2).mean()
            loss_data = F.mse_loss(out, Yn)
            loss = loss_data + args.w_phys * loss_phys + args.w_div * loss_div
            opt.zero_grad(); loss.backward(); opt.step(); sched2.step()
            tot += loss.item(); nb += 1
        ph = tot / nb
        log["phys"].append(dict(ep=ep, loss=ph))
        print(f"[phys] ep {ep:3d} loss {ph:.5f} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- dự đoán test ----------------
    preds, trues = fno_mod.predict_windows(model, F_norm, mesh, te, stats)
    np.savez_compressed(
        os.path.join(OUTM, f"preds_pino{tag}.npz"),
        preds=preds, trues=trues, starts=te,
        grid=dict(ny=dc.NDSY, nx=dc.NDSX))
    torch.save(model.state_dict(), os.path.join(OUTM, f"pino_model{tag}.pt"))
    json.dump(dict(sup=log["sup"], phys=log["phys"], pde_scale=pde_scale,
                   fold=args.fold, t_min=(time.time() - t0) / 60),
              open(os.path.join(OUTM, f"pino_train_log{tag}.json"), "w"))
    test_mse = float(np.mean((preds - trues) ** 2))
    print(f"test MSE (physical units) = {test_mse:.6e}")
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
