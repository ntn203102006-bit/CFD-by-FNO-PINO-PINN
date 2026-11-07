# -*- coding: utf-8 -*-
"""
Đánh giá & so sánh PINN / FNO / PINO — cùng test set, cùng nhiệm vụ
(dự đoán trực tiếp H=10 frame tương lai từ k=4 frame quá khứ).

Đọc predictions của cả ba mô hình trong results/<model>/; PINN được
downsample từ lưới 640x80 xuống 320x40 để so sánh công bằng.

Metrics: MSE, RMSE, MAE, R² (pooled + per-frame; u/v riêng + gộp; all/fluid nodes).
Vẽ: per-frame MSE, snapshot dự đoán vs truth, probe wake, bar chart.

Chạy:  python evaluate.py
Output: results/comparison/evaluation_report.md, metrics.json, cmp_*.png
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_common as dc

OUT = dc.OUT
RESULTS = os.path.join(dc.ROOT, "results")
CMP = os.path.join(RESULTS, "comparison")
os.makedirs(CMP, exist_ok=True)
MODEL_DIRS = {"fno": os.path.join(RESULTS, "fno"),
              "pino": os.path.join(RESULTS, "pino"),
              "pinn": os.path.join(RESULTS, "pinn")}
MODELS = ["fno", "pino", "pinn"]
LABELS = {"fno": "FNO", "pino": "PINO", "pinn": "PINN"}
COLORS = {"fno": "tab:blue", "pino": "tab:green", "pinn": "tab:red"}


# ------------------------------------------------------------------ utils
def _find(name):
    """Tìm file theo thứ tự: results/<algo>/ (ưu tiên) -> output/ (fallback)."""
    base = name.split("_", 1)[0]         # pinn_global.pt / fno_train_log.json -> pinn/fno
    if name.startswith("preds_"):
        base = name[len("preds_"):].rsplit(".npz", 1)[0]   # preds_fno.npz -> fno
    if base in MODEL_DIRS:
        d0 = os.path.join(MODEL_DIRS[base], name)
        if os.path.exists(d0):
            return d0
    fp = os.path.join(OUT, name)
    return fp if os.path.exists(fp) else None


def load_model(name):
    """(preds (n,2H,ny,nx), trues, starts, ny, nx) hoặc None."""
    fp = _find(f"preds_{name}.npz")
    if fp is None:
        return None
    z = np.load(fp, allow_pickle=True)
    g = z["grid"].item()
    return z["preds"], z["trues"], z["starts"], int(g["ny"]), int(g["nx"])


def downsample(p, ny, nx):
    if (ny, nx) == (dc.NDSY, dc.NDSX):
        return p.astype(np.float64)
    if (ny, nx) == (dc.NY, dc.NX):
        return p[..., ::2, ::2].astype(np.float64)
    raise ValueError(f"grid {ny}x{nx} không hỗ trợ")


def metrics(preds, trues):
    d = preds - trues
    se = float((d ** 2).sum())
    sst = float(((trues - trues.mean()) ** 2).sum())
    return dict(mse=se / d.size, rmse=float(np.sqrt(se / d.size)),
                mae=float(np.abs(d).mean()), r2=1.0 - se / sst)


def eval_common(data):
    """So sánh trên lưới chung 320x40."""
    mask = dc.mask_ds()                      # (40,320) True = node trụ
    res = {}
    for m, (preds, trues, starts, ny, nx) in data.items():
        p = downsample(preds, ny, nx)
        t = downsample(trues, ny, nx)
        e = dict()
        e["all"] = metrics(p, t)
        e["u"] = metrics(p[:, 0::2], t[:, 0::2])
        e["v"] = metrics(p[:, 1::2], t[:, 1::2])
        # fluid-only
        d = p - t
        mf = ~mask
        se = float((d[:, :, mf] ** 2).sum())
        sst = float(((t[:, :, mf] - t[:, :, mf].mean()) ** 2).sum())
        e["fluid"] = dict(mse=se / d[:, :, mf].size, rmse=float(np.sqrt(se / d[:, :, mf].size)),
                          mae=float(np.abs(d[:, :, mf]).mean()), r2=1.0 - se / sst)
        # per-frame
        d2 = (p - t) ** 2
        tmean = t.mean()
        per_u, per_v, per_r2 = [], [], []
        for mf_ in range(dc.H):
            mse_u = float(d2[:, 2 * mf_].mean())
            mse_v = float(d2[:, 2 * mf_ + 1].mean())
            per_u.append(mse_u)
            per_v.append(mse_v)
            # R2 per frame (gộp u+v): SST của frame đó so với mean toàn cục
            t_f = t[:, 2 * mf_: 2 * mf_ + 2]
            sst_f = float(((t_f - tmean) ** 2).sum())
            sse_f = float(d2[:, 2 * mf_: 2 * mf_ + 2].sum())
            per_r2.append(1.0 - sse_f / sst_f)
        e["per_frame_u_mse"] = per_u
        e["per_frame_v_mse"] = per_v
        e["per_frame_r2"] = per_r2
        res[m] = e
    return res


def fmt_row(m, e):
    a = e["all"]
    return (LABELS[m], f"{a['mse']:.4e}", f"{a['rmse']:.4f}", f"{a['mae']:.4f}",
            f"{e['u']['r2']:.3f}", f"{e['v']['r2']:.3f}", f"{a['r2']:.3f}",
            f"{e['fluid']['mse']:.4e}", f"{e['fluid']['r2']:.3f}")


# ------------------------------------------------------------------ plots
def plot_per_frame(res, fp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    t = np.arange(1, dc.H + 1) * dc.DT
    for m, e in res.items():
        axes[0].plot(t, e["per_frame_u_mse"], "-o", ms=3, label=LABELS[m], color=COLORS[m])
        axes[1].plot(t, e["per_frame_v_mse"], "-o", ms=3, label=LABELS[m], color=COLORS[m])
    axes[0].set_title("MSE theo frame — thành phần u")
    axes[1].set_title("MSE theo frame — thành phần v")
    for ax in axes:
        ax.set_xlabel("Δt trước (s)"); ax.set_ylabel("MSE")
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f"So sánh per-frame (k={dc.K} quá khứ → H={dc.H} tương lai, lưới chung 320x40)",
                 fontweight="bold")
    fig.savefig(fp, dpi=130, facecolor="white"); plt.close(fig)


def plot_snapshots(data, starts, i_win, fp):
    rows = [("Truth (CFD)", "true")] + [(LABELS[m], m) for m in data]
    ds_m = [0, 4, 9]
    extent = [dc.X0, dc.X0 + dc.NDSX * dc.DX * dc.DS,
              dc.Y0, dc.Y0 + dc.NDSY * dc.DY * dc.DS]
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 2.5 * len(rows)),
                             constrained_layout=True)
    vmin, vmax = -0.3, 1.6
    for r, (label, kind) in enumerate(rows):
        for c, m_ in enumerate(ds_m):
            ax = axes[r, c]
            if kind == "true":
                arr = downsample(data[list(data)[0]][1][i_win],
                                 data[list(data)[0]][3], data[list(data)[0]][4])[2 * m_]
            else:
                preds, _, _, ny, nx = data[kind]
                arr = downsample(preds[i_win], ny, nx)[2 * m_]
            im = ax.imshow(arr, origin="lower", extent=extent, cmap="RdBu_r",
                           vmin=vmin, vmax=vmax, aspect="auto")
            if r == 0:
                ax.set_title(f"Dự báo t+{(m_+1)*dc.DT:.2f} (frame +{m_+1})", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.add_patch(plt.Circle((0, 0), 0.05, fill=False, color="k", lw=0.9))
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"Trường u — window test s={starts[i_win]} (mô hình vs truth, 320x40)",
                 fontweight="bold")
    fig.savefig(fp, dpi=130, facecolor="white"); plt.close(fig)


def plot_model_previews(data, starts, i_win):
    """Write all model previews from one shared window and shared color limits."""
    steps = (0, 4, 9)
    reference_true = downsample(
        data[MODELS[0]][1][i_win], data[MODELS[0]][3], data[MODELS[0]][4])
    errors = {
        m: np.abs(downsample(data[m][0][i_win], data[m][3], data[m][4])
                  - reference_true)
        for m in MODELS
    }
    error_limit = max(float(np.max(errors[m][2 * step]))
                      for m in MODELS for step in steps)
    error_limit = max(error_limit, 1e-6)
    extent = [dc.X0, dc.X0 + dc.NDSX * dc.DX * dc.DS,
              dc.Y0, dc.Y0 + dc.NDSY * dc.DY * dc.DS]

    for m in MODELS:
        prediction = downsample(
            data[m][0][i_win], data[m][3], data[m][4])
        fig, axes = plt.subplots(3, 3, figsize=(14, 8),
                                 constrained_layout=True)
        for col, step in enumerate(steps):
            fields = (
                ("Truth (CFD)", reference_true[2 * step], "RdBu_r", -0.3, 1.6),
                (f"{LABELS[m]} prediction", prediction[2 * step],
                 "RdBu_r", -0.3, 1.6),
                ("Absolute error", errors[m][2 * step],
                 "magma", 0.0, error_limit),
            )
            for row, (title, field, cmap, vmin, vmax) in enumerate(fields):
                ax = axes[row, col]
                im = ax.imshow(field, origin="lower", extent=extent,
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               aspect="auto")
                ax.set_title(f"{title} — t+{step + 1}")
                ax.add_patch(plt.Circle((0, 0), 0.05, fill=False,
                                        color="k", lw=0.9))
                fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(
            f"{LABELS[m]} — shared test window s={int(starts[i_win])}, "
            "shared color scales, grid 320x40",
            fontweight="bold")
        fig.savefig(os.path.join(MODEL_DIRS[m], f"{m}_preview.png"),
                    dpi=130, facecolor="white")
        plt.close(fig)


def plot_probe(data, starts, i_win, fp):
    iy, ix = 20, 120
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    tt = np.arange(1, dc.H + 1) * dc.DT
    first = True
    for m, (preds, trues, _, ny, nx) in data.items():
        t = downsample(trues[i_win], ny, nx)
        p = downsample(preds[i_win], ny, nx)
        if first:
            axes[0].plot(tt, t[0::2, iy, ix], "k--o", ms=4, lw=1.3, label="Truth CFD")
            axes[1].plot(tt, t[1::2, iy, ix], "k--o", ms=4, lw=1.3, label="Truth CFD")
            first = False
        axes[0].plot(tt, p[0::2, iy, ix], "-o", ms=3, color=COLORS[m], label=LABELS[m])
        axes[1].plot(tt, p[1::2, iy, ix], "-o", ms=3, color=COLORS[m], label=LABELS[m])
    axes[0].set_title("u tại probe wake"); axes[1].set_title("v tại probe wake")
    for ax in axes:
        ax.set_xlabel("Δt dự báo (s)"); ax.set_ylabel("vận tốc")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Probe wake (y≈{iy*2*dc.DY:.2f}, x≈{ix*2*dc.DX:.2f}) — window s={starts[i_win]}",
                 fontweight="bold")
    fig.savefig(fp, dpi=130, facecolor="white"); plt.close(fig)


def plot_bars(res, fp):
    models = list(res)
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].bar(x, [res[m]["all"]["mse"] for m in models],
                color=[COLORS[m] for m in models])
    axes[0].set_yscale("log"); axes[0].set_xticks(x, [LABELS[m] for m in models])
    axes[0].set_ylabel("MSE pooled (log)"); axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_title("MSE tổng (đơn vị vật lý, 320x40)")
    axes[1].bar(x - 0.18, [res[m]["u"]["r2"] for m in models], width=0.36, label="u")
    axes[1].bar(x + 0.18, [res[m]["v"]["r2"] for m in models], width=0.36, label="v")
    axes[1].set_xticks(x, [LABELS[m] for m in models]); axes[1].legend()
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("R²")
    axes[1].set_title("R² theo thành phần"); axes[1].grid(alpha=0.3, axis="y")
    fig.savefig(fp, dpi=130, facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ rollout (phụ)
def rollout_simple(n_win=30, n_steps=4):
    """Autoregressive rollout cho FNO/PINO (KHÔNG dùng trong bảng chính).
    Mỗi bước: input k frame gần nhất (truth nếu còn trong range, else prediction
    của bước trước); trượt đầu vào s += K mỗi bước."""
    import torch
    import fno as fno_mod
    stats = dc.get_train_stats()
    F_all = dc.load_stacked_ds().astype(np.float32)
    mesh = fno_mod.mesh_channels(dc.NDSY, dc.NDSX)
    results = {}
    for m in ("fno", "pino"):
        wp = _find(f"{m}_model.pt")
        pp = _find(f"preds_{m}.npz")
        if not (wp and pp):
            continue
        starts = np.load(pp, allow_pickle=True)["starts"]
        model = fno_mod.FNO2d(in_ch=2 * dc.K + 2, out_ch=2 * dc.H)
        model.load_state_dict(torch.load(wp, map_location="cpu", weights_only=True))
        model.eval()
        rr = []
        for s in starts[starts + n_steps * dc.K < dc.NT - 1][:n_win]:
            cur_s = s
            errs = np.zeros(dc.H * n_steps)
            for step in range(n_steps):
                # input: k frame, dùng truth nếu có, else dùng pred đã lưu (đơn giản hoá)
                src = F_all[cur_s:cur_s + dc.K].copy()
                Xn = src.copy()
                Xn[:, 0] = (Xn[:, 0] - stats["u_mean"]) / stats["u_std"]
                Xn[:, 1] = (Xn[:, 1] - stats["v_mean"]) / stats["v_std"]
                X = torch.cat([torch.from_numpy(
                    Xn.reshape(1, 2 * dc.K, dc.NDSY, dc.NDSX).copy()),
                    torch.from_numpy(mesh)], dim=1)
                with torch.no_grad():
                    out = model(X).numpy()[0].astype(np.float32)
                out_p = out.copy()
                out_p[0::2] = out[0::2] * stats["u_std"] + stats["u_mean"]
                out_p[1::2] = out[1::2] * stats["v_std"] + stats["v_mean"]
                t_true = F_all[cur_s + dc.K: cur_s + dc.K + dc.H].reshape(2 * dc.H, dc.NDSY, dc.NDSX)
                errs[0] = float(((out_p - t_true) ** 2).mean())
                cur_s += dc.K                       # trượt theo K (mức input constant)
            rr.append(errs[0])
        if rr:
            results[m] = np.array(rr)
    return results


# ------------------------------------------------------------------ main
def main():
    data = {}
    for m in MODELS:
        got = load_model(m)
        if got is None:
            raise FileNotFoundError(
                f"Thiếu predictions cho {LABELS[m]}: preds_{m}.npz; "
                "không thể tạo báo cáo so sánh đủ cả ba mô hình."
            )
        data[m] = got
        print(f"{LABELS[m]}: {got[0].shape[0]} windows test, lưới {got[3]}x{got[4]}")

    starts = data[MODELS[0]][2]
    for m in MODELS[1:]:
        if not np.array_equal(starts, data[m][2]):
            raise ValueError(
                f"Predictions của {LABELS[m]} không dùng cùng test windows "
                f"với {LABELS[MODELS[0]]}."
            )
        reference_true = downsample(
            data[MODELS[0]][1], data[MODELS[0]][3], data[MODELS[0]][4])
        model_true = downsample(data[m][1], data[m][3], data[m][4])
        if not np.allclose(reference_true, model_true, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"Ground truth của {LABELS[m]} không trùng với "
                f"{LABELS[MODELS[0]]} sau khi downsample."
            )

    res = eval_common(data)
    i_win = int(np.argmin(np.abs(starts - np.median(starts))))

    # -------- bảng chính --------
    hdr = ["Model", "MSE", "RMSE", "MAE", "R2 u", "R2 v", "R2 pooled",
           "MSE fluid", "R2 fluid"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for m in res:
        lines.append("| " + " | ".join(fmt_row(m, res[m])) + " |")
    table_md = "\n".join(lines)

    # -------- JSON --------
    json_out = dict(
        n_test=int(len(starts)), compare_grid="320x40",
        split="purged 70/15/15 theo thời gian, purge k+H=14 frame",
        k=dc.K, H=dc.H, nu=dc.NU,
        metrics={m: {k: v for k, v in e.items()
                     if not k.startswith("per_frame")} for m, e in res.items()},
        per_frame={m: dict(mse_u=res[m]["per_frame_u_mse"],
                           mse_v=res[m]["per_frame_v_mse"],
                           r2=res[m]["per_frame_r2"]) for m in res},
    )
    json.dump(json_out, open(os.path.join(CMP, "metrics.json"), "w"), indent=2)

    # -------- per-frame table --------
    pt = ["\n### MSE theo frame Δt (trung bình u,v — lưới chung 320x40)\n",
          "| Δt s | " + " | ".join(LABELS[m] for m in res) + " |",
          "|---|" * 1 + "---|" * (len(res) - 1)]
    for m_ in range(dc.H):
        vals = [f"{0.5*(res[m]['per_frame_u_mse'][m_]+res[m]['per_frame_v_mse'][m_]):.4e}"
                for m in res]
        pt.append(f"| {(m_+1)*dc.DT:.2f} | " + " | ".join(vals) + " |")
    per_frame_md = "\n".join(pt)

    report_parts = [
        "# SO SÁNH PINN – FNO – PINO — dự báo tương lai cylinder2d",
        "",
        f"> Bài toán: X = k={dc.K} frame quá khứ → Y = H={dc.H} frame tương lai (dự báo trực tiếp).",
        f"> Split purged 70/15/15 theo thời gian; {len(starts)} window test; cùng test set cả 3.",
        f"> Bảng chính trên **lưới chung 320x40** (PINN 640x80 được downsample [::2,::2]).",
        f"> nu = U·D/Re ≈ {dc.NU:.3e} (Re≈82 — đã kiểm toán).",
        "",
        "## Bảng chính (tổng pool test set)",
        table_md,
        per_frame_md,
        "",
        "## Hình",
        "- `cmp_perframe_mse.png` — MSE theo frame",
        "- `cmp_snapshots.png` — trường u truth vs từng model",
        "- `cmp_probe.png` — probe wake time-series",
        "- `cmp_bars.png`",
        "- `results/<model>/<model>_preview.png` — cùng test window, lưới và thang màu cho cả ba",
    ]

    report_parts += ["",
                     "## Ghi chú kỹ thuật",
                     f"- PINN v5: fully convolutional residual U-Net trên các patch "
                     f"không gian có halo; nhận {dc.K} frame quá khứ và dự đoán trực "
                     f"tiếp {dc.H + 1} trường (gồm anchor m=0), adaptive weighting "
                     "cho biên và residual vật lý.",
                     f"- FNO: spectral conv (modes 12x20), width 24, supervised, lưới 320x40.",
                     f"- PINO: cùng backbone FNO — pretrain supervised rồi fine-tune "
                     f"với residual PDE (vận chuyển vorticity, không cần p) + penalty divergence.",
                     f"- Residual PDE: w_t + u w_x + v w_y = nu·lap(w), nu ≈ {dc.NU:.3e} "
                     f"(Re≈82 — suy từ St=0.157, đã kiểm toán).",
                     "- 'fluid' = bỏ node trụ (u=v=0) khi tính.",
                     f"— Sinh: {time.strftime('%Y-%m-%d %H:%M')}"]

    report = "\n".join(report_parts)
    with open(os.path.join(CMP, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    # -------- plots --------
    plot_per_frame(res, os.path.join(CMP, "cmp_perframe_mse.png"))
    plot_model_previews(data, starts, i_win)
    plot_snapshots(data, starts, i_win, os.path.join(CMP, "cmp_snapshots.png"))
    plot_probe(data, starts, i_win, os.path.join(CMP, "cmp_probe.png"))
    plot_bars(res, os.path.join(CMP, "cmp_bars.png"))
    print("\n" + table_md)
    print("\nreport: results/comparison/evaluation_report.md")


if __name__ == "__main__":
    main()
