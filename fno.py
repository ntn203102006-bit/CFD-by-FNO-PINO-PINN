# -*- coding: utf-8 -*-
"""
FNO (Fourier Neural Operator) — dự đoán trực tiếp H=10 frame tương lai
từ k=4 frame quá khứ trên lưới 320x40 (downsample x2).

Cùng dữ liệu/split/test-set với PINN & PINO (data_common.py).
Output:
    output/preds_fno.npz        (test predictions, đơn vị vật lý)
    output/fno_model.pt
    output/fno_train_log.json
Chạy:  python fno.py [--fold N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_common as dc

OUT = dc.OUT
OUTM = os.path.join(dc.ROOT, "results", "fno")   # ket qua fno nam rieng
os.makedirs(OUTM, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------- kiến trúc -------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (in_ch * out_ch)
        self.w = nn.Parameter(scale * torch.randn(
            in_ch, out_ch, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, Hh, Ww = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.w.shape[1], Hh, xf.shape[-1],
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.w)
        return torch.fft.irfft2(out, s=(Hh, Ww))


class FNO2d(nn.Module):
    def __init__(self, in_ch, out_ch, width=24, blocks=4, m1=12, m2=20):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                spec=SpectralConv2d(width, width, m1, m2),
                lin=nn.Conv2d(width, width, 1),
            )) for _ in range(blocks)])
        self.mid = nn.Conv2d(width, width, 1)
        self.out = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        x = self.lift(x)
        for b in self.blocks:
            x = F.gelu(b["lin"](x) + b["spec"](x))
        x = F.gelu(self.mid(x))
        return self.out(x)


# ------------------------- dữ liệu -------------------------
def get_norm_stack(stats):
    """F_norm [t, 2, y, x] đã chuẩn hoá (chỉ dùng Mean/std vùng train)."""
    Fs = dc.load_stacked_ds().astype(np.float32)
    F_norm = Fs.copy()
    F_norm[:, 0] = (F_norm[:, 0] - stats["u_mean"]) / stats["u_std"]
    F_norm[:, 1] = (F_norm[:, 1] - stats["v_mean"]) / stats["v_std"]
    return np.ascontiguousarray(F_norm)


def mesh_channels(ny, nx):
    gy = np.linspace(0, 1, ny, dtype=np.float32)
    gx = np.linspace(0, 1, nx, dtype=np.float32)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    return np.stack([yy, xx])[None]           # (1,2,ny,nx)


def batch_windows(F_norm, mesh, starts):
    """starts: (B,) -> X (B, 2K+2, ny, nx), Y (B, 2H, ny, nx) -- torch."""
    B = len(starts)
    X = np.empty((B, 2 * dc.K, dc.NDSY, dc.NDSX), np.float32)
    Y = np.empty((B, 2 * dc.H, dc.NDSY, dc.NDSX), np.float32)
    for i, s in enumerate(starts):
        X[i] = F_norm[s:s + dc.K].reshape(2 * dc.K, dc.NDSY, dc.NDSX)
        Y[i] = F_norm[s + dc.K: s + dc.K + dc.H].reshape(2 * dc.H, dc.NDSY, dc.NDSX)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    m = torch.from_numpy(mesh).expand(B, -1, -1, -1)
    return torch.cat([X, m], dim=1), Y


def predict_windows(model, F_norm, mesh, starts, stats, bs=16, device="cpu"):
    """Trả về preds (n, 2H, ny, nx) ĐƠN VỊ VẬT LÝ + trues."""
    model.eval()
    n = len(starts)
    preds = np.empty((n, 2 * dc.H, dc.NDSY, dc.NDSX), np.float32)
    trues = np.empty_like(preds)
    with torch.no_grad():
        for i in range(0, n, bs):
            X, Y = batch_windows(F_norm, mesh, starts[i:i + bs])
            out = model(X.to(device))
            preds[i:i + bs] = out.cpu().numpy()
            trues[i:i + bs] = Y.numpy()
    # denorm
    preds[:, 0::2] = preds[:, 0::2] * stats["u_std"] + stats["u_mean"]
    preds[:, 1::2] = preds[:, 1::2] * stats["v_std"] + stats["v_mean"]
    trues[:, 0::2] = trues[:, 0::2] * stats["u_std"] + stats["u_mean"]
    trues[:, 1::2] = trues[:, 1::2] * stats["v_std"] + stats["v_mean"]
    return preds, trues


def to_np(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def main():
    global stats
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1,
                    help=">=0: chạy fold đó (thí nghiệm phụ 5-fold); mặc định split chính")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    args = ap.parse_args()

    device = "cpu"
    stats = dc.get_train_stats()
    F_norm = get_norm_stack(stats)
    mesh = mesh_channels(dc.NDSY, dc.NDSX)
    ds_mask = dc.mask_ds()

    if args.fold >= 0:
        sp = dc.make_fold_split(args.fold)
        tag = f"_fold{args.fold}"
    else:
        sp = dc.make_split()
        tag = ""
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"splits: train {len(tr)} / val {len(va)} / test {len(te)}")

    model = FNO2d(in_ch=2 * dc.K + 2, out_ch=2 * dc.H).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"FNO width=24 blocks=4 modes=(12,20) params={n_par/1e3:.1f}K")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, len(tr) // args.batch), eta_min=args.lr * 0.05)
    rng = np.random.default_rng(SEED)

    log = dict(epoch=[], train_mse=[], val_mse=[], t=[])
    best_val, best_state = float("inf"), None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for i in range(0, len(tr), args.batch):
            idx = tr[perm[i:i + args.batch]]
            X, Y = batch_windows(F_norm, mesh, idx)
            opt.zero_grad()
            loss = F.mse_loss(model(X), Y)
            loss.backward()
            opt.step()
            sched.step()
            tot += to_np(loss); nb += 1
        # val mỗi epoch (mẫu cố định tối đa 100 window cho nhanh)
        vidx = va if len(va) <= 100 else va[rng.permutation(len(va))[:100]]
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(vidx), 32):
                X, Y = batch_windows(F_norm, mesh, vidx[i:i + 32])
                vl.append(to_np(F.mse_loss(model(X), Y)))
            v = float(np.mean(vl))
        log["epoch"].append(ep); log["train_mse"].append(tot / nb)
        log["val_mse"].append(v); log["t"].append(round(time.time() - t0, 1))
        if v < best_val:
            best_val = v
            best_state = {k: x.clone() for k, x in model.state_dict().items()}
        print(f"ep {ep:3d}  train {tot/nb:.5f}  val {v:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        if ep >= 20 and min(log["val_mse"][-8:]) > best_val * 0.98:
            print("early stop (val plateau)")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- dự đoán test ----
    preds, trues = predict_windows(model, F_norm, mesh, te, stats)
    np.savez_compressed(
        os.path.join(OUTM, f"preds_fno{tag}.npz"),
        preds=preds, trues=trues, starts=te,
        grid=dict(ny=dc.NDSY, nx=dc.NDSX))
    torch.save(model.state_dict(), os.path.join(OUTM, f"fno_model{tag}.pt"))
    json.dump(log, open(os.path.join(OUTM, f"fno_train_log{tag}.json"), "w"))
    test_mse = float(np.mean((preds - trues) ** 2))
    print(f"test MSE (physical units) = {test_mse:.6e}")
    print(f"done in {(time.time()-t0)/60:.1f} min")


stats = None  # gan trong main()


if __name__ == "__main__":
    main()
