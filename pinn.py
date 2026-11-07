# -*- coding: utf-8 -*-
"""CNN-based PINN for cylinder flow with adaptive data/physics loss balancing.

The fully convolutional encoder-decoder predicts complete space-time fields from
the four observed frames. Training uses spatial tiles with context halos, so
neighboring nodes interact through shared convolutions rather than flattened
per-node feature vectors.
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

OUTM = os.path.join(dc.ROOT, "results", "pinn")
os.makedirs(OUTM, exist_ok=True)
SEED = 42
DT = dc.DT

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class PINNNet(nn.Module):
    """Fully convolutional PINN; each output field depends on neighboring nodes."""

    def __init__(self, in_channels=2 * dc.K + 2, width=16, n_outputs=2 * (dc.H + 1)):
        super().__init__()
        c1, c2, c3 = width, 2 * width, 4 * width
        self.enc1 = ResidualBlock(in_channels, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.bottleneck = ResidualBlock(c3, c3)
        self.up2 = nn.Conv2d(c3 + c2, c2, 3, padding=1)
        self.dec2 = ResidualBlock(c2, c2)
        self.up1 = nn.Conv2d(c2 + c1, c1, 3, padding=1)
        self.dec1 = ResidualBlock(c1, c1)
        self.head = nn.Conv2d(c1, n_outputs, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear",
                            align_corners=False)
        d2 = self.dec2(F.silu(self.up2(torch.cat((d2, e2), dim=1))))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear",
                            align_corners=False)
        d1 = self.dec1(F.silu(self.up1(torch.cat((d1, e1), dim=1))))
        return self.head(d1)


class AdaptiveWeights:
    """Adaptive loss weighting (learning-rate annealing, Wang et al. 2021):
        lam_k <- lam_k * (|grad data| / |grad term k|) ** beta
    Cập nhật mỗi `freq` iter, giữ EMA: term nào gradient quá lớn bị hạ lam,
    term nào bị lơ là bị nang lam."""

    def __init__(self, net, names=("bc", "res", "cont"), beta=0.15, ema=0.95,
                 freq=20, floor=1e-2, ceiling=50.0):
        self.net = net
        self.names = list(names)
        self.beta, self.ema, self.freq = beta, ema, freq
        self.floor, self.ceiling = floor, ceiling
        self.lam = {n: 1.0 for n in self.names}
        self.ema_g = {}
        self.calls = 0

    def _gnorm(self, loss):
        g = torch.autograd.grad(loss, self.net.parameters(),
                                retain_graph=True, allow_unused=True)
        return float(sum(gg.norm() for gg in g if gg is not None))

    def _g(self, key, loss):
        if loss is None or not torch.is_tensor(loss):
            return
        gn = self._gnorm(loss)
        old = self.ema_g.get(key)
        self.ema_g[key] = gn if old is None else self.ema * old + (1 - self.ema) * gn

    def update(self, data_loss, **terms):
        """Gọi trước loss.backward() khi graph còn sống."""
        self.calls += 1
        if self.calls % self.freq:
            return
        self._g("data", data_loss)
        for n in self.names:
            self._g(n, terms[n])
        gd = self.ema_g.get("data")
        if gd is None:
            return
        for n in self.names:
            g_n = self.ema_g.get(n)
            if g_n is None:
                continue
            self.lam[n] = float(np.clip(
                self.lam[n] * (gd / max(g_n, 1e-12)) ** self.beta,
                self.floor, self.ceiling))


def soft_mask(x, y, r0=0.05, w=None):
    """s(x,y) in [0,1]: 0 trong trụ (u=v=0 chính xác), 1 ngoài, chuyển C1."""
    if w is None:
        w = 2.0 * dc.DX
    r = torch.sqrt(x ** 2 + y ** 2)
    t = ((r - r0) / w).clamp(0, 1)
    return t * t * (3 - 2 * t)


def load_norm_tensors(stats):
    """Return normalized u,v tensors with shape (NT, NY, NX)."""
    Uf, Vf = dc.load_full()
    Ut = torch.tensor((Uf - stats["u_mean"]) / stats["u_std"], dtype=torch.float32)
    Vt = torch.tensor((Vf - stats["v_mean"]) / stats["v_std"], dtype=torch.float32)
    return Ut, Vt


def extract_patch(field, top, left, size, halo, padding_mode="replicate"):
    """Extract a spatial tile with context halo, padding only at domain edges."""
    y0, x0 = top - halo, left - halo
    y1, x1 = top + size + halo, left + size + halo
    crop = field[..., max(0, y0):min(dc.NY, y1), max(0, x0):min(dc.NX, x1)]
    pad = (max(0, -x0), max(0, x1 - dc.NX),
           max(0, -y0), max(0, y1 - dc.NY))
    if any(pad):
        leading_shape = crop.shape[:-2]
        height, width = crop.shape[-2:]
        crop = F.pad(crop.reshape(-1, 1, height, width), pad,
                     mode=padding_mode)
        crop = crop.reshape(*leading_shape, size + 2 * halo, size + 2 * halo)
    return crop


def predict_physical(net, model_input, last_frame, mean, std, softmask):
    """Predict normalized residuals, add the last observed field, and unnormalize."""
    raw = net(model_input).reshape(
        model_input.shape[0], dc.H + 1, 2,
        model_input.shape[-2], model_input.shape[-1])
    normalized = raw + last_frame[:, None]
    return (normalized * std + mean) * softmask


def physics_residual(fields):
    """Full-field vorticity transport and continuity residuals via centered FD."""
    u, v = fields[:, :, 0], fields[:, :, 1]
    omega = (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / (2 * dc.DX) \
          - (u[:, :, 2:, 1:-1] - u[:, :, :-2, 1:-1]) / (2 * dc.DY)
    w0, w1, w2 = omega[:, :-2, 1:-1, 1:-1], \
                 omega[:, 1:-1, 1:-1, 1:-1], \
                 omega[:, 2:, 1:-1, 1:-1]
    wt = (w2 - w0) / (2 * DT)
    wx = (omega[:, 1:-1, 1:-1, 2:] -
          omega[:, 1:-1, 1:-1, :-2]) / (2 * dc.DX)
    wy = (omega[:, 1:-1, 2:, 1:-1] -
          omega[:, 1:-1, :-2, 1:-1]) / (2 * dc.DY)
    lap = (omega[:, 1:-1, 1:-1, 2:] - 2 * w1 +
           omega[:, 1:-1, 1:-1, :-2]) / dc.DX ** 2 \
        + (omega[:, 1:-1, 2:, 1:-1] - 2 * w1 +
           omega[:, 1:-1, :-2, 1:-1]) / dc.DY ** 2
    um, vm = u[:, 1:-1, 2:-2, 2:-2], v[:, 1:-1, 2:-2, 2:-2]
    residual = wt + um * wx + vm * wy - dc.NU * lap
    divergence = (u[:, 1:-1, 2:-2, 3:-1] -
                  u[:, 1:-1, 2:-2, 1:-3]) / (2 * dc.DX) \
               + (v[:, 1:-1, 3:-1, 2:-2] -
                  v[:, 1:-1, 1:-3, 2:-2]) / (2 * dc.DY)
    return residual, divergence


def masked_huber(values, mask, delta=1.0):
    expanded = mask.expand_as(values)
    selected = values[expanded]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return huber(selected, delta)


def get_valid_core():
    """Node collocation hợp lệ: cách solid >= 3 node, xa biên >= 2."""
    m = dc.mask_full()
    bad = m.copy()
    for dy_ in (-3, -2, -1, 0, 1, 2, 3):
        for dx_ in (-3, -2, -1, 0, 1, 2, 3):
            bad |= np.roll(np.roll(m, dy_, axis=0), dx_, axis=1)
    valid = ~bad
    valid[:2, :] = False; valid[-2:, :] = False
    valid[:, :2] = False; valid[:, -2:] = False
    return valid


def compute_scales(stats):
    """Thang residual ground-truth p90 (cache trong train_stats.json)."""
    if "pde_scale_full" in stats:
        return stats["pde_scale_full"], stats["div_scale_full"]
    U, V = dc.load_full()
    valid = get_valid_core()
    fmax = int(stats["train_fmax"])
    fr = np.linspace(dc.TRANSIENT + 3, fmax - 2, 10, dtype=int)
    rs, cs = [], []
    for f in fr:
        r = dc.pde_residual_np(U[f - 1:f + 2], V[f - 1:f + 2], DT, dc.NU,
                               dc.DX, dc.DY)
        c = dc.div_np(U[f], V[f], dc.DX, dc.DY)
        rs.append(np.abs(r[:, valid[2:-2, 2:-2]].ravel()))
        cs.append(np.abs(c[valid[1:-1, 1:-1]].ravel()))
    r = np.concatenate(rs); c = np.concatenate(cs)
    stats["pde_scale_full"] = float(np.percentile(r, 90))
    stats["div_scale_full"] = float(np.percentile(c, 90))
    json.dump(stats, open(dc.STATS_JSON, "w"), indent=2)
    return stats["pde_scale_full"], stats["div_scale_full"]


def huber(x, delta=1.0):
    ax = torch.abs(x)
    return torch.where(ax <= delta, 0.5 * x ** 2, delta * (ax - 0.5 * delta)).mean()


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--pretrain-iters", type=int, default=10000)
    ap.add_argument("--ft-iters", type=int, default=60)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--halo", type=int, default=16)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0: chi chay n window test dau tien (smoke)")
    args = ap.parse_args()
    if args.patch_size % 4 or args.patch_size > min(dc.NY, dc.NX):
        ap.error("--patch-size must be divisible by 4 and fit within the grid")
    if args.halo < 4:
        ap.error("--halo must be at least 4 grid cells")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    if args.fold >= 0:
        sp = dc.make_fold_split(args.fold); tag = f"_fold{args.fold}"
    else:
        sp = dc.make_split(); tag = ""
    if args.limit > 0:
        tag += "_smoke"

    stats = dc.get_train_stats()
    ps, ds_ = compute_scales(stats)
    print(f"pde_scale_full(p90)={ps:.3f}  div_scale_full(p90)={ds_:.4f}")

    mean = torch.tensor([stats["u_mean"], stats["v_mean"]],
                        dtype=torch.float32).view(1, 1, 2, 1, 1)
    std = torch.tensor([stats["u_std"], stats["v_std"]],
                       dtype=torch.float32).view(1, 1, 2, 1, 1)
    fmax = int(stats["train_fmax"])
    Ut, Vt = load_norm_tensors(stats)
    Uf, Vf = dc.load_full()
    xs = dc.X0 + np.arange(dc.NX) * dc.DX
    ys = dc.Y0 + np.arange(dc.NY) * dc.DY
    xg, yg = torch.meshgrid(torch.tensor(xs, dtype=torch.float32),
                            torch.tensor(ys, dtype=torch.float32), indexing="xy")
    coords = torch.stack((2 * (xg - dc.X0) / dc.XW - 1,
                          2 * (yg - dc.Y0) / dc.YW - 1))
    softmask_full = soft_mask(xg, yg).view(1, 1, 1, dc.NY, dc.NX)
    valid_full = torch.tensor(get_valid_core(), dtype=torch.float32).view(
        1, 1, dc.NY, dc.NX)
    core_size, halo = args.patch_size, args.halo
    input_size = core_size + 2 * halo
    if input_size % 4:
        ap.error("patch-size + 2*halo must be divisible by 4")
    net = PINNNet(width=args.width)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"CNN-PINN v5: U-Net patches {input_size}x{input_size}, "
          f"core {core_size} + halo {halo}, width {args.width}, "
          f"params {n_par/1e3:.1f}K")
    opt = torch.optim.Adam(net.parameters(), lr=8e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pretrain_iters,
                                                     eta_min=8e-5)

    K, H = dc.K, dc.H
    s_lo, s_hi = dc.TRANSIENT, fmax - K - H          # include m=H
    aw = AdaptiveWeights(net, names=("bc", "res", "cont"), beta=0.12,
                         ema=0.9, freq=20, floor=0.02, ceiling=20.0)

    def sample_patch():
        top = int(rng.integers(0, dc.NY - core_size + 1))
        if rng.random() < 0.2:
            left = 0
        elif rng.random() < 0.6:
            left = int(rng.integers(0, max(1, dc.NX // 2 - core_size + 1)))
        else:
            left = int(rng.integers(0, dc.NX - core_size + 1))
        return top, left

    def make_input(s, top, left):
        frames = torch.stack((Ut[s:s + K], Vt[s:s + K]), dim=1)
        fields = frames.reshape(1, 2 * K, dc.NY, dc.NX)
        full_input = torch.cat((fields, coords.unsqueeze(0)), dim=1)
        return extract_patch(full_input, top, left, core_size, halo)

    def make_targets(start, count, top, left):
        u = torch.from_numpy(Uf[start:start + count])
        v = torch.from_numpy(Vf[start:start + count])
        physical = torch.stack((u, v), dim=1).unsqueeze(0)
        return extract_patch(physical, top, left, core_size, halo)

    def patch_maps(top, left):
        mask = extract_patch(softmask_full[0, 0], top, left, core_size, halo)
        valid = extract_patch(valid_full, top, left, core_size, halo,
                              padding_mode="constant")
        core = torch.zeros((1, 1, input_size, input_size), dtype=torch.bool)
        core[..., halo:halo + core_size, halo:halo + core_size] = True
        return mask.view(1, 1, 1, input_size, input_size), \
            (valid > 0.5) & core

    def compute_losses(s, top, left, include_data):
        model_input = make_input(s, top, left)
        last = model_input[:, 2 * K - 2:2 * K]
        mask, valid_patch = patch_maps(top, left)
        predicted = predict_physical(net, model_input, last, mean, std, mask)
        core_slice = (..., slice(halo, halo + core_size),
                       slice(halo, halo + core_size))
        pred_core = predicted[core_slice]
        fluid = mask[core_slice] > 0.25
        target = make_targets(s + K - 1, H + 1, top, left)
        target_core = target[core_slice]
        anchor = (((pred_core[:, 0] - target_core[:, 0]) / std[:, 0]) ** 2)
        loss_anchor = anchor[fluid[:, 0].expand_as(anchor)].mean()

        if include_data:
            future_error = ((pred_core[:, 1:] - target_core[:, 1:]) /
                            std) ** 2
            future_mask = fluid.expand_as(future_error)
            loss_data = future_error[future_mask].mean()
        else:
            loss_data = pred_core.sum() * 0.0

        ustd, vstd = float(std[0, 0, 0, 0, 0]), float(std[0, 0, 1, 0, 0])
        bc_terms = []
        future = pred_core[:, 1:]
        if left == 0:
            inlet = future[..., 0]
            bc_terms.append(((inlet[:, :, 0] - dc.U_INF) / ustd).pow(2).mean())
            bc_terms.append((inlet[:, :, 1] / vstd).pow(2).mean())
        if top == 0:
            bc_terms.append((future[:, :, 1, 0, :] / vstd).pow(2).mean())
        if top + core_size == dc.NY:
            bc_terms.append((future[:, :, 1, -1, :] / vstd).pow(2).mean())
        loss_bc = torch.stack(bc_terms).mean() if bc_terms else pred_core.sum() * 0.0

        residual, divergence = physics_residual(predicted)
        valid_res = valid_patch[..., 2:-2, 2:-2]
        loss_res = masked_huber(residual / ps, valid_res)
        loss_cont = masked_huber(divergence / ds_, valid_res)
        return loss_data, loss_anchor, loss_bc, loss_res, loss_cont

    # ---------------- global CNN pretraining ----------------
    for it in range(1, args.pretrain_iters + 1):
        s = int(rng.integers(s_lo, s_hi + 1))
        top, left = sample_patch()
        loss_data, loss_anchor, loss_bc, loss_res, loss_cont = \
            compute_losses(s, top, left, include_data=True)
        loss = loss_data + 0.3 * loss_anchor + aw.lam["bc"] * loss_bc \
             + aw.lam["res"] * loss_res + aw.lam["cont"] * loss_cont
        aw.update(loss_data, bc=loss_bc, res=loss_res, cont=loss_cont)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sch.step()
        if it % 100 == 0 or it == 1:
            print(f"[pre] it {it:5d} loss {loss.item():.4f} data {loss_data.item():.4f} "
                  f"anch {loss_anchor.item():.4f} res {loss_res.item():.4f} | "
                  f"lam bc {aw.lam['bc']:.2f} res {aw.lam['res']:.2f} "
                  f"cont {aw.lam['cont']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(net.state_dict(), os.path.join(OUTM, "pinn_global.pt"))
    print(f"pretrain done {(time.time()-t0)/60:.1f} min, final loss {loss.item():.4f}")

    # ---------------- per-window fine-tune and tiled prediction ----------------
    te = sp["test"]
    if args.limit > 0:
        te = te[:args.limit]
    n_w = len(te)
    preds = np.empty((n_w, 2 * H, dc.NY, dc.NX), np.float32)
    stride = max(1, core_size // 2)
    blend_1d = np.hanning(core_size + 2)[1:-1].astype(np.float32)
    blend = np.outer(blend_1d, blend_1d)

    for w_i, s in enumerate(te):
        if args.ft_iters > 0:
            optw = torch.optim.Adam(net.parameters(), lr=2e-4)
            schw = torch.optim.lr_scheduler.CosineAnnealingLR(
                optw, T_max=args.ft_iters, eta_min=2e-5)
            aw_ft = AdaptiveWeights(net, names=("bc", "res", "cont"),
                                    beta=0.12, ema=0.9,
                                    freq=max(1, args.ft_iters // 2),
                                    floor=0.02, ceiling=20.0)
            for itw in range(args.ft_iters):
                top, left = sample_patch()
                _, loss_anchor, loss_bc, loss_res, loss_cont = \
                    compute_losses(int(s), top, left, include_data=False)
                loss = loss_anchor + aw_ft.lam["bc"] * loss_bc \
                     + aw_ft.lam["res"] * loss_res \
                     + aw_ft.lam["cont"] * loss_cont
                aw_ft.update(loss_anchor, bc=loss_bc,
                             res=loss_res, cont=loss_cont)
                optw.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optw.step(); schw.step()

        with torch.no_grad():
            accumulated = np.zeros((H, 2, dc.NY, dc.NX), dtype=np.float32)
            weight_sum = np.zeros((dc.NY, dc.NX), dtype=np.float32)
            for top in range(0, dc.NY, stride):
                for left in range(0, dc.NX, stride):
                    model_input = make_input(int(s), top, left)
                    last = model_input[:, 2 * K - 2:2 * K]
                    mask, _ = patch_maps(top, left)
                    output = predict_physical(
                        net, model_input, last, mean, std, mask)
                    core = output[0, 1:, :, halo:halo + core_size,
                                  halo:halo + core_size].numpy()
                    height = min(core_size, dc.NY - top)
                    width = min(core_size, dc.NX - left)
                    weights = blend[:height, :width]
                    accumulated[:, :, top:top + height,
                                left:left + width] += (
                                    core[:, :, :height, :width] * weights)
                    weight_sum[top:top + height, left:left + width] += weights
            preds[w_i] = (accumulated / weight_sum[None, None]).reshape(
                2 * H, dc.NY, dc.NX)
        if (w_i + 1) % 25 == 0 or w_i == 0:
            print(f"window {w_i+1}/{n_w} (s={s}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

    trues = np.empty_like(preds)
    for w_i, s_ in enumerate(te):
        for m in range(1, H + 1):
            f_abs = s_ + K - 1 + m
            trues[w_i, 2 * (m - 1)] = Uf[f_abs]
            trues[w_i, 2 * (m - 1) + 1] = Vf[f_abs]
    np.savez_compressed(os.path.join(OUTM, f"preds_pinn{tag}.npz"),
                        preds=preds, trues=trues, starts=te,
                        grid=dict(ny=dc.NY, nx=dc.NX))
    json.dump(dict(pretrain_iters=args.pretrain_iters, ft_iters=args.ft_iters,
                   patch_size=core_size, halo=halo, width=args.width,
                   adaptive_weights=aw.lam, fold=args.fold, version=5,
                   arch="fully convolutional residual U-Net, predicts H+1 normalized fields",
                   t_min=(time.time() - t0) / 60),
              open(os.path.join(OUTM, f"pinn_train_log{tag}.json"), "w"))
    test_mse = float(np.mean((preds - trues) ** 2))
    print(f"test MSE (physical units) = {test_mse:.6e}")
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
