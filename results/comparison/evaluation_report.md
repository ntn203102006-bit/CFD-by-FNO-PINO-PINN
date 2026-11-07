# SO SÁNH PINN – FNO – PINO — dự báo tương lai cylinder2d

> Bài toán: X = k=4 frame quá khứ → Y = H=10 frame tương lai (dự báo trực tiếp).
> Split purged 70/15/15 theo thời gian; 181 window test; cùng test set cả 3.
> Bảng chính trên **lưới chung 320x40** (PINN 640x80 được downsample [::2,::2]).
> nu = U·D/Re ≈ 1.220e-03 (Re≈82 — đã kiểm toán).

## Bảng chính (tổng pool test set)
| Model | MSE | RMSE | MAE | R2 u | R2 v | R2 pooled | MSE fluid | R2 fluid |
|---|---|---|---|---|---|---|---|---|
| FNO | 2.3988e-04 | 0.0155 | 0.0106 | 0.991 | 0.991 | 0.999 | 2.3926e-04 | 0.999 |
| PINO | 1.2547e-03 | 0.0354 | 0.0200 | 0.956 | 0.950 | 0.995 | 1.2234e-03 | 0.996 |
| PINN | 1.1252e-03 | 0.0335 | 0.0241 | 0.933 | 0.981 | 0.996 | 1.1265e-03 | 0.996 |

### MSE theo frame Δt (trung bình u,v — lưới chung 320x40)

| Δt s | FNO | PINO | PINN |
|---|---|---|
| 0.01 | 1.9418e-04 | 1.4841e-03 | 2.9026e-04 |
| 0.02 | 1.6614e-04 | 1.4042e-03 | 4.6919e-04 |
| 0.03 | 1.7041e-04 | 1.2577e-03 | 6.1660e-04 |
| 0.04 | 1.9188e-04 | 1.1996e-03 | 8.3685e-04 |
| 0.05 | 2.1267e-04 | 1.1528e-03 | 1.0448e-03 |
| 0.06 | 2.3101e-04 | 1.1488e-03 | 1.2943e-03 |
| 0.07 | 2.6595e-04 | 1.1623e-03 | 1.5244e-03 |
| 0.08 | 2.8437e-04 | 1.1871e-03 | 1.8293e-03 |
| 0.09 | 3.1435e-04 | 1.2368e-03 | 1.9602e-03 |
| 0.10 | 3.6780e-04 | 1.3139e-03 | 1.3856e-03 |

## Hình
- `cmp_perframe_mse.png` — MSE theo frame
- `cmp_snapshots.png` — trường u truth vs từng model
- `cmp_probe.png` — probe wake time-series
- `cmp_bars.png`
- `results/<model>/<model>_preview.png` — cùng test window, lưới và thang màu cho cả ba

## Ghi chú kỹ thuật
- PINN v5: fully convolutional residual U-Net trên các patch không gian có halo; nhận 4 frame quá khứ và dự đoán trực tiếp 11 trường (gồm anchor m=0), adaptive weighting cho biên và residual vật lý.
- FNO: spectral conv (modes 12x20), width 24, supervised, lưới 320x40.
- PINO: cùng backbone FNO — pretrain supervised rồi fine-tune với residual PDE (vận chuyển vorticity, không cần p) + penalty divergence.
- Residual PDE: w_t + u w_x + v w_y = nu·lap(w), nu ≈ 1.220e-03 (Re≈82 — suy từ St=0.157, đã kiểm toán).
- 'fluid' = bỏ node trụ (u=v=0) khi tính.
— Sinh: 2026-11-07 11:53