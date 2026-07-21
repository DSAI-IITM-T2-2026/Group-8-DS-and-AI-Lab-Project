# Released checkpoint — CNN + LSTM + S5P (2022–2025)

Trained with `python train.py --use-sentinel5p`.

| File | Role |
|------|------|
| `best.pt` | Model weights (best val PR-AUC) |
| `calibrator.joblib` | Isotonic calibration |
| `seq_norm_stats.npz` | Sequence feature mean/std |
| `s5p_norm_stats.npz` | S5P mean/std |
| `metrics.json` | Val/test ROC & PR |

**Test calibrated (this release):** ROC-AUC ≈ 0.805, PR-AUC ≈ 0.489

To use: copy into `outputs/model/` after building a matching manifest, or load with `torch.load("best.pt")` and the `CNNLSTMFusion` config in the checkpoint.
