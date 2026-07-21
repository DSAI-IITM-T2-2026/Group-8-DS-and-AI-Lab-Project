# Released checkpoint — MultimodalFusion (2022–2025)

All branches on: S2 CNN + S5P CNN + ERA5 LSTM + S2/S5P numerical MLPs.

Trained with `python train.py` (best **val PR-AUC** at epoch 2).

| File | Role |
|------|------|
| `best.pt` | Model weights + flags / column lists / `seq_dim` |
| `calibrator.joblib` | Isotonic calibration |
| `seq_norm_stats.npz` | Sequence feature mean/std |
| `s2_num_norm.npz` | S2 numerical mean/std/columns |
| `s5p_num_norm.npz` | S5P numerical mean/std/columns |
| `metrics.json` | Val/test ROC & PR |

### Metrics (this release)

| Split | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Val (raw, best ckpt) | 0.832 | 0.569 |
| Test calibrated | 0.831 | 0.531 |

Load via `MultimodalFusion` + `config.yaml` — see parent [`README.md`](../README.md).
