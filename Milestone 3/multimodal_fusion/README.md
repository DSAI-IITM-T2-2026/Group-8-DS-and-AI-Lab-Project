# Multimodal Fusion (Milestone 3)

Full hybrid next-day California wildfire risk model. Does **not** modify `cnn_lstm_fusion`.

| Branch | Input | Source |
|--------|--------|--------|
| **S2 CNN** | `6×64×64` patch | Monthly S2 mosaics (`dsai-lab-project`) |
| **S5P CNN** | `2×64×64` patch | Monthly S5P mosaics (`dsai-lab-project`) |
| **LSTM** | `[7, 27]` | ERA5 + DEM sequences |
| **S2 MLP** | 19-d spectral / index vector | 5-day numerical `sentinel2_features_v3` |
| **S5P MLP** | 9-d AAI / CO vector | numerical `sentinel5p_features_daily` |

**Honest wording:** image patches are **monthly** mosaics; the **5-day cadence** comes from S2 numerical tables (not 5-day GeoTIFF images). S5P numerical coverage is mostly **2025** in the current buckets.

Split: train **2022–2023** / val **2024** / test **2025** (fire season May–Nov).

Architecture & preprocessing diagrams: [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §11.  
**Full experiment report:** [`REPORT.md`](REPORT.md).

---

## Released checkpoint (use without retraining)

`artifacts/multimodal_full_2022_2025/`

| File | Role |
|------|------|
| `best.pt` | Weights (best **val PR-AUC**) + flags / column lists / dims |
| `calibrator.joblib` | Isotonic calibration (val → confidence %) |
| `seq_norm_stats.npz` | ERA5+DEM sequence mean / std |
| `s2_num_norm.npz` | S2 numerical mean / std / columns |
| `s5p_num_norm.npz` | S5P numerical mean / std / columns |
| `metrics.json` | Val / test ROC & PR (raw + calibrated) |

**Test calibrated (this release):** ROC-AUC ≈ **0.831**, PR-AUC ≈ **0.531**  
(Best val PR ≈ 0.569 / ROC ≈ 0.832 at epoch 2.)

**Load weights:**

```python
import joblib
import torch
from src.config import load_config
from src.model import MultimodalFusion

cfg = load_config()
mcfg = cfg["model"]
ckpt = torch.load(
    "artifacts/multimodal_full_2022_2025/best.pt",
    map_location="cpu",
    weights_only=False,
)
flags = ckpt["flags"]
model = MultimodalFusion(
    seq_dim=ckpt["seq_dim"],
    s2_num_dim=len(ckpt["s2_num_cols"]),
    s5p_num_dim=len(ckpt["s5p_num_cols"]),
    s2_in_ch=int(cfg["patch"]["bands"]),
    s5p_in_ch=int(cfg["patch_s5p"]["bands"]),
    cnn_embed=int(mcfg["cnn_embed_dim"]),
    s5p_cnn_embed=int(mcfg["s5p_cnn_embed_dim"]),
    lstm_embed=int(mcfg["lstm_embed_dim"]),
    lstm_hidden=int(mcfg["lstm_hidden"]),
    s2_num_embed=int(mcfg["s2_num_embed_dim"]),
    s5p_num_embed=int(mcfg["s5p_num_embed_dim"]),
    dropout=float(mcfg["dropout"]),
    **flags,
)
model.load_state_dict(ckpt["model_state"])
model.eval()
calibrator = joblib.load("artifacts/multimodal_full_2022_2025/calibrator.joblib")
# logits = model(seq=..., s2_image=..., s5p_image=..., s2_num=..., s5p_num=...)
# p_cal = calibrator.predict(torch.sigmoid(logits).numpy())
```

To score a full test set: rebuild inputs (pipeline below), copy artifact files into `outputs/model/`, then run `map_predictions.py`.

---

## Setup

Use **Python 3.11 or 3.12**.

```bash
cd "Milestone 3/multimodal_fusion"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GS_NO_SIGN_REQUEST=YES
# Numerical feature buckets may need:
# gcloud auth application-default login
```

Prereq — tabular backbone:

```bash
cd ../mvp_era5_dem
python build_dataset.py --start 2022-05-01 --end 2025-11-30 --fire-season
```

**Reuse tip:** you can symlink `outputs/patches` and `outputs/sequences` from `cnn_lstm_fusion` if already built for the same split, then only run S5P patches + numerical features here.

---

## Pipeline

```bash
cd ../multimodal_fusion
export GS_NO_SIGN_REQUEST=YES

# 1) S2 image patches
python build_dataset.py --download-tiles

# 2) S5P image patches (2×64×64)
python build_s5p_patches.py --download-tiles

# 3) ERA5/DEM sequences
python build_sequences.py

# 4) 5-day S2 + S5P numerical features
python build_numerical_features.py
# smoke: python build_numerical_features.py --limit-windows 2

# 5) Train + maps
python train.py
python map_predictions.py
```

Branch toggles under `model:` in `config.yaml`:
`use_s2_patches`, `use_s5p_patches`, `use_s2_numerical`, `use_s5p_numerical`.

---

## GCS sources

| Modality | Path |
|----------|------|
| ERA5 | `gs://dsai-lab-project/wildfire_satellite/era5/raw/` |
| S2 mosaics | `gs://dsai-lab-project/wildfire_satellite/raw/sentinel2/` |
| S5P mosaics | `gs://dsai-lab-project/wildfire_satellite/raw/sentinel5p/` |
| FIRMS | `gs://wildfire-detection-first/firms_daily_geotiff/` |
| S2 numerical (5-day) | `gs://sentinel-2-data-2016-2025/sentinel2_features_v3/` |
| S5P numerical | `gs://sentinel-2-2016-2025/sentinel5p_features_daily/` |

Shipped small tables (in git): `data/california.geojson`, `data/era5_to_feature_grid.parquet`.

**Not in git:** `outputs/` (patches, sequences, numerical cache — multi-GB).

---

## Layout

```text
multimodal_fusion/
  build_dataset.py              # S2 patches
  build_s5p_patches.py          # S5P patches
  build_sequences.py
  build_numerical_features.py   # S2/S5P tabular features
  train.py
  map_predictions.py
  config.yaml
  src/
  data/                         # geojson + ERA5↔feature grid
  artifacts/                    # released weights (tracked)
  outputs/                      # gitignored rebuild cache
```
