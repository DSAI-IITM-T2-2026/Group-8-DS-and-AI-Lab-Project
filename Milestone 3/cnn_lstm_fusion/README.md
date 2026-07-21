# CNN + LSTM Fusion (Milestone 3)

Multimodal next-day wildfire model — **separate** from `cnn_s2_mvp`.

| Branch | Input | Default |
|--------|--------|---------|
| CNN | Sentinel-2 patch `[6, 64, 64]` | On |
| LSTM | ERA5 + DEM sequence `[7, 27]` | On |
| S5P MLP | monthly aerosol at cell (day `D`) | Off |

## Split

| Split | Years | Fire season |
|-------|-------|-------------|
| Train | 2022–2023 | May–Nov |
| Val | 2024 | May–Nov |
| Test | 2025 | May–Nov |

## Setup

Use **Python 3.11 or 3.12** (avoid 3.14 — macOS GCS/`gsutil` fork crashes are common).

```bash
cd "Milestone 3/cnn_lstm_fusion"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GS_NO_SIGN_REQUEST=YES
```

`--download-tiles` uses the GCS Python client (not a `gsutil` subprocess) to avoid
macOS `SIGSEGV` / “crashed on child side of fork pre-exec”.

## Pipeline

### 1) Rebuild tabular backbone (2022–2025)

```bash
cd "../mvp_era5_dem"
source ../cnn_lstm_fusion/.venv/bin/activate   # or its own venv
python build_dataset.py --start 2022-05-01 --end 2025-11-30 --fire-season
```

Config defaults are already `train_end=2023-12-31`, `val_end=2024-12-31`.

### 2) S2 patches (optimized: group-by-tile, handle reuse, skip existing)

```bash
cd "../cnn_lstm_fusion"
python build_dataset.py --download-tiles
# smoke: python build_dataset.py --download-tiles --limit 200
```

### 3) ERA5/DEM sequences `[7, F]`

```bash
python build_sequences.py
```

### 4) Train (S5P off)

```bash
python train.py --no-sentinel5p
python map_predictions.py
```

### 5) Optional S5P (same optimizations: group-by-month, reuse, skip)

```yaml
# config.yaml
sources:
  sentinel5p:
    enabled: true
```

```bash
python build_s5p_features.py --download-tiles
python train.py --use-sentinel5p
```

Retrain from scratch when flipping S5P on/off (fusion head width changes).

## Layout

```text
cnn_lstm_fusion/
  build_dataset.py       # S2 patches
  build_sequences.py     # LSTM inputs
  build_s5p_features.py  # optional aerosol
  train.py
  map_predictions.py
  config.yaml
  src/
  outputs/
```

## Relation to other folders

- `../cnn_s2_mvp` — previous CNN+MLP baseline (**do not modify** for this work)
- `../mvp_era5_dem` — shared ERA5/DEM/FIRMS cache + split parquets
