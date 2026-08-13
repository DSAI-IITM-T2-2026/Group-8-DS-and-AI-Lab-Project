# `utils/` layout

Internal package for the daily pipeline. Top-level entrypoints (`run_daily.py`, `verify_gcs.py`) add this folder to `sys.path` via `bootstrap.py`.

---

## Top-level files in `utils/`

| File | Purpose |
|------|---------|
| `config.yaml` | **Main knobs:** GCS bucket, prefixes, GEE `project_id`, S2 `grid_asset_id`, lag/lookback. Edit here to retarget environments. |
| `config_loader.py` | Loads `config.yaml`, merges vendored Stage-C config with daily GCS overrides. |
| `paths.py` | Resolves paths relative to `utils/` vs outer `daily_pipeline/` (e.g. `.cache`). |
| `bootstrap.py` | Puts `utils/` + vendored M4 on `sys.path`, loads `.env`, sets `GS_NO_SIGN_REQUEST`. |
| `create_s2_grid_asset.py` | Creates the Sentinel-2 1 km California grid EE asset for a new GEE project. |
| `__init__.py` | Marks `utils` as a package. |

---

## Folders

### `contracts/`

Frozen champion feature contract (`champion_86_features.json`): the locked list of **86** model columns (SHAP keep-list), neighbor-fire flag, and alert blend weights. Used when exporting `final_processed/YYYY-MM-DD_test.parquet` so columns always match training.

### `download/`

Raw data → GCS.

| Module | Role |
|--------|------|
| `firms.py` | Daily FIRMS GeoTIFF → `firms_daily_geotiff/` |
| `sentinel2.py` | Flat S2 window export → `sentinel2/` |
| `sentinel5p.py` | S5P day export + flatten → `sentinel5p/` |
| `era5.py` | Wrapper around `_era5_cds.py` → `era5/YYYY/*.nc` |
| `_era5_cds.py` | CDS NetCDF download + GCS upload |
| `dem.py` | One-shot DEM parquet publish → `dem/` |

### `preprocess/`

GCS raw → Stage C history → 86-feature parquet.

| Module | Role |
|--------|------|
| `adapters_gcs.py` | Flat S2/S5P listing; ERA5 URI resolve (primary + legacy); parquet upload |
| `build_stage_c_day.py` | Runs Stage A/B/C + KNN over lookback window |
| `build_champion_features.py` | Feature engineering (neighbor fire ON) + prune to 86 |
| `export_inference_day.py` | Writes local `.cache/final_processed/` and GCS `final_processed/` |

### `vendor/`

Bundled third-party / milestone code so this folder runs **without** the rest of the repo.

| Subfolder | Role |
|-----------|------|
| `numerical_nextday/` | Stage A/B/C builders |
| `mvp_era5_dem/` | ERA5/FIRMS/DEM helpers + DEM parquet |
| `multimodal_fusion/` | S2/S5P → ERA5-cell aggregation |
| `sentinel2/` | EE S2 export library (`s2_lib`) + AOI geojson |
| `sentinel5p/` | EE S5P day export helper |
| `fire_analysis2.csv` | High/medium fire cell subset (~437 cells) |

Do not edit science here casually — re-sync from milestones if upstream changes.

### `cron/`

| File | Role |
|------|------|
| `run_cron.sh` | `python run_daily.py all --label-date <California today>` (not UTC tomorrow) |

Optional for Cloud Scheduler / VM crontab. Not required if an API triggers `run_daily.py`.

---

## Runtime cache (outside `utils/`)

Created at `Milestone 6/Application/daily_pipeline/.cache/` (gitignored):

- `.cache/m4_shared_cache/` — Stage C intermediates
- `.cache/final_processed/` — local copy of the 86-feature table

---

## Related top-level files

| Path | Role |
|------|------|
| `../run_daily.py` | CLI: download / preprocess / export / all |
| `../verify_gcs.py` | Check bucket prefixes + optional EE grid |
| `../.env.example` | Env var template (copy to `../.env`) |
| `../requirements.txt` | Python deps |
| `../README.md` | Operator guide |
