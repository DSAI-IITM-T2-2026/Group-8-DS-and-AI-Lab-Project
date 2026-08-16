# Daily Wildfire Pipeline (self-contained)

Self-contained package under **`Milestone 6/Application/daily_pipeline`**.
Produces the same 86-feature per-day table Milestone 5 scores via `prepared_champion_day`.

---

## Run setup

```bash
cd "Milestone 6/Application/daily_pipeline"
pip install -r requirements.txt
cp .env.example .env          # fill CDS_API_KEY / GOOGLE_CLOUD_PROJECT

python verify_gcs.py --create-missing --skip-ee
python run_daily.py publish_dem
python run_daily.py all --label-date 2026-08-10
```

Folder notes: `[utils/README.md](utils/README.md)`.

### Skip existing vs regenerate final

| Layer | If already in GCS | Flag to force re-download |
| ----- | ----------------- | ------------------------- |
| ERA5 / FIRMS / S2 / S5P raw | **Skipped** (`download.skip_existing: true`) | `--no-skip-existing` |
| `final_processed/…_test.parquet` | Reused only after the full 86-feature/date/cell contract passes | missing or invalid objects are rebuilt |

### Date range

`--start-date` / `--end-date` are **label dates** (inclusive). Labels are capped
at California **tomorrow**, the model's one-day horizon. Tomorrow is
preflight-only: the command reuses a valid final parquet or requires every raw
source object to be present, and returns an `unavailable` pipeline event
without scheduling cloud work when anything is missing. ERA5 uses **daily**
`era5/YYYY/era5_YYYY_MM_DD.nc` when present, or **monthly** under `era5/raw/YYYY/`.

```bash
# Inclusive label dates (each day: download → preprocess → export)
python run_daily.py all --start-date 2026-08-10 --end-date 2026-08-12

# Single day
python run_daily.py all --label-date 2026-08-10

# Raw downloads only for a label range (lookback_days=30 pulls prior history)
python run_daily.py download --start-date 2026-08-10 --end-date 2026-08-12
```

### Timing (how long one day takes)

Logs print per-stage and end-of-day totals. **S5P** is the long pole (~30 min/day on EE). Wait loop polls until parquet exists (`download.s5p_wait: true`).

Cancel stray EE tasks: [Earth Engine Tasks](https://code.earthengine.google.com/tasks).

**S5P recovery after cancelling:** CANCELLED/FAILED are **not** auto-resubmitted. Use `--force-s5p` or `download.s5p_force: true`. Optional: clear `.cache/s5p_ee_tasks.json`.

---

## Per-day `D_test.parquet` (prediction)

Milestone 4/5 date math (LightGBM artifact stays valid):

| Role | Formula | Example D = 2026-08-10 |
|------|---------|-------------------------|
| **label_date** | D | **2026-08-10** → `…/final_processed/2026-08-10_test.parquet` |
| **eo_asof** (S2 / S5P) | D − 1 | **2026-08-09** (one causal snapshot, not a 7-day EO mean) |
| **ERA5 feature_end** | (D−1) − 5 = **D − 6** | **2026-08-04**; `*_7d` ≈ Jul 29–Aug 4 |
| **FIRMS on D** | **not a model input** | `y_fire` column kept as **0** for live forecast |
| **Neighbor `fire_*`** | prior `y_fire`, lag2 through D−2 | FIRMS downloaded through **D−1** only |

Export asserts: 86 features, single `label_date`, `eo_asof = D−1`, `feature_end = D−6`, ~437 high/medium cells.

With `lookback_days: 30` for D = 2026-08-10 … 2026-08-12:

- FIRMS history: **2026-07-11 … 2026-08-11** (not the predict day)
- EO/S5P: **2026-07-10 … 2026-08-11**
- ERA5: **2026-06-28 … 2026-08-06**

**Sentinel-2:** only unique 5-day windows on the 2018-01-01 grid; windows with `start > today` are **skipped** (fixes accidental Aug 16–31 submits).

---

## Demo download (July–August 2026)

Do **not** request labels through 31 Aug. Cap at tomorrow (or the last day you
will score); tomorrow succeeds only when its complete causal source set already
exists.

### Recommended: few live days (start ASAP)

```bash
cd "Milestone 6/Application/daily_pipeline"
python run_daily.py download --start-date 2026-08-10 --end-date 2026-08-12
python run_daily.py all --label-date 2026-08-10
```

S5P ≈ 33 eo_asof days (Jul 10–Aug 11) is the long pole.

### Optional: any day in July + August so far

```bash
python run_daily.py download --start-date 2026-07-01 --end-date 2026-08-12
```

~73 S5P days — start early; top up before demo:

```bash
python run_daily.py download --start-date 2026-08-12 --end-date 2026-08-12
```

### Zero-download 2025 replay

Historical 86-feature panel (schema match):

```bash
gsutil cp gs://wildfire-detection-first/final_processed/2019_2025/2019-2025.parquet /tmp/
```

Use Milestone 5 `Wildfire_Inference.ipynb` with `INFERENCE_INPUT_KIND=prepared_champion_day`.

---

## Local output

Created under `Milestone 6/Application/daily_pipeline/.cache/`:

| What                       | Local path                                                          |
| -------------------------- | ------------------------------------------------------------------- |
| **Final 86-feature table** | `.cache/final_processed/YYYY-MM-DD_test.parquet`                    |
| Stage C / KNN work         | `.cache/m4_shared_cache/`                                           |
| Stage C one day            | `.cache/m4_shared_cache/stage_c_knn/day=YYYY-MM-DD/stage_c.parquet` |

```bash
python run_daily.py export_day --label-date 2026-08-10 --local-only
```

---

## What goes into GCS (paths)

| What                       | GCS path                                                   | Format        |
| -------------------------- | ---------------------------------------------------------- | ------------- |
| FIRMS                      | `gs://<bucket>/firms_daily_geotiff/YYYY-MM-DD.tif`         | GeoTIFF       |
| Sentinel-2                 | `gs://<bucket>/sentinel2/s2feat_YYYYMMDD_YYYYMMDD.parquet` | Parquet / CSV |
| Sentinel-5P                | `gs://<bucket>/sentinel5p/s5pfeat_YYYYMMDD_YYYYMMDD.parquet` | Parquet     |
| ERA5                       | `gs://<bucket>/era5/YYYY/era5_YYYY_MM_DD.nc`               | NetCDF        |
| DEM                        | `gs://<bucket>/dem/era5_grid_dem_features.parquet`         | Parquet       |
| **Final (app reads this)** | `gs://<bucket>/final_processed/YYYY-MM-DD_test.parquet`    | Parquet       |

Example: `gs://wildfire-detection-first/final_processed/2026-08-10_test.parquet`

---

## How to change bucket / GEE project / grid / cells

**Single config file:** `[utils/config.yaml](utils/config.yaml)`

| What                       | Key(s)                                       |
| -------------------------- | -------------------------------------------- |
| GCS bucket                 | `gcs.bucket`                                 |
| Folder names under bucket  | `gcs.prefixes.*`                             |
| FIRMS `/vsigs/` mount      | `gcs.firms_vsigs_prefix` (must match bucket) |
| GEE project                | `gee.project_id` **and** `gcs.project`       |
| S2 1 km grid asset         | `gee.grid_asset_id`                          |
| **Cells in final parquet** | `preprocess.cell_subset`                     |
| Feature lookback           | `task.lookback_days` (default **30**)        |

### Cell subset (`preprocess.cell_subset`)

| Value                        | What you get                           | Approx. rows / day |
| ---------------------------- | -------------------------------------- | ------------------ |
| `high_medium_fire` (default) | High Outlier + High + Medium           | **~437–439**       |
| `high_only`                  | High Outlier + High only               | **~146**           |
| `all`                        | No fire-region filter                  | **up to ~672**     |

---

## Layout

```text
Milestone 6/Application/daily_pipeline/
  README.md
  requirements.txt
  .env.example
  run_daily.py
  verify_gcs.py
  utils/          ← config, download, preprocess, vendor, cron
  .cache/         ← local processed work
```

---

## Auth

```bash
cp .env.example .env   # CDS_API_KEY / GOOGLE_CLOUD_PROJECT / SA path
```

- GEE: `utils/config.yaml` → `gee.project_id`
- CDS: `.env` or `~/.cdsapirc`
- GCS: write access to `gcs.bucket`

**Cron (California today, 06:30 Pacific):** `./utils/cron/run_cron.sh`

- Label date = **today in `America/Los_Angeles`** (override with `LABEL_DATE=YYYY-MM-DD` or `PIPELINE_TZ=...`).
- A valid existing `final_processed/YYYY-MM-DD_test.parquet` is reused immediately. If it is missing or invalid, required raw GCS objects are inventoried, existing inputs are skipped, missing inputs are fetched, and the final parquet is regenerated.
- Overlapping runs exit 0. Logs: `daily_pipeline/logs/YYYY-MM-DD.log`.
- VM must have ADC / `GOOGLE_APPLICATION_CREDENTIALS` and a **persistent** `.cache` (S2 CSVs are ~300 MB each). Do not use interactive `ee.Authenticate()` on cron.
