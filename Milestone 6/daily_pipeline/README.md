# Daily Wildfire Pipeline (self-contained)

---

## Run setup

```bash
cd ops/daily_pipeline
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
| `final_processed/…_test.parquet` | **Always rebuilt** (processing may change) | n/a — always overwrite |

### Date range

`--start-date` / `--end-date` are **label dates** (inclusive). The pipeline downloads
only the decision/ERA5 days needed for that range (plus a short lookback for rolling
features). ERA5 uses **daily** `era5/YYYY/era5_YYYY_MM_DD.nc` when present, or
**monthly** under `era5/raw/YYYY/` if that already covers the month.

```bash
# Inclusive label dates (each day: download → preprocess → export)
python run_daily.py all --start-date 2026-08-01 --end-date 2026-08-10

# Single day
python run_daily.py all --label-date 2026-08-10

# Raw downloads only for a label range
python run_daily.py download --start-date 2026-08-01 --end-date 2026-08-10
```

### Timing (how long one day takes)

Logs print per-stage and end-of-day totals, e.g.:

```text
Timing ERA5: …
Timing FIRMS: …
Timing S2: …
Timing S5P: …
Timing download total: …
Timing preprocess: …
Timing export_day: …
=== Timing summary label_date=… | download=… | preprocess=… | export=… | TOTAL=… ===
```

**S5P note:** first-time S5P starts an Earth Engine export and returns before the CSV is ready. Wall time for that stage is only “submit”; wait until the EE task is **COMPLETED**, then re-run `download` / `all` (skip will pick up the file and finish parquet). Use the Timing summary line after a full successful day (raw already present) for a fair “download→final” estimate.

Cancel stray EE tasks from a bad run (e.g. year=2019 flood): [Earth Engine Tasks](https://code.earthengine.google.com/tasks).

---

## Local output

Created under `ops/daily_pipeline/.cache/` while the pipeline runs:


| What                       | Local path                                                          |
| -------------------------- | ------------------------------------------------------------------- |
| **Final 86-feature table** | `.cache/final_processed/YYYY-MM-DD_test.parquet`                    |
| Stage C / KNN work         | `.cache/m4_shared_cache/`                                           |
| Stage C one day            | `.cache/m4_shared_cache/stage_c_knn/day=YYYY-MM-DD/stage_c.parquet` |


Skip uploading the final table to GCS (keep local only):

```bash
python run_daily.py export_day --label-date 2026-08-10 --local-only
```

---

## What goes into GCS (paths)

Raw downloads + final processed table land on the bucket from `utils/config.yaml` → `gcs.bucket` (default `wildfire-detection-first`):


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

## GCS run and outputs

On a GCP VM / Cloud Run Job / API host (same repo folder + credentials):

```bash
python run_daily.py all --label-date YYYY-MM-DD
# or
./utils/cron/run_cron.sh
```


| Output                | Where                                                   |
| --------------------- | ------------------------------------------------------- |
| App deliverable       | `gs://<bucket>/final_processed/YYYY-MM-DD_test.parquet` |
| Raw layers            | same GCS paths as above                                 |
| Optional local mirror | `.cache/final_processed/` on the machine                |


```text
Cloud Scheduler OR App API
        │
        ▼
  run_daily.py all --label-date ...
        │
        ▼
  gs://.../final_processed/YYYY-MM-DD_test.parquet
```

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


### Bucket + project example

```yaml
gcs:
  bucket: my-new-bucket
  project: my-gee-project
  firms_vsigs_prefix: /vsigs/my-new-bucket/firms_daily_geotiff

gee:
  project_id: my-gee-project
  grid_asset_id: projects/my-gee-project/assets/california_s2_grid_1km_v3
```

### Cell subset (`preprocess.cell_subset`)


| Value                        | What you get                           | Approx. rows / day |
| ---------------------------- | -------------------------------------- | ------------------ |
| `high_medium_fire` (default) | High Outlier + High + Medium           | **~439**           |
| `high_only`                  | High Outlier + High only               | **~146**           |
| `all`                        | No fire-region filter (full day panel) | **up to ~672**     |


```yaml
preprocess:
  cell_subset: high_medium_fire   # or high_only  or  all
```

Categories from `utils/vendor/fire_analysis2.csv`. Champion training used **high + medium**.

---

## How to create S2 grid asset

Use when you switch to a **new GEE project** and do not already have a shared 1 km California grid:

```bash
python utils/create_s2_grid_asset.py --project my-gee-project --dry-run
python utils/create_s2_grid_asset.py --project my-gee-project
```

When the Earth Engine task is **COMPLETED** (~413k cells; can take a long time):

1. Set `gee.grid_asset_id` in `utils/config.yaml` to the printed id
2. Run `python verify_gcs.py`

Reuse an existing shared asset if you already have one — do not recreate casually.

---

## Layout

```text
ops/daily_pipeline/
  README.md
  requirements.txt
  .env.example
  run_daily.py
  verify_gcs.py
  utils/          ← config, download, preprocess, vendor, cron
  .cache/         ← local processed work
```


| Path                            | Role                                 |
| ------------------------------- | ------------------------------------ |
| `utils/config.yaml`             | Bucket, project, grid, cell_subset   |
| `utils/README.md`               | Folder-by-folder internals           |
| `utils/create_s2_grid_asset.py` | Create S2 grid for a new GEE project |
| `utils/contracts/`              | Frozen 86 features                   |
| `utils/download/`               | Raw → GCS                            |
| `utils/preprocess/`             | → final_processed                    |
| `utils/vendor/`                 | Bundled Stage C / DEM / S2 / S5P     |
| `utils/cron/run_cron.sh`        | Daily wrapper                        |


---

## Auth

```bash
cp .env.example .env   # CDS_API_KEY / GOOGLE_CLOUD_PROJECT / SA path
```

- GEE: `utils/config.yaml` → `gee.project_id`
- CDS: `.env` or `~/.cdsapirc`
- GCS: write access to `gcs.bucket`

---

## Example: one label day (cron / `all`)

Milestone 4 date rules (predict day **D**):

| Role | Formula | Example D = 2026-08-10 |
|------|---------|-------------------------|
| **label_date** | D | **2026-08-10** → `…/final_processed/2026-08-10_test.parquet` |
| **eo_asof** (S2 / S5P) | D − 1 | **2026-08-09** |
| **ERA5 feature_end** | D − (lag+lead) = D − 6 | **2026-08-04** |
| **ERA5 7d history** | ending at feature_end | **~2026-07-29 … 2026-08-04** |
| **FIRMS `y_fire`** | on D (+ lookback labels) | **2026-08-10** (and prior labels if `lookback_days>0`) |

```bash
python run_daily.py all --label-date 2026-08-10
# cron uses tomorrow: ./utils/cron/run_cron.sh
```

With `lookback_days: 7`, downloads also cover prior labels for feature engineering:

- EO/S5P eo_asof: **2026-08-02 … 2026-08-09**
- FIRMS labels: **2026-08-03 … 2026-08-10**
- ERA5: **2026-07-21 … 2026-08-04**

`*_test.parquet` still has **one row set for D only** (~439 cells × 86 features). S2/S5P in that file are a **single causal snapshot as of D−1** (not a 7-day S2/S5P average). ERA5 `*_7d` rolls use the 7 weather days. Neighbor `fire_*_lag2` uses prior FIRMS labels through D−2.

**S5P recovery after cancelling EE tasks:** CANCELLED/FAILED tasks are not auto-resubmitted. Use:

```bash
python run_daily.py all --label-date 2026-08-10 --force-s5p
```

Or set `download.s5p_force: true`. Optional: delete `.cache/s5p_ee_tasks.json` entries for those days.

**Cron at 06:00 IST:** `run_cron.sh` sets `label_date = tomorrow`, builds that day’s `*_test.parquet`. Existing raw GCS objects are skipped; the final parquet is always regenerated.