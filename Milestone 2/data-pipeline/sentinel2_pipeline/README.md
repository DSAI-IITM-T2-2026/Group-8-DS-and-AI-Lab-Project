# Sentinel-2 Standalone Fetch Pipeline

Fully self-contained. This folder has no dependency on anything outside itself —
copy/clone just this directory into its own repo and it runs end to end.

## Structure
```
sentinel2_pipeline/
├── main.py              # CLI entrypoint — run this
├── config.yaml           # all settings live here
├── requirements.txt
├── README.md
└── src/
    ├── config.py         # load/validate config.yaml, build AOI geometry
    ├── ee_client.py       # Earth Engine auth + init
    ├── dates.py           # date-window generator
    ├── masking.py         # cloud masking (both logic variants)
    ├── indices.py         # NDVI/NDMI/NBR/EVI/SAVI (alt_design only)
    ├── composite.py       # filters + masks + builds the composite image
    ├── progress.py        # local JSON progress tracking (+ optional GCS mirror)
    ├── export.py          # submits GEE -> GCS export tasks, reconciles status
    └── pipeline.py         # ties the above together into the main run loop
```
Each module has one job. `main.py` is the only file you run directly.

## What it does
Fetches `COPERNICUS/S2_SR_HARMONIZED` over the California AOI, in 5-day windows,
across a configured date range, cloud-filters and masks each window, builds a
median composite, and exports it as a GeoTIFF to Google Cloud Storage.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edit `config.yaml`:
- `ee_project`: your Google Earth Engine cloud project ID (required; placeholders are rejected)
- `export.gcs_bucket`: your target GCS bucket (must already exist, and your
  GEE service account / user must have write access to it)
- `temporal.start_year` / `end_year`: newest and oldest years respectively
  (`start_year` must be `>= end_year` — the pipeline walks backward)

Authenticate once:
```bash
earthengine authenticate
```
(or just run the script — it will trigger the auth flow automatically on
first run if no credentials are cached).

## Run
```bash
python main.py
```

Check on submitted export tasks and reconcile local progress:
```bash
python main.py --check-tasks
```

### Progress / idempotency
Re-running is safe. Statuses in `sentinel2_progress.json`:

| Status | Meaning | On re-run |
|---|---|---|
| `empty` | No scenes in window | Skipped |
| `submitted` | GEE export started | Skipped (still in flight) |
| `done` | GEE export **completed** | Skipped |
| `failed` | GEE export failed/cancelled | Retried |

`--check-tasks` (and the start of every run) reconciles `submitted` → `done` /
`failed` from the live GEE task list. Progress is also mirrored once per run to
`progress.gcs_metadata_prefix` when GCS credentials allow.

## Logic modes — READ THIS BEFORE USING THE OUTPUT

This pipeline implements **two distinct versions** of the fetch logic. Which
one runs is controlled by `logic_mode` in `config.yaml`, or overridden per-run
with `--logic confirmed` / `--logic alt_design`.

| | `confirmed` (default) | `alt_design` |
|---|---|---|
| Verified against production thresholds? | **Yes** — 40% cloud filter, raw bands, date range | **No** — from original design spec |
| Cloud filter | `CLOUDY_PIXEL_PERCENTAGE < 40` | `< 30` |
| Cloud masking | SCL **and** QA60 (`src/masking.py`) — SCL is required because QA60 is often empty on `S2_SR_HARMONIZED` after ~2022 | SCL only |
| Derived indices | None — raw bands only | NDVI, NDMI, NBR, EVI, SAVI (`src/indices.py`) |
| Output filename | `s2_{date}.tif` | `ALT_DESIGN_UNVERIFIED_s2_{date}.tif` |
| Progress log key | `confirmed:{date}` | `alt_design:{date}` |

**Use `confirmed` unless you specifically want to reproduce/compare against
the original, never-validated design.** Every artifact produced by
`alt_design` mode is filename-tagged and progress-log-tagged so it can never
be silently mistaken for verified production data. The branching between the
two lives entirely in `src/composite.py`, `src/masking.py`, and
`src/indices.py` — that's the only place logic-mode behavior differs.

## Provenance
The `confirmed` logic in this package was reconstructed by directly
inspecting the executed script (`satellite_fetch/scripts/fetch_sentinel2.py`
in the original monorepo), its config (`satellite_fetch/config.yaml`), and
cross-checking submission counts and cloud-masking/threshold values against
real GEE task logs and `gsutil` bucket listings. Date range (2016–2025),
5-day step size, and 40% cloud threshold are confirmed. Per-pixel masking
uses SCL+QA60 so cloud masks remain effective on Collection-1 / HARMONIZED
scenes where QA60 alone is unpopulated.
