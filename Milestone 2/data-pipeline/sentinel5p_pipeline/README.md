# Sentinel-5P (TROPOMI) Standalone Fetch Pipeline

Fully self-contained. This folder has no dependency on anything outside
itself — including the `sentinel2_pipeline` folder — so copy/clone just this
directory into its own repo and it runs end to end.

## Structure
```
sentinel5p_pipeline/
├── main.py              # CLI entrypoint — run this
├── config.yaml           # all settings live here
├── requirements.txt
├── README.md
└── src/
    ├── config.py         # load/validate config.yaml, build AOI, pick active products
    ├── ee_client.py       # Earth Engine auth + init
    ├── dates.py           # date-window generator
    ├── composite.py       # per-product fetch + nodata-fallback + band stacking
    ├── progress.py        # local JSON progress tracking (+ optional GCS mirror)
    ├── export.py          # submits GEE -> GCS export tasks, reconciles status
    └── pipeline.py         # ties the above together into the main run loop
```
Each module has one job. `main.py` is the only file you run directly.

## What it does
Fetches Sentinel-5P TROPOMI products (Aerosol Index, CO, and optionally NO2)
over the California AOI, in 5-day windows, across a configured date range,
composites each window per-band with a `mean` reducer, stacks the bands into
one multi-band image, and exports it as a GeoTIFF to Google Cloud Storage.

Missing product bands in a partial window are written as **masked nodata**
(not numeric zeros).

## Data location

Exported Sentinel-5P GeoTIFFs are stored on GCS under:

```
gs://dsai-lab-project/wildfire_satellite/raw/sentinel5p/
```

| Path | Contents |
|------|----------|
| `gs://dsai-lab-project/wildfire_satellite/raw/sentinel5p/` | Mean composites (`s5p_{date}.tif`) — AerosolIndex, CO |

**GCS project:** `iitm-dsai-lab`  
**Bucket:** `dsai-lab-project`  
**Prefix:** `wildfire_satellite/raw/sentinel5p`

Parent raw folder (all satellite sources):

```
gs://dsai-lab-project/wildfire_satellite/raw/
├── sentinel2/
├── sentinel5p/
└── landsat/
```

### List / download from GCS

```bash
gsutil ls gs://dsai-lab-project/wildfire_satellite/raw/sentinel5p/

gsutil -m cp -r \
  gs://dsai-lab-project/wildfire_satellite/raw/sentinel5p/ \
  ./data/raw/sentinel5p/
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edit `config.yaml`:
- `ee_project`: your Google Earth Engine cloud project ID (required; placeholders are rejected)
- `export.gcs_bucket`: your target GCS bucket
- `temporal.start_year` / `end_year`: newest and oldest years respectively
  (`start_year` must be `>= end_year` — the pipeline walks backward)

Authenticate once:
```bash
earthengine authenticate
```
(or just run the script — it triggers the auth flow automatically if no
credentials are cached).

## Run
```bash
python main.py
```

Check on submitted export tasks and reconcile local progress:
```bash
python main.py --check-tasks
```

### Progress / idempotency
Re-running is safe. Statuses in `sentinel5p_progress.json`:

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

| | `confirmed` (default) | `alt_design` |
|---|---|---|
| Verified to have actually run in production? | **Yes** — reconciled against real GEE task logs and GCS bucket contents | **No** — from the original design spec, never executed/verified |
| Bands exported | AerosolIndex, CO | AerosolIndex, CO, **NO2** |
| Output filename | `s5p_{date}.tif` | `ALT_DESIGN_UNVERIFIED_s5p_{date}.tif` |
| Progress log key | `confirmed:{date}` | `alt_design:{date}` |

**Use `confirmed` unless you specifically want to reproduce/compare against
the original, never-validated design** — which additionally fetches NO2, a
band confirmed absent from the actual production run. Which products are
active for a given mode is decided in one place: `get_active_products()` in
`src/config.py`.

## Expected empty windows — not a bug
TROPOMI launched in October 2017 and wasn't fully operational until roughly
mid-2018. Any window before that will legitimately have zero scenes for
every product and will be logged as `empty`. This was directly confirmed
against production logs: 100% empty windows Jan 2016–May 2018, a transition
in June 2018, and 0% empty from July 2018 onward. If a run shows a different
pattern than this, that's worth investigating — this pattern itself is
expected.

## Provenance
The `confirmed` logic in this package was reconstructed by directly
inspecting the executed script (`satellite_fetch/scripts/fetch_sentinel5p.py`
in the original monorepo) and its config, and cross-checking export
filenames, band lists, and submission counts against real GEE task logs and
`gsutil` bucket listings. The absence of NO2, the 5-day step size, the
`mean` composite, and the 1000m export scale are all confirmed, not assumed.
