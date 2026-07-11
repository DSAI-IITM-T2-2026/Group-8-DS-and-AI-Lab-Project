# Landsat-8/9 Fetch Pipeline

Fetches Landsat 8/9 surface reflectance and thermal bands over the California
AOI via Google Earth Engine, builds cloud-masked median composites, and
exports them as GeoTIFFs to Google Cloud Storage.

## Structure
```
landsat8_download/
├── fetch_landsat.py       # CLI entrypoint — run this to submit GEE exports
├── download_exports.py    # download completed exports from GCS to local disk
├── common.py              # shared helpers (config, EE init, export, progress)
├── config.yaml            # all settings live here
└── README.md
```

## What it does
Fetches `LANDSAT/LC08/C02/T1_L2` and `LANDSAT/LC09/C02/T1_L2` over the
California AOI, in configured date windows, applies QA_PIXEL cloud masking,
builds a median composite, and exports it as a GeoTIFF to GCS.

## Data location

Exported Landsat GeoTIFFs are stored on GCS under:

```
gs://dsai-lab-project/wildfire_satellite/raw/landsat/
```

| Path | Contents |
|------|----------|
| `gs://dsai-lab-project/wildfire_satellite/raw/landsat/` | Cloud-masked median composites |

**GCS project:** `iitm-dsai-lab`  
**Bucket:** `dsai-lab-project`  
**Prefix:** `wildfire_satellite/raw/landsat`

Parent raw folder (all satellite sources):

```
gs://dsai-lab-project/wildfire_satellite/raw/
├── sentinel2/
├── sentinel5p/
└── landsat/
```

### List / download from GCS

```bash
gsutil ls gs://dsai-lab-project/wildfire_satellite/raw/landsat/

# Or use the helper script
python download_exports.py \
  --project iitm-dsai-lab \
  --bucket dsai-lab-project \
  --prefix wildfire_satellite/raw/landsat \
  --dest ./data/wildfire_satellite/raw/landsat
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install earthengine-api google-cloud-storage pyyaml tqdm
```

Edit `config.yaml`:
- `gee.project`: your Google Earth Engine cloud project ID
- `export.gcs_bucket`: your target GCS bucket
- `export.gcs_prefix`: prefix under the bucket (e.g. `wildfire_satellite`)

Authenticate once:
```bash
earthengine authenticate
gcloud auth application-default login
```

## Run
```bash
python fetch_landsat.py
python fetch_landsat.py --check-tasks
```
