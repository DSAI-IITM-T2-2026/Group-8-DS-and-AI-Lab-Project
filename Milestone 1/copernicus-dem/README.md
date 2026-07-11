# Copernicus DEM Pipeline

End-to-end pipeline for acquiring, processing, and uploading **Copernicus DEM GLO-30** terrain data for the California wildfire study area. Terrain features are fused with satellite imagery and environmental variables in the wildfire early detection model.

> **Data is not stored in this repository.** Processed outputs are on Google Cloud Storage (see [Data location](#data-location)).

## Study area

| Boundary | Value |
|----------|-------|
| North | 42.01° |
| South | 32.53° |
| West | -124.41° |
| East | -114.13° |

**Resolution:** ~30 m (1 arc-second)  
**Temporal coverage:** Static terrain — same dataset applies to all years in the 2021–2025 project window.

## Data location

All processed DEM data is stored on GCS:

```
gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/
```

| Folder | Contents | Use |
|--------|----------|-----|
| `raw/` | 100 original Copernicus 1°×1° tiles | Archive / reprocessing |
| `merged/` | `dem_merged.tif` — seamless elevation mosaic | Intermediate product |
| `clipped/` | `dem_clipped.tif` — elevation clipped to California bbox | Single-band elevation |
| `terrain/` | **ML-ready feature layers** (see below) | **Model training inputs** |
| `metadata/` | `metadata.json`, `dataset_summary.csv`, `terrain_preview.png`, `upload_manifest.json` | QA & documentation |
| `logs/` | `pipeline.log` | Processing log |

### Terrain layers (`terrain/`)

| File | Description |
|------|-------------|
| `elevation.tif` | Height above sea level (m) |
| `slope.tif` | Terrain steepness (degrees) |
| `aspect.tif` | Slope direction (0–360°) |
| `hillshade.tif` | Shaded relief (0–255) |
| `tri.tif` | Terrain Ruggedness Index |
| `tpi.tif` | Topographic Position Index |

**GCS project:** `iitm-dsai-lab`  
**Bucket:** `dsai-lab-project`

### Download from GCS

```bash
# List files
gsutil ls gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/terrain/

# Download terrain layers locally
gsutil -m cp -r \
  gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/terrain/ \
  ./DEM/terrain/
```

## Pipeline overview

```
Configuration → Tile identification → Download → Validate → Merge → Clip
    → Terrain derivatives → QA → Metadata → GCS upload
```

## Setup

```bash
cd "Milestone 1/copernicus-dem"
pip install -r requirements.txt
```

Authenticate for GCS upload (optional):

```bash
gcloud auth application-default login
gcloud config set project iitm-dsai-lab
```

## Usage

### Run full pipeline (download + process)

```bash
python run_pipeline.py
```

Creates local `DEM/` folder with outputs. Use skip flags to resume:

```bash
python run_pipeline.py --skip-download --skip-merge --skip-clip --skip-terrain
```

### Look up terrain at a lat/lon

```bash
python lookup_terrain.py 37.7749 -122.4194
```

```python
from dem_pipeline.lookup import get_terrain_features

features = get_terrain_features(37.7749, -122.4194, config_path="config.yaml")
print(features.to_dict())
```

### Upload to GCS

```bash
# ML-ready outputs (terrain + clipped + metadata)
python upload_to_gcs.py

# Upload remaining raw tiles + merged DEM
python upload_to_gcs.py --remaining

# Upload everything
python upload_to_gcs.py --all

# Preview without uploading
python upload_to_gcs.py --dry-run
```

## Project structure

```
copernicus-dem/
├── config.yaml           # Study area, processing, GCS upload settings
├── requirements.txt
├── run_pipeline.py       # Main pipeline entry point
├── upload_to_gcs.py      # GCS upload script
├── lookup_terrain.py     # Point-based terrain lookup CLI
└── dem_pipeline/
    ├── config.py         # Config loading
    ├── tiles.py          # Tile identification
    ├── download.py       # Parallel download
    ├── validate.py       # File validation
    ├── merge.py          # Tile merging
    ├── clip.py           # Study-area clipping
    ├── terrain.py        # Slope, aspect, hillshade, TRI, TPI
    ├── metadata.py       # metadata.json + dataset_summary.csv
    ├── qa.py             # Quality checks + preview plot
    ├── lookup.py         # Terrain feature lookup API
    ├── gcs_upload.py     # GCS upload helpers
    └── pipeline.py       # Orchestrator
```

## Configuration

Edit `config.yaml` to change study area bounds, worker count, or GCS destination. Local outputs go to `DEM/` (gitignored).

## Integration with wildfire model

Terrain layers from `terrain/` are static inputs that can be stacked with:

- Sentinel-2 multispectral imagery
- Landsat thermal imagery
- ERA5 weather variables
- Sentinel-5P atmospheric observations
- NASA FIRMS fire labels

Use `get_terrain_features(lat, lon)` or read rasters directly from GCS for batch training.
