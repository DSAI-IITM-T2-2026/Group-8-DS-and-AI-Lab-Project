# ERA5 Wildfire Data Pipeline

End-to-end pipeline for acquiring, processing, and uploading **ECMWF ERA5** reanalysis weather data for the California wildfire study area. Meteorological variables are fused with satellite imagery and terrain features in the wildfire early detection model.

> **Data is not stored in this repository.** Processed outputs are on Google Cloud Storage (see [Data location](#data-location)).

## Study area

| Boundary | Value |
|----------|-------|
| North | 42.01° |
| South | 32.53° |
| West | -124.41° |
| East | -114.13° |

**Dataset:** `reanalysis-era5-single-levels` (hourly)  
**Temporal coverage:** 2016–2025  
**Format:** NetCDF (monthly files)

## Data location

All ERA5 raw monthly NetCDF files are stored on GCS:

```
gs://dsai-lab-project/wildfire_satellite/era5/raw/
```

| Path pattern | Contents |
|--------------|----------|
| `raw/{year}/era5_{year}_{month}.nc` | Monthly NetCDF for each year/month (2016–2025) |

**GCS project:** `iitm-dsai-lab`  
**Bucket:** `dsai-lab-project`  
**Prefix:** `wildfire_satellite/era5/raw`

### Download from GCS

```bash
# List years
gsutil ls gs://dsai-lab-project/wildfire_satellite/era5/raw/

# Download one year locally
gsutil -m cp -r \
  gs://dsai-lab-project/wildfire_satellite/era5/raw/2024/ \
  ./outputs/raw/2024/

# Download all years
gsutil -m cp -r \
  gs://dsai-lab-project/wildfire_satellite/era5/raw/ \
  ./outputs/raw/
```

## Variables

| Variable | Description |
|----------|-------------|
| `2m_temperature` | 2 m air temperature |
| `2m_dewpoint_temperature` | 2 m dewpoint |
| `surface_pressure` | Surface pressure |
| `10m_u_component_of_wind` | 10 m U wind |
| `10m_v_component_of_wind` | 10 m V wind |
| `instantaneous_10m_wind_gust` | 10 m wind gust |
| `total_precipitation` | Total precipitation |
| `volumetric_soil_water_layer_1` | Soil water layer 1 |
| `volumetric_soil_water_layer_2` | Soil water layer 2 |
| `high_vegetation_cover` | High vegetation cover |
| `low_vegetation_cover` | Low vegetation cover |
| `leaf_area_index_high_vegetation` | LAI high vegetation |
| `leaf_area_index_low_vegetation` | LAI low vegetation |
| `boundary_layer_height` | Boundary layer height |

## Pipeline overview

```
Configuration → CDS download (multi-account) → Validate → Monthly merge
    → Feature engineering → Metadata / QA reports → GCS upload
```

## Setup

```bash
pip install -r requirements.txt
```

CDS API credentials (required for download):

```bash
# Option A: ~/.cdsapirc
# Option B: project credentials/ folder
cp credentials/account.example credentials/account2.cdsapirc
# Edit and set: url + key (UID:API_KEY)
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

Run a single phase or year:

```bash
python run_pipeline.py --phase download
python run_pipeline.py --phase merge --year 2024
python run_pipeline.py --phase report
```

### Upload to GCS

```bash
# Upload configured year range (see config.yaml upload.years)
python upload_to_gcs.py

# Preview without uploading
python upload_to_gcs.py --dry-run
```

## Project structure

```
ERA5/
├── config.yaml           # Study area, CDS accounts, GCS upload settings
├── requirements.txt
├── run_pipeline.py       # Main pipeline entry point
├── upload_to_gcs.py      # GCS upload script
├── script.py             # Legacy entry point → run_pipeline.py
├── credentials/
│   └── account.example   # CDS credential template (no secrets)
└── pipeline/
    ├── config.py
    ├── credentials.py
    ├── download_manager.py
    ├── request_builder.py
    ├── work_partition.py
    ├── checkpoint.py
    ├── validator.py
    ├── merger.py
    ├── features.py
    ├── metadata.py
    ├── reports.py
    └── visualization.py
```

Local outputs go to `outputs/` (gitignored). Data for model training should be read from GCS.

## Configuration

Edit `config.yaml` to change years, study area bounds, CDS accounts, or GCS destination.

## Integration with wildfire model

ERA5 weather variables can be stacked with:

- Copernicus DEM terrain layers
- Sentinel-2 multispectral imagery
- Landsat thermal imagery
- Sentinel-5P atmospheric observations
- NASA FIRMS fire labels
