# Copernicus DEM — California EDA

Exploratory analysis of **Copernicus DEM GLO-30** clipped to the **California state polygon** (not the rectangular bbox), with optional fusion planning against ERA5 grid cells.

Related pipeline code: [`../copernicus-dem-30m/`](../copernicus-dem-30m/)

## Data location (GeoTIFFs on GCS)

Clipped California terrain layers are **not stored in Git**. They live on GCS:

```
gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/eda/clipped_ca/
```

| File | Description |
|------|-------------|
| `elevation_ca.tif` | Elevation (m), CA-masked |
| `slope_ca.tif` | Slope (degrees) |
| `aspect_ca.tif` | Aspect (degrees) |
| `hillshade_ca.tif` | Hillshade |
| `tri_ca.tif` | Terrain Ruggedness Index |
| `tpi_ca.tif` | Topographic Position Index |

**GCS project:** `iitm-dsai-lab`  
**Bucket:** `dsai-lab-project`

### Download clipped layers

```bash
mkdir -p outputs/clipped_ca
gsutil -m cp \
  "gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/eda/clipped_ca/*.tif" \
  ./outputs/clipped_ca/
```

Upstream full-resolution terrain (pre-EDA) remains at:

```
gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/terrain/
```

## Pipeline

Requires local DEM terrain inputs (see `config.yaml` → `paths.terrain_dir`), typically from [`copernicus-dem-30m`](../copernicus-dem-30m/) or GCS `terrain/`.

```bash
# from this directory
python scripts/run_eda.py
# or step-by-step:
python scripts/01_clip_to_california.py
python scripts/02_extract_numerical_values.py
python scripts/03_era5_fusion_analysis.py
```

## Jupyter notebook

```bash
jupyter notebook california_dem_eda.ipynb
```

Covers summary stats, histograms/KDEs/correlations, spatial maps, DEM sampled on the ERA5 0.25° grid, and fusion patterns.

## Repo layout

```
copernicus-dem-eda/
├── config.yaml
├── california_dem_eda.ipynb
├── boundaries/              # CA polygon + TIGER shapefile
├── scripts/
│   ├── run_eda.py
│   ├── 01_clip_to_california.py
│   ├── 02_extract_numerical_values.py
│   └── 03_era5_fusion_analysis.py
└── outputs/
    ├── clipped_ca/          # boundary geojson only; *.tif on GCS (see above)
    ├── numerical/           # CSV / parquet summaries & samples
    ├── figures/
    └── reports/
```

## Fusion idea (short)

DEM is **static**. ERA5 is **time series**. Sample DEM once on each ERA5 grid cell, then join those columns to every `(time, lat, lon)` ERA5 row.
