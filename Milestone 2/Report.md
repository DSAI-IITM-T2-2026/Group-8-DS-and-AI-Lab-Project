# AI Powered Wildfire Early Detection and Alerting System using Multi Source Remote Sensing — Milestone 2

This project builds a per-pixel wildfire risk model for California from six independently sourced geospatial datasets, mirrored into two Google Cloud Storage buckets and accessed anonymously via gcsfs/rasterio (GDAL configured with `GS_NO_SIGN_REQUEST=YES`). All six sources were verified by directly opening sample files rather than assumed from documentation, and the structure below reflects that live verification.

## Datasets Used

Each of the six sources originates from a public earth-observation or reanalysis program; the GCS buckets used in this project are a working mirror assembled for the course project rather than the original distribution point:

- **FIRMS** active fire detections — NASA FIRMS / MODIS active fire product (LANCE / NASA Earthdata).
- **Landsat 8** surface reflectance and thermal bands — USGS/NASA Landsat program.
- **Sentinel-2** multispectral imagery — ESA Copernicus Programme.
- **Sentinel-5P** atmospheric composition (aerosol index) — ESA Copernicus Programme.
- **ERA5** hourly reanalysis — ECMWF / Copernicus Climate Change Service (C3S), Copernicus Climate Data Store.
- **Copernicus DEM** (GLO-30 derived terrain layers) — ESA Copernicus / Copernicus DEM.

## Data Sources

| Source | Bucket / Path | Role |
|--------|---------------|------|
| FIRMS (MODIS Active Fire) | `gs://wildfire-detection-first/firms_daily_geotiff/` | Label (fire / no-fire) |
| Landsat 8 | `gs://dsai-lab-project/wildfire_satellite/raw/landsat/` | Vegetation + thermal input |
| Sentinel-2 | `gs://dsai-lab-project/wildfire_satellite/raw/sentinel2/` | Vegetation / moisture input |
| Sentinel-5P | `gs://dsai-lab-project/wildfire_satellite/raw/sentinel5p/` | Atmospheric / pre-smoke input |
| ERA5 (reanalysis) | `gs://dsai-lab-project/wildfire_satellite/era5/raw/{year}/` | Weather input |
| Copernicus DEM | `gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/terrain/` | Static terrain input |

## Data Structure

| Source | File count | Native resolution | Temporal coverage |
|--------|------------|-------------------|-------------------|
| FIRMS | 3,642 daily GeoTIFFs | ~1 km (1056 × 1153 px, MODIS grid) | 2016-01-01 to 2025-12-31, daily |
| Landsat 8 | 264 tiles found (144 in the earlier progress-file count; 36 tiles for 2024 alone) | ~30 m (10,496 × 10,496 px tiles) | Confirmed present only for 2024–2025 |
| Sentinel-2 | 480 tiles | ~10 m (9,472 × 9,472 px tiles) | 2016–2025 |
| Sentinel-5P | 91 monthly composites | ~1–2 km (1,069 × 1,146 px) | 2018-06 to 2025-12 |
| ERA5 | 12 monthly files/year × 5 years (2021–2025) | ~0.25° (38 × 41 grid, hourly) | 2021–2025, hourly, aggregated to daily |
| Copernicus DEM | 6 precomputed static layers | Matches DEM native tiling | Static (2021–2025 acquisition label) |

## Classes / Categories

The prediction target is a per-pixel binary label derived from FIRMS: **fire (1)** vs. **no-fire (0)**, thresholded on detection confidence ≥ 30.

On a representative fire-season date (2020-08-20):

- Raw (unfiltered) fire-pixel prevalence was **0.41%**
- After the confidence ≥ 30 filter it dropped to **0.177%** (5,045 raw valid pixels vs. 2,155 after filtering)

This confirms a severe class imbalance that motivates a weighted loss (e.g., focal loss or weighted BCE) rather than plain accuracy as the primary training/evaluation metric.

## More about the Data

- **Area of interest:** California, bounding box north 42.01°, south 32.53°, west −124.41°, east −114.13°.
- **Coordinate reference system:** EPSG:4326 across sources (confirmed on Sentinel-2 and Landsat 8 samples).
- **Known open item:** Landsat 8 coverage is confirmed only for 2024–2025 (144–264 tiles depending on listing pass), despite an accompanying progress file claiming coverage since 2016 — flagged for the team rather than assumed away.

## References

1. Copernicus Climate Change Service, Climate Data Store (2023): *ERA5 hourly data on single levels from 1940 to present*. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47) (Accessed on 05-07-2026).

2. Copernicus Digital Elevation Model (DEM) was accessed on 05-07-2026 from [https://registry.opendata.aws/copernicus-dem](https://registry.opendata.aws/copernicus-dem).

3. MODIS Collection 6 NRT Hotspot / Active Fire Detections MCD14DL. Available online: [https://earthdata.nasa.gov/firms](https://earthdata.nasa.gov/firms). DOI: [10.5067/FIRMS/MODIS/MCD14DL.NRT.006](https://doi.org/10.5067/FIRMS/MODIS/MCD14DL.NRT.006).
