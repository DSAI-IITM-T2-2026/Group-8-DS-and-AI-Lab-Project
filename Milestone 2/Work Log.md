# Work Logs - Milestone 2

## Ripunjay Kumar (Roll Number: 21F3002511)

### Copernicus DEM GLO-30 data pipeline
- Designed and implemented an end-to-end Copernicus DEM GLO-30 pipeline for the California wildfire study area (N 42.01°, S 32.53°, W −124.41°, E −114.13°).
- Built tile identification, parallel download, validation, merge, and study-area clip stages to produce a seamless elevation mosaic from ~100 original 1°×1° Copernicus tiles.
- Derived ML-ready terrain feature layers from the clipped DEM: elevation, slope, aspect, hillshade, Terrain Ruggedness Index (TRI), and Topographic Position Index (TPI).
- Added QA checks, metadata generation (`metadata.json`, `dataset_summary.csv`), and a terrain preview for documentation and reproducibility.
- Implemented GCS upload tooling and published processed DEM outputs to `gs://dsai-lab-project/wildfire_satellite/dem/2021-2025/california/` (project `iitm-dsai-lab`, bucket `dsai-lab-project`).
- Documented setup, usage, and data location in the pipeline README; kept large raster data out of Git and referenced GCS instead.
- Organized the code under `Milestone 2/data-pipeline/copernicus-dem-30m/` in the group GitHub repository.

### ERA5 reanalysis data pipeline
- Designed and implemented an end-to-end ECMWF ERA5 (`reanalysis-era5-single-levels`) pipeline for the same California bounding box, covering hourly weather variables for 2016–2025.
- Configured multi-account CDS downloads with work partitioning, checkpointing, retries, and queue-limit handling to reliably pull large NetCDF request volumes.
- Selected and downloaded wildfire-relevant variables including 2 m temperature/dewpoint, surface pressure, 10 m wind components and gust, precipitation, soil water, vegetation cover/LAI, and boundary layer height.
- Built validation, monthly merge, feature-engineering, metadata, and reporting stages on top of the raw downloads.
- Implemented GCS upload with skip-existing / manifest tracking and published monthly NetCDF files to `gs://dsai-lab-project/wildfire_satellite/era5/raw/` (`raw/{year}/era5_{year}_{month}.nc`).
- Documented the pipeline, variables, and GCS data location in the ERA5 README; excluded local data, secrets, and planning docs from version control.
- Organized the code under `Milestone 2/data-pipeline/ERA5/` and updated the repository README to link both Milestone 2 data pipelines.

## Roushan Kumar Singh (Roll Number: 23F1002240)

### Sentinel-2
- Built a data fetch pipeline for Sentinel-2 imagery over the target AOI.
- Fetched the data using Google Earth Engine and exported it to GCS bucket.

### Sentinel-5P
- Built a data fetch pipeline for Sentinel-5P (TROPOMI) imagery over the same AOI.
- Fetched the data using Google Earth Engine and exported it to GCS bucket.

### Landsat-8
- Built a data fetch pipeline for Landsat-8/9 imagery over the same AOI.
- Fetched the data using Google Earth Engine and exported it to GCS bucket.

### Output
- All three datasets (Sentinel-2, Sentinel-5P, and Landsat-8) are now available in the GCS bucket for further processing.


## Lakshmi Sruthi K (Roll Number: 21F1005626)

- Explored Google Earth Engine for available datasets and contributed in finalizing the datasets.
- Worked on the FIRMS wildfire dataset for the California wildfire prediction pipeline. Extracted daily FIRMS GeoTIFF label rasters, reviewed the dataset structure and bands such as T21 and confidence, and examined how daily fire detections should be converted into binary fire/no-fire labels.
- Visualized this data in GEE and then created a script to export 10 years of FIRMS data to GCS bucket.

## Signatures
|Member|Roll Number|Signature Commit|
|--|--|--|
|Ripunjay Kumar|21F3002511|✅|
|Lakshay Garg|21F3001076||
|Roushan Kumar Singh|23F1002240|✅|
|Lakshmi Sruthi K|21F1005626|✅|
|R Aditya|21F1004839||
