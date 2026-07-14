# DEM × ERA5 Fusion Analysis (California)

## Goal
Use **static Copernicus DEM terrain features** together with your **ERA5 time series**
for wildfire modeling over California (2021–2025).

## Key idea
DEM does not change with time. ERA5 does. So:
1. Build a **static lookup table** of DEM features on the ERA5 grid (one row per cell).
2. For every ERA5 timestep `(time, lat, lon)`, **join** DEM features by cell id / lat-lon.
3. Train models on the combined feature vector.

## Your ERA5 variables

### Dynamic weather / atmosphere (change hourly–daily)
- `2m_temperature`
- `2m_dewpoint_temperature`
- `surface_pressure`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `instantaneous_10m_wind_gust`
- `total_precipitation`
- `volumetric_soil_water_layer_1`
- `volumetric_soil_water_layer_2`
- `boundary_layer_height`

### Vegetation / land cover (slow / near-static in ERA5)
- `high_vegetation_cover`
- `low_vegetation_cover`
- `leaf_area_index_high_vegetation`
- `leaf_area_index_low_vegetation`

### DEM static covariates (this pipeline)
- `elevation`
- `slope`
- `aspect`
- `hillshade`
- `tri`
- `tpi`

## Recommended join schema

```
ERA5 row:  time | lat | lon | 2m_temperature | ... | boundary_layer_height
DEM row:         lat | lon | elevation | slope | aspect | hillshade | tri | tpi
Joined:    time | lat | lon | [all ERA5] | [all DEM]
```

- ERA5 resolution assumed: **0.25°**
- California ERA5 cells with DEM samples: **675**
- Cells with valid elevation: **672**

## How DEM helps with your ERA5 features

| ERA5 feature group | How DEM adds value |
|--------------------|--------------------|
| Temperature / dewpoint | Elevation explains cold air drainage & lapse rate; slope/aspect drive local heating |
| Wind U/V + gust | TRI / TPI / slope capture channeling, ridges, and exposure |
| Precipitation | Orographic lift controlled by elevation + slope facing moisture flow |
| Soil water L1/L2 | Slope & TPI relate to runoff vs retention; valleys hold moisture |
| Vegetation cover / LAI | Elevation & aspect control vegetation zones; DEM is prior for fuel structure |
| Boundary layer height | Terrain roughness (TRI) and elevation modulate BLH patterns |
| Surface pressure | Strongly tied to elevation (hydrostatic); DEM is a strong covariate |

## Practical fusion recipes

### A. Grid-cell tabular model (recommended start)
- Resample / sample DEM to ERA5 0.25° centers (done in `era5_grid_dem_features.parquet`).
- Join DEM columns onto every ERA5 timestep.
- Optional derived DEM features: `wind_exposure = slope * sin(aspect)`, `vpd` from T + dewpoint, `wind_speed = hypot(u,v)`.

### B. Patch / multimodal model
- Keep DEM at higher resolution (30–300 m) as spatial context around each ERA5 cell.
- ERA5 provides the temporal weather sequence; DEM provides local terrain patch.

### C. Derived physics-inspired features
```python
wind_speed = hypot(u10, v10)
vpd ≈ es(T) - es(Td)   # from 2m_temperature & 2m_dewpoint
orographic_index = elevation * slope
ridge_indicator = (tpi > threshold)
```

## Suggested feature matrix for ML

**Static (DEM):** elevation, slope, aspect (or sin/cos aspect), tri, tpi  
**Slow ERA5:** vegetation cover + LAI  
**Dynamic ERA5:** T, Td, pressure, winds, gust, precip, soil water, BLH  
**Labels:** FIRMS / fire occurrence at matching space-time

## Files produced
- `outputs/numerical/era5_grid_dem_features.parquet` — static DEM @ ERA5 cells
- `outputs/figures/era5_grid_elevation.png` — map of elevation on ERA5 grid
- this report
