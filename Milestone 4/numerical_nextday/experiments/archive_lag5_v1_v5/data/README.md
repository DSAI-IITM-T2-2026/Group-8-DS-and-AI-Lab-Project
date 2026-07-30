# Local static data

Input data is intentionally excluded from Git. If the general pipeline is
used, place the ERA5-grid DEM table at:

```text
data/era5_grid_dem_features.parquet
```

It must contain `cell_id`, `latitude`, `longitude`, `elevation`, `slope`,
`aspect`, `hillshade`, `tri`, and `tpi`.

The V1–V5 archive experiments instead expect their input under
`local_data/archive/`; see the parent README for the complete layout.
