"""GCS inventory for FIRMS/ERA5/DEM + new S2/S5P CSV feature stores."""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any, Optional

from src import config
from src.data.feature_csv import (
    list_feature_csv_paths,
    list_partition_dirs,
    peek_window_dates,
    read_features_csv,
)
from src.data.gcs import get_fs, list_bucket_files

FIRMS_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.tif$")
ERA5_RE = re.compile(r"era5_(\d{4})_(\d{2})\.nc$")

SOURCE_META = {
    "FIRMS": {
        "file_type": "GeoTIFF (.tif)",
        "path_pattern": f"gs://{config.FIRMS_BUCKET}/{config.FIRMS_PREFIX}/YYYY-MM-DD.tif",
        "timestamp_encoding": "Filename date YYYY-MM-DD",
        "expected_frequency": "daily",
        "expected_contents": "3 bands: confidence (0-100), brightness_temp_K, detection_flag",
        "preprocess": "Threshold confidence>=30 → binary fire label; use as reference grid",
    },
    "Sentinel-2": {
        "file_type": "CSV features.csv (Hive partitions)",
        "path_pattern": (
            f"gs://{config.S2_BUCKET}/{config.S2_PREFIX}/"
            "year=YYYY/month=MM/window=NNN/features.csv"
        ),
        "timestamp_encoding": "Partition year/month/window + columns window_start, window_end",
        "expected_frequency": "~5-day composite windows (~6 windows/month; window IDs 001…073/year)",
        "expected_contents": (
            "~413k rows (CA grid cells): grid_id, lat/lon, band means/stds, "
            "NDVI/NBR/NDWI/EVI/… stats, cloud_percentage, s2_data_available"
        ),
        "preprocess": (
            "Already featurized — join on grid_id/lat/lon; map window_start→target dates; "
            "no GeoTIFF reproject needed for these features"
        ),
        "project_id": config.S2_PROJECT_ID,
        "bucket": config.S2_BUCKET,
    },
    "Sentinel-5P": {
        "file_type": "CSV features.csv (Hive partitions)",
        "path_pattern": (
            f"gs://{config.S5P_BUCKET}/{config.S5P_PREFIX}/"
            "year=YYYY/month=MM/window=NNN/features.csv"
        ),
        "timestamp_encoding": "Partition year/month/window (window≈day-of-year) + window_start/end",
        "expected_frequency": "daily (window_days typically 1)",
        "expected_contents": (
            "~413k rows: grid_id, lat/lon, s5p_aai_*, s5p_co_*, availability flags"
        ),
        "preprocess": "Join on grid_id; use s5p_aai_mean (and optionally CO) as atmospheric features",
        "project_id": config.S5P_PROJECT_ID,
        "bucket": config.S5P_BUCKET,
    },
    "ERA5": {
        "file_type": "ZIP archive with .nc extension containing 2 NetCDF files",
        "path_pattern": f"gs://{config.GCS_BUCKET}/{config.PREFIX_ERA5}/{{year}}/era5_{{year}}_{{mm}}.nc",
        "timestamp_encoding": "Path year/month; inner NetCDF time dim 'valid_time' (hourly)",
        "expected_frequency": "monthly file of hourly steps → aggregate to daily; coverage 2016–2025 CA bbox",
        "expected_contents": (
            "14 short codes: t2m,d2m,sp,u10,v10,i10fg,tp,swvl1,swvl2,cvh,cvl,lai_hv,lai_lv,blh"
        ),
        "preprocess": "Open as ZIP; merge instant+accum; rename valid_time→time; daily agg; align to grid",
    },
    "DEM": {
        "file_type": "GeoTIFF (.tif), non-COG ~4–5GB each",
        "path_pattern": f"gs://{config.GCS_BUCKET}/{config.PREFIX_DEM_TERRAIN}/{{layer}}.tif",
        "timestamp_encoding": "Static (no time dimension)",
        "expected_frequency": "static",
        "expected_contents": "6 layers: elevation, slope, aspect, hillshade, tpi, tri (1 band each)",
        "preprocess": "Stream-reproject / sample onto modeling grid; delete local temps",
    },
}


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _month_keys(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def inventory_firms(year: Optional[int] = 2024) -> dict[str, Any]:
    files = list_bucket_files(config.FIRMS_BUCKET, config.FIRMS_PREFIX, suffix=".tif")
    dates = []
    for f in files:
        name = f.split("/")[-1]
        m = FIRMS_DATE_RE.match(name)
        if m:
            dates.append(m.group(1))
    dates = sorted(dates)
    meta = dict(SOURCE_META["FIRMS"])
    meta["n_files"] = len(dates)
    meta["first"] = dates[0] if dates else None
    meta["last"] = dates[-1] if dates else None
    missing = []
    if year is not None:
        present = set(dates)
        for d in _daterange(date(year, 1, 1), date(year, 12, 31)):
            key = d.isoformat()
            if key not in present:
                missing.append(
                    {
                        "source": "FIRMS",
                        "missing": key,
                        "expected_path": f"gs://{config.FIRMS_BUCKET}/{config.FIRMS_PREFIX}/{key}.tif",
                        "file_type": meta["file_type"],
                        "should_contain": meta["expected_contents"],
                        "time_range": key,
                    }
                )
    meta["missing"] = missing
    meta["missing_count"] = len(missing)
    return meta


def inventory_sentinel2_csv(year: int = 2024) -> dict[str, Any]:
    """Inventory new S2 features.csv store; expect ~6 windows/month for a full year."""
    rows = list_feature_csv_paths(config.S2_BUCKET, config.S2_PREFIX, year=year)
    years_all = list_partition_dirs(config.S2_BUCKET, config.S2_PREFIX)
    years_present = sorted(
        int(re.search(r"year=(\d{4})", y).group(1))
        for y in years_all
        if re.search(r"year=(\d{4})", y)
    )

    by_ym: dict[tuple[int, int], list[int]] = {}
    for r in rows:
        by_ym.setdefault((r["year"], r["month"]), []).append(r["window"])

    missing = []
    # Expected: 6 windows/month Jan–Sep,Nov–Dec; Oct often 7 (leap-year-ish packing) — use observed max pattern
    # For gap report: months with 0 windows, and years 2016-2020 absent from bucket name claim
    for y, m in _month_keys(year, 1, year, 12):
        wins = sorted(by_ym.get((y, m), []))
        if not wins:
            missing.append(
                {
                    "source": "Sentinel-2",
                    "missing": f"{y:04d}-{m:02d} (no windows)",
                    "expected_path": (
                        f"gs://{config.S2_BUCKET}/{config.S2_PREFIX}/"
                        f"year={y:04d}/month={m:02d}/window=NNN/features.csv"
                    ),
                    "file_type": SOURCE_META["Sentinel-2"]["file_type"],
                    "should_contain": SOURCE_META["Sentinel-2"]["expected_contents"],
                    "time_range": f"{y:04d}-{m:02d}",
                }
            )
        elif len(wins) < 6:
            missing.append(
                {
                    "source": "Sentinel-2",
                    "missing": f"{y:04d}-{m:02d} only {len(wins)} windows (expected ≥6)",
                    "expected_path": (
                        f"gs://{config.S2_BUCKET}/{config.S2_PREFIX}/"
                        f"year={y:04d}/month={m:02d}/window=*/features.csv"
                    ),
                    "file_type": SOURCE_META["Sentinel-2"]["file_type"],
                    "should_contain": SOURCE_META["Sentinel-2"]["expected_contents"],
                    "time_range": f"{y:04d}-{m:02d}",
                }
            )

    for y in range(2016, 2021):
        if y not in years_present:
            missing.append(
                {
                    "source": "Sentinel-2",
                    "missing": f"year={y} partition absent",
                    "expected_path": f"gs://{config.S2_BUCKET}/{config.S2_PREFIX}/year={y}/",
                    "file_type": SOURCE_META["Sentinel-2"]["file_type"],
                    "should_contain": SOURCE_META["Sentinel-2"]["expected_contents"],
                    "time_range": str(y),
                    "note": "Bucket name says 2016-2025 but only 2021+ partitions found",
                }
            )

    sample_meta = {}
    windows_for_map = {}
    if rows:
        sample = rows[len(rows) // 2]
        try:
            sample_meta = peek_window_dates(sample["path"])
            sample_meta["path"] = sample["path"]
            sample_meta["size"] = sample["size"]
            # schema check
            df = read_features_csv(sample["path"], nrows=2)
            sample_meta["columns"] = list(df.columns)
            sample_meta["n_columns"] = len(df.columns)
            missing_cols = [c for c in config.S2_CSV_KEY_COLUMNS if c not in df.columns]
            sample_meta["missing_key_columns"] = missing_cols
        except Exception as exc:
            sample_meta = {"error": str(exc)}

        for r in rows:
            key = f"{r['year']:04d}-{r['month']:02d}"
            windows_for_map.setdefault(key, []).append(
                {
                    "window": r["window"],
                    "path": f"gs://{r['path']}",
                    "present": True,
                    "size": r["size"],
                }
            )

    meta = dict(SOURCE_META["Sentinel-2"])
    meta["n_files"] = len(rows)
    meta["years_present"] = years_present
    meta["windows_per_month"] = {
        f"{y:04d}-{m:02d}": sorted(wins) for (y, m), wins in sorted(by_ym.items())
    }
    meta["sample"] = sample_meta
    meta["missing"] = missing
    meta["missing_count"] = len(missing)
    meta["windows_by_month"] = windows_for_map
    return meta


def inventory_sentinel5p_csv(
    year: Optional[int] = None,
    expected_start: tuple[int, int] = (2016, 1),
    expected_end: tuple[int, int] = (2025, 12),
) -> dict[str, Any]:
    years_all = list_partition_dirs(config.S5P_BUCKET, config.S5P_PREFIX)
    years_present = sorted(
        int(re.search(r"year=(\d{4})", y).group(1))
        for y in years_all
        if re.search(r"year=(\d{4})", y)
    )
    rows = list_feature_csv_paths(
        config.S5P_BUCKET, config.S5P_PREFIX, year=year
    )
    # If year filter empty but years exist, list all for inventory
    if not rows and year is None:
        rows = list_feature_csv_paths(config.S5P_BUCKET, config.S5P_PREFIX)

    by_ym: dict[tuple[int, int], list[int]] = {}
    for r in rows:
        by_ym.setdefault((r["year"], r["month"]), []).append(r["window"])

    missing = []
    for y, m in _month_keys(expected_start[0], expected_start[1], expected_end[0], expected_end[1]):
        wins = by_ym.get((y, m), [])
        if not wins:
            missing.append(
                {
                    "source": "Sentinel-5P",
                    "missing": f"{y:04d}-{m:02d}",
                    "expected_path": (
                        f"gs://{config.S5P_BUCKET}/{config.S5P_PREFIX}/"
                        f"year={y:04d}/month={m:02d}/window=NNN/features.csv"
                    ),
                    "file_type": SOURCE_META["Sentinel-5P"]["file_type"],
                    "should_contain": SOURCE_META["Sentinel-5P"]["expected_contents"],
                    "time_range": f"{y:04d}-{m:02d} (daily windows)",
                    "note": "Daily features — expect ~28–31 window folders per month",
                }
            )

    sample_meta = {}
    windows_for_map = {}
    if rows:
        sample = rows[0]
        try:
            sample_meta = peek_window_dates(sample["path"])
            sample_meta["path"] = sample["path"]
            df = read_features_csv(sample["path"], nrows=2)
            sample_meta["columns"] = list(df.columns)
            missing_cols = [c for c in config.S5P_CSV_KEY_COLUMNS if c not in df.columns]
            sample_meta["missing_key_columns"] = missing_cols
            # row count from a known small-ish daily file
            from src.data.feature_csv import latlon_bounds_from_csv

            # only compute bounds once (expensive) — use peek + documented 413k
            sample_meta["approx_nrows"] = 413000
            sample_meta["bounds_note"] = "Same CA grid as S2 (~413k cells); sample lat/lon in peek"
        except Exception as exc:
            sample_meta = {"error": str(exc)}

        for r in rows:
            key = f"{r['year']:04d}-{r['month']:02d}"
            windows_for_map.setdefault(key, []).append(
                {
                    "window": r["window"],
                    "path": f"gs://{r['path']}",
                    "present": True,
                    "size": r["size"],
                }
            )

    meta = dict(SOURCE_META["Sentinel-5P"])
    meta["n_files"] = len(rows)
    meta["years_present"] = years_present
    meta["windows_per_month"] = {
        f"{y:04d}-{m:02d}": sorted(wins) for (y, m), wins in sorted(by_ym.items())
    }
    meta["sample"] = sample_meta
    meta["missing"] = missing
    meta["missing_count"] = len(missing)
    meta["windows_by_month"] = windows_for_map
    return meta


def inventory_era5(expected_years: Optional[list[int]] = None, probe_zip: bool = True) -> dict[str, Any]:
    if expected_years is None:
        expected_years = list(range(2016, 2026))

    present: dict[tuple[int, int], str] = {}
    years_found = []
    for year in expected_years:
        prefix = f"{config.PREFIX_ERA5}/{year}"
        try:
            files = list_bucket_files(config.GCS_BUCKET, prefix, suffix=".nc")
        except Exception:
            files = []
        if files:
            years_found.append(year)
        for f in files:
            name = f.split("/")[-1]
            m = ERA5_RE.search(name)
            if m:
                present[(int(m.group(1)), int(m.group(2)))] = f

    missing = []
    for year in expected_years:
        for month in range(1, 13):
            if (year, month) not in present:
                missing.append(
                    {
                        "source": "ERA5",
                        "missing": f"{year:04d}-{month:02d}",
                        "expected_path": (
                            f"gs://{config.GCS_BUCKET}/{config.PREFIX_ERA5}/"
                            f"{year}/era5_{year}_{month:02d}.nc"
                        ),
                        "file_type": SOURCE_META["ERA5"]["file_type"],
                        "should_contain": SOURCE_META["ERA5"]["expected_contents"],
                        "time_range": f"{year:04d}-{month:02d} (hourly inside)",
                    }
                )

    zip_ok = None
    vars_found = None
    missing_vars = []
    if probe_zip and present:
        sample = present[sorted(present.keys())[0]]
        try:
            from src.data.loaders import open_era5_file

            ds = open_era5_file(sample)
            vars_found = sorted(ds.data_vars)
            zip_ok = True
            missing_vars = sorted(set(config.ERA5_EXPECTED_VARIABLES) - set(vars_found))
        except Exception as exc:
            zip_ok = False
            missing_vars = [str(exc)]

    meta = dict(SOURCE_META["ERA5"])
    meta["n_files"] = len(present)
    meta["years_found"] = years_found
    meta["first"] = f"{min(present)[0]:04d}-{min(present)[1]:02d}" if present else None
    meta["last"] = f"{max(present)[0]:04d}-{max(present)[1]:02d}" if present else None
    meta["missing"] = missing
    meta["missing_count"] = len(missing)
    meta["zip_open_ok"] = zip_ok
    meta["variables_in_sample"] = vars_found
    meta["missing_variables_in_sample"] = missing_vars
    meta["coverage_note"] = "Generated for California bounding box; years 2016–2025"
    return meta


def inventory_dem() -> dict[str, Any]:
    missing = []
    present = []
    fs = get_fs()
    for layer, filename in config.DEM_TERRAIN_FILES.items():
        path = f"{config.GCS_BUCKET}/{config.PREFIX_DEM_TERRAIN}/{filename}"
        if fs.exists(path):
            present.append(filename)
        else:
            missing.append(
                {
                    "source": "DEM",
                    "missing": filename,
                    "expected_path": f"gs://{path}",
                    "file_type": SOURCE_META["DEM"]["file_type"],
                    "should_contain": f"Single-band {layer} GeoTIFF",
                    "time_range": "static",
                }
            )
    meta = dict(SOURCE_META["DEM"])
    meta["n_files"] = len(present)
    meta["present"] = present
    meta["missing"] = missing
    meta["missing_count"] = len(missing)
    return meta


def load_aoi_metadata() -> dict:
    path = config.METADATA_DIR / "aoi_bounds.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "aoi_bounds": config.DEFAULT_AOI_BOUNDS,
        "per_source_bounds": {},
        "notes": "aoi_bounds.json not found — using DEFAULT_AOI_BOUNDS",
    }


def binding_edges(aoi_doc: dict) -> dict[str, str]:
    aoi = aoi_doc.get("aoi_bounds", {})
    sources = aoi_doc.get("per_source_bounds", {})
    binding = {}
    for edge, key in [("west", "west"), ("south", "south"), ("east", "east"), ("north", "north")]:
        vals = {name: b[key] for name, b in sources.items() if key in b}
        if not vals:
            binding[edge] = "unknown"
            continue
        target = aoi.get(key)
        winners = [
            name for name, v in vals.items() if target is not None and abs(v - target) < 1e-6
        ]
        if not winners:
            winners = [min(vals, key=lambda n: abs(vals[n] - (target or 0)))]
        binding[edge] = ", ".join(winners)
    return binding


def build_full_inventory(year: int = 2024, **_kwargs) -> dict[str, Any]:
    print("Inventory FIRMS...")
    firms = inventory_firms(year=year)
    print(f"  FIRMS: {firms['n_files']} files, missing {firms['missing_count']} days in {year}")

    print("Inventory Sentinel-2 CSV features...")
    s2 = inventory_sentinel2_csv(year=year)
    print(
        f"  S2: {s2['n_files']} window CSVs, years={s2['years_present']}, "
        f"missing items={s2['missing_count']}"
    )

    print("Inventory Sentinel-5P CSV features...")
    s5p = inventory_sentinel5p_csv()
    print(
        f"  S5P: {s5p['n_files']} window CSVs, years={s5p['years_present']}, "
        f"missing items={s5p['missing_count']}"
    )

    print("Inventory ERA5...")
    era5 = inventory_era5()
    print(f"  ERA5: {era5['n_files']} files, missing {era5['missing_count']} months")

    print("Inventory DEM...")
    dem = inventory_dem()
    print(f"  DEM: {dem['n_files']} present, missing {dem['missing_count']}")

    aoi_doc = load_aoi_metadata()
    # Enrich with CSV grid extent note (full CA ~ intended box)
    csv_extent = {
        "west": -124.406,
        "south": 32.533,
        "east": -114.135,
        "north": 42.013,
        "nrows": 413115,
        "note": "Measured from S5P sample window; S2 uses same grid_id/lat-lon",
    }
    aoi_doc.setdefault("per_source_bounds", {})
    aoi_doc["per_source_bounds"]["Sentinel-2-CSV"] = {
        k: csv_extent[k] for k in ("west", "south", "east", "north")
    }
    aoi_doc["per_source_bounds"]["Sentinel-5P-CSV"] = {
        k: csv_extent[k] for k in ("west", "south", "east", "north")
    }

    binding = binding_edges(aoi_doc)

    return {
        "year_focus": year,
        "sources": {
            "FIRMS": firms,
            "Sentinel-2": s2,
            "Sentinel-5P": s5p,
            "ERA5": era5,
            "DEM": dem,
        },
        "aoi": aoi_doc,
        "csv_grid_extent": csv_extent,
        "binding_edges": binding,
        "alignment": {
            "spatial": (
                "New S2/S5P CSVs share grid_id + lat/lon (~413k CA cells). "
                "FIRMS/ERA5/DEM are rasters — join by nearest grid cell or reproject onto FIRMS. "
                "CSV extent ≈ full intended CA box; ERA5 may still bind if its NetCDF bbox is smaller."
            ),
            "temporal": (
                "FIRMS+ERA5: daily. S2 CSV: ~5-day windows (window_start/end). "
                "S5P CSV: daily windows. DEM: static."
            ),
            "region_for_modeling": aoi_doc.get("aoi_bounds"),
            "to_expand_aoi": (
                "Re-check ERA5 spatial bbox vs CSV grid extent. "
                "S2/S5P CSV already cover ~full CA; S5P time coverage is still sparse on GCS."
            ),
        },
    }


def render_missing_markdown(inv: dict) -> str:
    lines = [
        "# Missing / anomalous GCS data — for the team",
        "",
        f"Focus year: **{inv['year_focus']}**. Generated by `scripts/verify_gcs_data.py`.",
        "",
        "## New Sentinel feature stores",
        "",
        f"- **S2:** `gs://{config.S2_BUCKET}/{config.S2_PREFIX}/` (project `{config.S2_PROJECT_ID}`)",
        f"- **S5P:** `gs://{config.S5P_BUCKET}/{config.S5P_PREFIX}/` (project `{config.S5P_PROJECT_ID}`)",
        "- Format: `year=/month=/window=/features.csv`, ~413k rows per file (CA grid cells)",
        "",
        "## Locked AOI (raster intersection — see also CSV grid extent)",
        "",
        f"```json\n{json.dumps(inv['aoi'].get('aoi_bounds'), indent=2)}\n```",
        "",
        "### CSV grid extent (S2/S5P shared)",
        "",
        f"```json\n{json.dumps(inv.get('csv_grid_extent'), indent=2)}\n```",
        "",
        "### Binding edges (raster AOI)",
        "",
        "| Edge | Binding source(s) |",
        "|---|---|",
    ]
    for edge, src in inv.get("binding_edges", {}).items():
        lines.append(f"| {edge} | {src} |")

    lines += [
        "",
        "## How timestamps work",
        "",
        "| Source | File type | Timestamp | Cadence |",
        "|---|---|---|---|",
    ]
    for name, meta in inv["sources"].items():
        lines.append(
            f"| {name} | {meta['file_type']} | {meta['timestamp_encoding']} | "
            f"{meta.get('expected_frequency', '')} |"
        )

    lines += ["", "## Gaps and anomalies", ""]
    for name, meta in inv["sources"].items():
        missing = meta.get("missing", [])
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Path pattern: `{meta['path_pattern']}`")
        lines.append(f"- Should contain: {meta['expected_contents']}")
        lines.append(f"- Preprocess before use: {meta['preprocess']}")
        if meta.get("years_present") is not None:
            lines.append(f"- Years present: {meta['years_present']}")
        if meta.get("sample"):
            lines.append(f"- Sample: `{meta['sample']}`")
        if not missing:
            lines.append("- No gaps detected in the checked window.")
        else:
            lines.append(f"- **{len(missing)} missing item(s)** (showing up to 80):")
            lines.append("")
            lines.append("| Missing | Expected path | File type | Should contain | Time |")
            lines.append("|---|---|---|---|---|")
            for row in missing[:80]:
                lines.append(
                    f"| {row['missing']} | `{row['expected_path']}` | "
                    f"{row['file_type']} | {row['should_contain']} | {row.get('time_range','')} |"
                )
            if len(missing) > 80:
                lines.append(f"| … | ({len(missing) - 80} more) | | | |")
        lines.append("")

    lines += [
        "",
        "## Alignment summary",
        "",
        f"- Spatial: {inv['alignment']['spatial']}",
        f"- Temporal: {inv['alignment']['temporal']}",
        f"- Region for modeling: `{inv['alignment']['region_for_modeling']}`",
        f"- To expand / fix coverage: {inv['alignment']['to_expand_aoi']}",
        "",
    ]
    return "\n".join(lines)


def build_coverage_map_data(inv: dict) -> dict:
    s2 = inv["sources"]["Sentinel-2"]
    s5p = inv["sources"]["Sentinel-5P"]
    aoi = inv["aoi"].get("aoi_bounds", {})
    csv_extent = inv.get("csv_grid_extent", {})

    windows = sorted(s2.get("windows_by_month", {}).keys())
    n_miss_s2 = s2.get("missing_count", 0)
    n_miss_s5p = s5p.get("missing_count", 0)

    verdict = (
        "Sentinel-2/5P are now tabular CA-grid CSVs (~413k rows/file) with Hive partitions. "
        f"S2 years on GCS: {s2.get('years_present')}. "
        f"S5P years on GCS: {s5p.get('years_present')} (still sparse). "
    )
    if n_miss_s2 or n_miss_s5p:
        verdict += (
            f"Gap counts — S2 items missing={n_miss_s2}, S5P month gaps={n_miss_s5p}. "
            "See missing_for_team.md."
        )
    else:
        verdict += "No S2/S5P gaps in the checked windows."

    return {
        "aoi_bounds": aoi,
        "csv_grid_extent": csv_extent,
        "per_source_bounds": inv["aoi"].get("per_source_bounds", {}),
        "binding_edges": inv.get("binding_edges", {}),
        "windows": windows,
        "s2_windows_by_month": s2.get("windows_by_month", {}),
        "s5p_windows_by_month": s5p.get("windows_by_month", {}),
        "verdict": verdict,
        "intended_aoi": config.DEFAULT_AOI_BOUNDS,
        "data_mode": "csv_features_v3",
    }
