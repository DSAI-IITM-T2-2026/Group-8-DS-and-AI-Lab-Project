#!/usr/bin/env python3
"""
Preflight: estimate GCS download / local cache size vs free disk.

Does NOT download. Exit code 1 if free space < estimated need + margin.

Usage:
  python scripts/estimate_disk.py --year 2025 --start 2025-06-01 --end 2025-11-30
  python scripts/estimate_disk.py --year 2025 --start 2025-06-01 --end 2025-11-30 --require-gb 40

Wire into next build:
  python scripts/estimate_disk.py ... --require-free && \\
  python scripts/build_dataset.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")

from src import config
from src.data.feature_csv import list_feature_csv_paths
from src.data.gcs import get_fs


def _gb(n: float) -> str:
    return f"{n / (1024**3):.2f} GiB"


def _local_usage() -> dict:
    cache = config.CACHE_DIR
    processed = config.OUTPUT_DIR
    usage = {
        "cache_total": _dir_size(cache),
        "dem": _dir_size(config.CACHE_DEM_DIR),
        "s2_csv": _dir_size(config.CACHE_S2_DIR),
        "s5p_csv": _dir_size(config.CACHE_S5P_DIR),
        "era5": _dir_size(config.CACHE_ERA5_DIR),
        "firms": _dir_size(config.CACHE_FIRMS_DIR),
        "processed": _dir_size(processed),
    }
    disk = shutil.disk_usage(str(config.M3_ROOT))
    usage["disk_total"] = disk.total
    usage["disk_used"] = disk.used
    usage["disk_free"] = disk.free
    return usage


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _sum_sizes(paths: list[str], fs, label: str, sample_max: int | None = None) -> tuple[int, int]:
    """Return (total_bytes_or_estimate, n_files). Samples if sample_max set."""
    if not paths:
        return 0, 0
    to_stat = paths if sample_max is None else paths[:sample_max]
    sizes = []
    for i, p in enumerate(to_stat):
        try:
            sizes.append(int(fs.info(p).get("size") or 0))
        except Exception as exc:
            print(f"  warn {label} size fail {p}: {exc}")
        if (i + 1) % 20 == 0:
            print(f"  … sized {i+1}/{len(to_stat)} {label}")
    if not sizes:
        return 0, len(paths)
    avg = sum(sizes) / len(sizes)
    if sample_max is not None and len(paths) > len(sizes):
        return int(avg * len(paths)), len(paths)
    return int(sum(sizes)), len(paths)


def estimate(year: int, start: str, end: str) -> dict:
    fs = get_fs()
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    # history buffer for 7-day windows
    hist_start = start_d - timedelta(days=config.HISTORY_DAYS)
    months = sorted(
        {
            (hist_start + timedelta(days=i)).month
            for i in range((end_d - hist_start).days + 1)
            if (hist_start + timedelta(days=i)).year == year
        }
        | {
            (start_d + timedelta(days=i)).month
            for i in range((end_d - start_d).days + 1)
            if (start_d + timedelta(days=i)).year == year
        }
    )

    print(f"Estimating for {start} → {end} (year={year}, months={months})")
    print("Querying GCS object sizes (anonymous)…\n")

    # --- DEM (already often cached as ~22 GiB full CA tifs) ---
    dem_remote = 0
    dem_paths = []
    for name, fname in config.DEM_TERRAIN_FILES.items():
        path = f"{config.GCS_BUCKET}/{config.PREFIX_DEM_TERRAIN}/{fname}"
        dem_paths.append(path)
        local = config.CACHE_DEM_DIR / fname
        if local.exists() and local.stat().st_size > 0:
            continue
        try:
            dem_remote += int(fs.info(path).get("size") or 0)
        except Exception as exc:
            print(f"  DEM {fname}: {exc}")
    dem_local = _dir_size(config.CACHE_DEM_DIR)

    # --- S2 CSV ---
    s2_rows = list_feature_csv_paths(config.S2_BUCKET, config.S2_PREFIX, year=year)
    s2_rows = [r for r in s2_rows if r.get("month") in set(months)]
    s2_need = []
    for r in s2_rows:
        # disk cache key uses window_end — unknown without peek; count all as remote if no year cache files
        end_guess = None
        disk_hits = list((config.CACHE_S2_DIR / str(year)).glob("*.npz")) if (config.CACHE_S2_DIR / str(year)).exists() else []
        # If we have fewer cached scenes than windows, remaining ≈ (1 - hit_frac) * total
        s2_need.append(r["path"])
    n_s2_cached = len(list((config.CACHE_S2_DIR / str(year)).glob("*.npz"))) if (config.CACHE_S2_DIR / str(year)).exists() else 0
    s2_bytes_all, n_s2 = _sum_sizes(s2_need, fs, "S2", sample_max=None)
    # Remaining remote ≈ proportional to uncached windows (rasterize still streams CSV once)
    s2_remain_frac = max(0.0, 1.0 - (n_s2_cached / max(n_s2, 1)))
    s2_remain = int(s2_bytes_all * s2_remain_frac)

    # --- S5P CSV ---
    s5p_rows = list_feature_csv_paths(config.S5P_BUCKET, config.S5P_PREFIX, year=year)
    s5p_rows = [r for r in s5p_rows if r.get("month") in set(months)]
    s5p_paths = [r["path"] for r in s5p_rows]
    n_s5p_cached = len(list(config.CACHE_S5P_DIR.glob("*.npy"))) if config.CACHE_S5P_DIR.exists() else 0
    s5p_bytes_all, n_s5p = _sum_sizes(s5p_paths, fs, "S5P", sample_max=min(12, len(s5p_paths)) or None)
    s5p_remain_frac = max(0.0, 1.0 - (n_s5p_cached / max(n_s5p, 1)))
    s5p_remain = int(s5p_bytes_all * s5p_remain_frac)

    # --- ERA5 monthly ---
    era5_remain = 0
    era5_detail = []
    for m in months:
        path = f"{config.GCS_BUCKET}/{config.PREFIX_ERA5}/{year}/era5_{year}_{m:02d}.nc"
        marker = config.CACHE_ERA5_DIR / str(year) / f"{m:02d}" / "_complete.json"
        try:
            sz = int(fs.info(path).get("size") or 0)
        except Exception as exc:
            print(f"  ERA5 {year}-{m:02d}: {exc}")
            sz = 0
        era5_detail.append((m, sz, marker.exists()))
        if not marker.exists():
            era5_remain += sz

    # --- FIRMS daily (streamed; label cache is tiny; count remote read volume) ---
    from src.data.loaders import list_firms_dates

    firms_dates = [d for d in list_firms_dates() if hist_start.isoformat() <= d <= end]
    # sample a few sizes
    firms_sample_paths = []
    # discover naming via gcs list one file — use known pattern from loaders
    from src.data.loaders import load_firms_raster
    import inspect
    # Prefer inventory: list bucket files with date in range
    from src.data.gcs import list_bucket_files

    firms_files = list_bucket_files(config.FIRMS_BUCKET, config.FIRMS_PREFIX, suffix=".tif")
    # map date → path heuristically
    date_to_path = {}
    for f in firms_files:
        name = f.split("/")[-1]
        # firms_YYYY-MM-DD.tif or similar
        for d in firms_dates:
            if d in name or d.replace("-", "_") in name or d.replace("-", "") in name:
                date_to_path[d] = f if f.startswith("gs://") else f"gs://{f}"
                break
    missing_firms = [d for d in firms_dates if not (config.CACHE_FIRMS_DIR / f"{d}.npy").exists()]
    sample_paths = list(date_to_path.values())[:5]
    firms_bytes_est = 0
    if sample_paths:
        fb, _ = _sum_sizes(
            [p.replace("gs://", "") for p in sample_paths],
            fs,
            "FIRMS",
            sample_max=None,
        )
        avg = fb / max(len(sample_paths), 1)
        firms_bytes_est = int(avg * len(missing_firms))
    else:
        # fallback ~5 MB/day typical
        firms_bytes_est = int(5e6 * len(missing_firms))

    # Processed .npy output ballpark: smoke was ~190 MB for 57 patches;
    # fire season could be 10–50× → reserve 10 GiB headroom for patches + checkpoints
    processed_headroom = 10 * (1024**3)

    # Local cache inflation: S2/S5P npz often much smaller than CSV; ERA5 daily npz can be large
    # Reserve extra 50% of remaining remote for local cache writes
    remote_remain = dem_remote + s2_remain + s5p_remain + era5_remain + firms_bytes_est
    local_write_estimate = int(remote_remain * 0.5) + processed_headroom
    # DEM already local — don't double count
    need_free = dem_remote + local_write_estimate + int(s2_remain * 0.1)  # CSV stream is transient
    # More honest: need free ≈ remaining downloads that land on disk
    # DEM downloads full files; S2/S5P/ERA5/FIRMS land as cache + processed
    s2_cache_est = int(n_s2 * 15e6)  # ~15 MB npz/scene for 1031x1114x3
    s5p_cache_est = int(n_s5p * 5e6)
    era5_cache_est = int(sum(sz for _, sz, done in era5_detail if not done) * 1.2)  # daily grids ≥ nc
    firms_cache_est = int(len(missing_firms) * 1e6)  # binary masks small
    need_on_disk = (
        dem_remote
        + max(0, s2_cache_est - _dir_size(config.CACHE_S2_DIR))
        + max(0, s5p_cache_est - _dir_size(config.CACHE_S5P_DIR))
        + max(0, era5_cache_est - _dir_size(config.CACHE_ERA5_DIR))
        + firms_cache_est
        + processed_headroom
    )

    return {
        "year": year,
        "start": start,
        "end": end,
        "months": months,
        "dem_local_bytes": dem_local,
        "dem_remote_remaining_bytes": dem_remote,
        "s2": {
            "windows": n_s2,
            "cached_scenes": n_s2_cached,
            "remote_csv_bytes": s2_bytes_all,
            "remote_csv_remaining_bytes": s2_remain,
            "local_cache_est_bytes": s2_cache_est,
        },
        "s5p": {
            "windows": n_s5p,
            "cached_days": n_s5p_cached,
            "remote_csv_bytes_est": s5p_bytes_all,
            "remote_csv_remaining_bytes": s5p_remain,
            "local_cache_est_bytes": s5p_cache_est,
        },
        "era5": {
            "months": [
                {"month": m, "remote_bytes": sz, "cached": done} for m, sz, done in era5_detail
            ],
            "remote_remaining_bytes": era5_remain,
            "local_cache_est_bytes": era5_cache_est,
        },
        "firms": {
            "days_in_range": len(firms_dates),
            "days_uncached": len(missing_firms),
            "remote_read_est_bytes": firms_bytes_est,
            "local_cache_est_bytes": firms_cache_est,
        },
        "processed_headroom_bytes": processed_headroom,
        "estimated_additional_disk_bytes": need_on_disk,
        "local": _local_usage(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=config.CANDIDATE_YEAR)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--margin-gb",
        type=float,
        default=15.0,
        help="Extra free space to keep beyond estimate (default 15 GiB)",
    )
    parser.add_argument(
        "--require-free",
        action="store_true",
        help="Exit 1 if free disk < estimate + margin (block build)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(config.METADATA_DIR / "disk_estimate.json"),
    )
    args = parser.parse_args()
    start = args.start or f"{args.year}-06-01"
    end = args.end or (f"{args.year}-11-30" if args.year == 2025 else f"{args.year}-12-31")

    report = estimate(args.year, start, end)
    local = report["local"]
    need = report["estimated_additional_disk_bytes"]
    margin = int(args.margin_gb * (1024**3))
    free = local["disk_free"]
    ok = free >= need + margin

    print("\n========== LOCAL NOW ==========")
    print(f"  Free disk:     {_gb(free)}  (of {_gb(local['disk_total'])})")
    print(f"  data/cache:    {_gb(local['cache_total'])}  (DEM {_gb(local['dem'])})")
    print(f"  data/processed:{_gb(local['processed'])}")

    print("\n========== REMAINING (estimate) ==========")
    print(f"  DEM still to download: {_gb(report['dem_remote_remaining_bytes'])}")
    s2 = report["s2"]
    print(
        f"  S2 CSV: {s2['cached_scenes']}/{s2['windows']} scenes cached; "
        f"remote CSV ~{_gb(s2['remote_csv_bytes'])} "
        f"(~{_gb(s2['remote_csv_remaining_bytes'])} left to stream); "
        f"local npz est {_gb(s2['local_cache_est_bytes'])}"
    )
    s5p = report["s5p"]
    print(
        f"  S5P CSV: {s5p['cached_days']}/{s5p['windows']} cached; "
        f"remote ~{_gb(s5p['remote_csv_bytes_est'])} "
        f"(~{_gb(s5p['remote_csv_remaining_bytes'])} left); "
        f"local npz est {_gb(s5p['local_cache_est_bytes'])}"
    )
    print(f"  ERA5 remaining remote: {_gb(report['era5']['remote_remaining_bytes'])}")
    for row in report["era5"]["months"]:
        flag = "cached" if row["cached"] else "NEED"
        print(f"    {args.year}-{row['month']:02d}: {_gb(row['remote_bytes'])} [{flag}]")
    fr = report["firms"]
    print(
        f"  FIRMS: {fr['days_uncached']}/{fr['days_in_range']} days uncached; "
        f"remote read est {_gb(fr['remote_read_est_bytes'])}"
    )
    print(f"  Processed .npy headroom: {_gb(report['processed_headroom_bytes'])}")
    print(f"\n  >>> Estimated ADDITIONAL disk needed: {_gb(need)}")
    print(f"  >>> Plus safety margin:               {_gb(margin)}")
    print(f"  >>> Free now:                         {_gb(free)}")
    print(f"  >>> Verdict: {'OK — enough space' if ok else 'NOT ENOUGH SPACE — free disk or shrink date range'}")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")

    if args.require_free and not ok:
        print(
            "\nRefusing to proceed (--require-free). "
            "Free space or use a shorter --start/--end / --max-days.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
