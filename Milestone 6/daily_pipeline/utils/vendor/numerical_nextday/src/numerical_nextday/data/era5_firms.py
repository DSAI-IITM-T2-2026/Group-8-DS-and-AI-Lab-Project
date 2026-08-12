"""ERA5 + FIRMS cache and Stage A assemble with era5_lag_days."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from numerical_nextday.config import shared_cache
from numerical_nextday.data.claims import claim, is_done, mark_done
from numerical_nextday.data.m3_imports import load_mvp_modules

logger = logging.getLogger(__name__)


def _atomic_to_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    df.to_parquet(partial, index=False)
    partial.replace(dest)


def parse_year_month_ranges(
    years: list[int],
    months: list[int] | None,
) -> list[tuple[int, int]]:
    months = months or list(range(1, 13))
    return [(y, m) for y in years for m in months]


def build_era5_firms_month(
    cfg: dict,
    year: int,
    month: int,
    worker: str = "local",
    force: bool = False,
    end_clip: "pd.Timestamp | None" = None,
) -> Path | None:
    """Download/cache ERA5 daily + FIRMS cells for one calendar month.

    If end_clip is set (daily pipeline), do not read past that date so a mid-month
    label does not require future FIRMS/ERA5 days.
    """
    cache = shared_cache(cfg)
    stage = "era5_firms"
    # When clipping mid-month, always rebuild so a prior full-month cache is not reused.
    force_month = force or end_clip is not None
    if not claim(cache, worker=worker, stage=stage, year=year, month=month, force=force_month):
        return cache / "era5_daily" / f"year={year}" / f"month={month:02d}.parquet"

    mvp = load_mvp_modules(cfg)
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    if end_clip is not None:
        end = min(end, pd.Timestamp(end_clip).normalize())
        if end < start:
            logger.warning("era5_firms skip %04d-%02d (end_clip=%s before month start)", year, month, end_clip)
            mark_done(cache, worker=worker, stage=stage, year=year, month=month)
            return None

    era5_dir = cache / "era5_daily" / f"year={year}"
    firms_dir = cache / "firms_cells" / f"year={year}"
    era5_path = era5_dir / f"month={month:02d}.parquet"
    firms_path = firms_dir / f"month={month:02d}.parquet"
    raw_era5 = cache / "era5_raw"
    daily_tmp = cache / "_era5_daily_mvp"
    firms_tmp = cache / "_firms_mvp"

    # Prefer monthly under era5/raw/, else stitch daily era5_YYYY_MM_DD.nc
    era5 = mvp["era5_daily"].build_era5_daily_range(
        start=start,
        end=end,
        gcs_prefix=cfg["gcs"]["era5_prefix"],
        raw_cache=raw_era5,
        daily_cache=daily_tmp,
        legacy_prefix=cfg["gcs"].get("era5_legacy_prefix"),
    )
    _atomic_to_parquet(era5, era5_path)

    firms = mvp["firms_labels"].build_firms_cell_labels(
        start=start,
        end=end,
        vsigs_prefix=cfg["gcs"]["firms_vsigs_prefix"],
        confidence_min=float(cfg["task"]["firms_confidence_min"]),
        cache_dir=firms_tmp,
        resolution=float(cfg["era5"]["resolution_deg"]),
        months=[month],
    )
    _atomic_to_parquet(firms, firms_path)

    mark_done(cache, worker=worker, stage=stage, year=year, month=month)
    logger.info("era5_firms done %04d-%02d era5=%d firms=%d", year, month, len(era5), len(firms))
    return era5_path


def _stage_a_root(cache: Path, lag: int) -> Path:
    """Primary lag-5 tables live at stage_a/; lag-0 ablation under lag0/stage_a/."""
    if lag == 0:
        return cache / "lag0" / "stage_a"
    return cache / "stage_a"


def assemble_stage_a_year(
    cfg: dict,
    year: int,
    months: list[int] | None = None,
    worker: str = "local",
    force: bool = False,
    era5_lag_days: int | None = None,
) -> Path:
    """
    Build Stage A parquet for one label year.

    ERA5 feature_end_date = D_era5; label_date = D_era5 + lag + lead.
    lag=0 writes under m4_shared_cache/lag0/stage_a/.
    """
    cache = shared_cache(cfg)
    lag = int(cfg["task"]["era5_lag_days"] if era5_lag_days is None else era5_lag_days)
    stage = "stage_a_year" if lag != 0 else "stage_a_year_lag0"
    stage_root = _stage_a_root(cache, lag)
    out = stage_root / f"year={year}.parquet"

    if not force and is_done(cache, stage, year, month=None) and out.exists():
        logger.info("Stage A year=%s lag=%s already done", year, lag)
        return out

    claim(cache, worker=worker, stage=stage, year=year, month=None, force=force)

    mvp = load_mvp_modules(cfg)
    lead = int(cfg["task"]["lead_days"])
    history = int(cfg["task"]["history_days"])
    months = months or list(range(1, 13))

    # Label year window; need ERA5 from (start - history - lag) through end
    label_start = pd.Timestamp(year=year, month=1, day=1)
    label_end = pd.Timestamp(year=year, month=12, day=31)
    # Restrict to requested months on label_date later
    era5_start = label_start - pd.Timedelta(days=history + lag + lead)
    era5_end = label_end - pd.Timedelta(days=lead)  # feature days for last labels
    _ = era5_end

    # Load ERA5/FIRMS from hive caches spanning needed months
    era5_frames = []
    firms_frames = []
    period_start = era5_start.to_period("M")
    period_end = label_end.to_period("M")
    for period in pd.period_range(period_start, period_end, freq="M"):
        y, m = period.year, period.month
        epath = cache / "era5_daily" / f"year={y}" / f"month={m:02d}.parquet"
        fpath = cache / "firms_cells" / f"year={y}" / f"month={m:02d}.parquet"
        if not epath.exists() or not fpath.exists():
            logger.info("Missing cache for %04d-%02d — building", y, m)
            build_era5_firms_month(cfg, y, m, worker=worker, force=force)
        era5_frames.append(pd.read_parquet(epath))
        firms_frames.append(pd.read_parquet(fpath))

    era5 = pd.concat(era5_frames, ignore_index=True)
    firms = pd.concat(firms_frames, ignore_index=True)
    era5["date"] = pd.to_datetime(era5["date"]).dt.normalize()
    if len(firms):
        firms["date"] = pd.to_datetime(firms["date"]).dt.normalize()

    dem = mvp["cells"].load_dem_cells(Path(cfg["paths"]["dem_cells"]))

    # Assemble with lead=1 as in M3 (label = feature_end + 1), then shift label by lag
    samples = mvp["assemble"].assemble_samples(
        dem=dem,
        era5_daily=era5,
        firms_cells=firms,
        history_days=history,
        lead_days=lead,
    )
    # M3: label = feature_end + lead. For lag-L we need label = feature_end + lead + L
    # → shift label_date forward by lag days and re-merge FIRMS on new label_date.
    if lag > 0:
        samples = samples.drop(
            columns=["y_fire", "firms_n_pixels", "firms_max_confidence"],
            errors="ignore",
        )
        samples["label_date"] = samples["feature_end_date"] + pd.Timedelta(days=lead + lag)
        if len(firms):
            labels = firms.rename(
                columns={
                    "date": "label_date",
                }
            )
            samples = samples.merge(
                labels[
                    [
                        c
                        for c in [
                            "label_date",
                            "cell_id",
                            "firms_n_pixels",
                            "firms_max_confidence",
                            "y_fire",
                        ]
                        if c in labels.columns
                    ]
                ],
                on=["label_date", "cell_id"],
                how="left",
            )
        else:
            samples["y_fire"] = 0
            samples["firms_n_pixels"] = 0
            samples["firms_max_confidence"] = np.nan
        samples["y_fire"] = samples["y_fire"].fillna(0).astype("int8")
        samples["firms_n_pixels"] = samples["firms_n_pixels"].fillna(0).astype("int32")

    samples["era5_lag_days"] = lag
    samples["eo_asof_date"] = samples["label_date"] - pd.Timedelta(days=1)

    # Keep label year / months
    samples = samples.loc[
        (samples["label_date"] >= label_start) & (samples["label_date"] <= label_end)
    ]
    samples = samples.loc[samples["label_date"].dt.month.isin(months)].reset_index(drop=True)

    _atomic_to_parquet(samples, out)
    meta = {
        "year": year,
        "era5_lag_days": lag,
        "lead_days": lead,
        "n_rows": len(samples),
        "n_pos": int(samples["y_fire"].sum()),
        "label_span": [
            str(samples["label_date"].min().date()) if len(samples) else None,
            str(samples["label_date"].max().date()) if len(samples) else None,
        ],
    }
    (stage_root / f"year={year}.meta.json").write_text(json.dumps(meta, indent=2))
    mark_done(cache, worker=worker, stage=stage, year=year, month=None)
    logger.info("Wrote Stage A %s rows=%d pos=%d", out, len(samples), int(samples["y_fire"].sum()))
    return out
