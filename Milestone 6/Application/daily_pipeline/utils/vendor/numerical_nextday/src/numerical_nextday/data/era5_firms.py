"""ERA5 + FIRMS cache and Stage A assemble with era5_lag_days."""

from __future__ import annotations

import calendar
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


def _finished_month_reusable(
    era5_path: Path,
    firms_path: Path,
    year: int,
    month: int,
    clip_end: "pd.Timestamp | None",
) -> bool:
    """True when this calendar month has fully elapsed and both caches exist."""
    if not era5_path.exists() or not firms_path.exists():
        return False
    if era5_path.stat().st_size == 0:
        return False
    last = calendar.monthrange(year, month)[1]
    month_end = pd.Timestamp(year=year, month=month, day=last).normalize()
    if clip_end is None:
        return False
    return month_end < pd.Timestamp(clip_end).normalize()


def build_era5_firms_month(
    cfg: dict,
    year: int,
    month: int,
    worker: str = "local",
    force: bool = False,
    end_clip: "pd.Timestamp | None" = None,
    start_clip: "pd.Timestamp | None" = None,
    era5_end_clip: "pd.Timestamp | None" = None,
    firms_start_clip: "pd.Timestamp | None" = None,
    firms_end_clip: "pd.Timestamp | None" = None,
) -> Path | None:
    """Download/cache ERA5 daily + FIRMS cells for one calendar month.

    Daily pipeline: ERA5 is clipped to feature_end (D−6); FIRMS to the label
    lookback. Finished calendar months are reused so cron does not re-read
    July vsigs every August morning.
    """
    cache = shared_cache(cfg)
    stage = "era5_firms"
    era5_dir = cache / "era5_daily" / f"year={year}"
    firms_dir = cache / "firms_cells" / f"year={year}"
    era5_path = era5_dir / f"month={month:02d}.parquet"
    firms_path = firms_dir / f"month={month:02d}.parquet"

    if not force and _finished_month_reusable(era5_path, firms_path, year, month, end_clip):
        logger.info("era5_firms reuse finished month %04d-%02d", year, month)
        return era5_path

    # Open month (or missing cache): always rebuild so cron picks up the new day.
    claim(cache, worker=worker, stage=stage, year=year, month=month, force=True)

    mvp = load_mvp_modules(cfg)
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)

    era5_start = month_start
    era5_end = month_end
    if start_clip is not None:
        era5_start = max(era5_start, pd.Timestamp(start_clip).normalize())
    era5_stop = era5_end_clip if era5_end_clip is not None else end_clip
    if era5_stop is not None:
        era5_end = min(era5_end, pd.Timestamp(era5_stop).normalize())

    firms_start = month_start
    firms_end = month_end
    if firms_start_clip is not None:
        firms_start = max(firms_start, pd.Timestamp(firms_start_clip).normalize())
    elif start_clip is not None:
        firms_start = max(firms_start, pd.Timestamp(start_clip).normalize())
    firms_stop = firms_end_clip if firms_end_clip is not None else end_clip
    if firms_stop is not None:
        firms_end = min(firms_end, pd.Timestamp(firms_stop).normalize())

    raw_era5 = cache / "era5_raw"
    daily_tmp = cache / "_era5_daily_mvp"
    firms_tmp = cache / "_firms_mvp"

    if era5_end >= era5_start:
        era5 = mvp["era5_daily"].build_era5_daily_range(
            start=era5_start,
            end=era5_end,
            gcs_prefix=cfg["gcs"]["era5_prefix"],
            raw_cache=raw_era5,
            daily_cache=daily_tmp,
            legacy_prefix=cfg["gcs"].get("era5_legacy_prefix"),
        )
        _atomic_to_parquet(era5, era5_path)
    else:
        logger.info("era5_firms skip ERA5 %04d-%02d (empty window)", year, month)

    if firms_end >= firms_start:
        firms = mvp["firms_labels"].build_firms_cell_labels(
            start=firms_start,
            end=firms_end,
            vsigs_prefix=cfg["gcs"]["firms_vsigs_prefix"],
            confidence_min=float(cfg["task"]["firms_confidence_min"]),
            cache_dir=firms_tmp,
            resolution=float(cfg["era5"]["resolution_deg"]),
            months=[month],
        )
        _atomic_to_parquet(firms, firms_path)
    else:
        logger.info("era5_firms skip FIRMS %04d-%02d (empty window)", year, month)
        if not firms_path.exists():
            _atomic_to_parquet(
                pd.DataFrame(
                    columns=["date", "cell_id", "firms_n_pixels", "firms_max_confidence", "y_fire"]
                ),
                firms_path,
            )

    if not era5_path.exists():
        mark_done(cache, worker=worker, stage=stage, year=year, month=month)
        return None

    mark_done(cache, worker=worker, stage=stage, year=year, month=month)
    n_era5 = len(pd.read_parquet(era5_path, columns=["date"])) if era5_path.exists() else 0
    n_firms = len(pd.read_parquet(firms_path, columns=["date"])) if firms_path.exists() else 0
    logger.info("era5_firms done %04d-%02d era5=%d firms=%d", year, month, n_era5, n_firms)
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
    end_clip: "pd.Timestamp | None" = None,
    start_clip: "pd.Timestamp | None" = None,
) -> Path:
    """
    Build Stage A parquet for one label year.

    ERA5 feature_end_date = D_era5; label_date = D_era5 + lag + lead.
    lag=0 writes under m4_shared_cache/lag0/stage_a/.

    When `months` is a subset of the year (daily pipeline) and/or clips are set,
    only load ERA5/FIRMS for the label window plus 7-day weather history — not
    Jan–Dec, and not an extra 30d pad from the 1st of the first label month.
    """
    cache = shared_cache(cfg)
    lag = int(cfg["task"]["era5_lag_days"] if era5_lag_days is None else era5_lag_days)
    stage = "stage_a_year" if lag != 0 else "stage_a_year_lag0"
    stage_root = _stage_a_root(cache, lag)
    out = stage_root / f"year={year}.parquet"

    months = months or list(range(1, 13))
    clip_to_months = (
        months != list(range(1, 13)) or end_clip is not None or start_clip is not None
    )

    if not force and not clip_to_months and is_done(cache, stage, year, month=None) and out.exists():
        logger.info("Stage A year=%s lag=%s already done", year, lag)
        return out

    claim(cache, worker=worker, stage=stage, year=year, month=None, force=force or clip_to_months)

    mvp = load_mvp_modules(cfg)
    lead = int(cfg["task"]["lead_days"])
    history = int(cfg["task"]["history_days"])

    if clip_to_months:
        label_start = pd.Timestamp(year=year, month=min(months), day=1)
        label_end = pd.Timestamp(year=year, month=max(months), day=1) + pd.offsets.MonthEnd(0)
        if start_clip is not None:
            label_start = max(label_start, pd.Timestamp(start_clip).normalize())
        if end_clip is not None:
            label_end = min(label_end, pd.Timestamp(end_clip).normalize())
        if label_end < label_start:
            raise ValueError(
                f"Stage A empty window year={year} months={months} "
                f"start_clip={start_clip} end_clip={end_clip}"
            )
        # Match daily download: ERA5 through feature_end, 7d history.
        # lookback_days is already encoded in start_clip (first panel label).
        era5_start = label_start - pd.Timedelta(days=lag + lead + history)
        logger.info(
            "Stage A clipped year=%s labels %s…%s; ERA5/FIRMS months %s…%s",
            year,
            label_start.date(),
            label_end.date(),
            era5_start.to_period("M"),
            label_end.to_period("M"),
        )
    else:
        label_start = pd.Timestamp(year=year, month=1, day=1)
        label_end = pd.Timestamp(year=year, month=12, day=31)
        era5_start = label_start - pd.Timedelta(days=history + lag + lead)

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
            build_era5_firms_month(
                cfg,
                y,
                m,
                worker=worker,
                force=force,
                start_clip=era5_start,
                end_clip=end_clip if end_clip is not None else label_end,
                era5_end_clip=label_end - pd.Timedelta(days=lag + lead),
                firms_start_clip=label_start,
            )
        if epath.exists():
            era5_frames.append(pd.read_parquet(epath))
        if fpath.exists():
            firms_frames.append(pd.read_parquet(fpath))

    if not era5_frames:
        raise FileNotFoundError(
            f"No ERA5 month caches for year={year} {period_start}…{period_end}"
        )
    era5 = pd.concat(era5_frames, ignore_index=True)
    firms = (
        pd.concat(firms_frames, ignore_index=True)
        if firms_frames
        else pd.DataFrame()
    )
    era5["date"] = pd.to_datetime(era5["date"]).dt.normalize()
    if len(firms):
        firms["date"] = pd.to_datetime(firms["date"]).dt.normalize()
        firms["cell_id"] = firms["cell_id"].astype(str)
        firms = firms.drop_duplicates(["date", "cell_id"], keep="last")

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
