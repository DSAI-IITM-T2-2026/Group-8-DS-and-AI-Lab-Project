from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..grid import coordinates_to_cell_ids
from ..io import atomic_parquet, gcs_list, gcs_read_csv

logger = logging.getLogger(__name__)


def eo_cache_path(cfg: dict, source: str, year: int, month: int) -> Path:
    return (
        cfg["paths"]["cache_dir"] / f"{source}_cell" / f"year={year}" / f"month={month:02d}.parquet"
    )


def source_prefix(cfg: dict, source: str, year: int) -> str:
    mapping = cfg["gcs"][f"{source}_prefix_by_year"]
    try:
        return mapping[str(year)]
    except KeyError:
        raise KeyError(f"No {source.upper()} GCS prefix configured for year {year}") from None


def _output_name(source: str, original: str) -> str:
    lowered = original.lower()
    if lowered.startswith(f"{source}_"):
        return lowered
    return f"{source}_{lowered}"


def configured_feature_columns(cfg: dict, source: str) -> list[str]:
    return [_output_name(source, feature) for feature in cfg["features"][source]]


def _aggregate_frame(
    frame: pd.DataFrame,
    source: str,
    selected_features: list[str],
    resolution: float,
) -> pd.DataFrame:
    required = {"latitude", "longitude", "window_end"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{source.upper()} file missing columns: {sorted(missing)}")
    available = [feature for feature in selected_features if feature in frame]
    if not available:
        raise ValueError(
            f"{source.upper()} file has none of the configured features; "
            f"configured={selected_features[:5]}"
        )
    frame["cell_id"] = coordinates_to_cell_ids(
        frame["longitude"].to_numpy(),
        frame["latitude"].to_numpy(),
        resolution,
    )
    frame["window_end"] = pd.to_datetime(frame["window_end"]).dt.normalize()
    numeric = frame[available].apply(pd.to_numeric, errors="coerce")
    numeric["cell_id"] = frame["cell_id"]
    numeric["window_end"] = frame["window_end"]
    aggregated = numeric.groupby(["cell_id", "window_end"], as_index=False).mean()
    return aggregated.rename(
        columns={feature: _output_name(source, feature) for feature in available}
    )


def _stream_and_aggregate(
    uri: str,
    cfg: dict,
    source: str,
    selected_features: list[str],
    wanted_columns: set[str],
) -> pd.DataFrame:
    streamed = gcs_read_csv(
        uri,
        wanted_columns,
        max_attempts=int(cfg["execution"]["eo_stream_retries"]),
    )
    return _aggregate_frame(
        streamed,
        source,
        selected_features,
        float(cfg["aoi"]["resolution_deg"]),
    )


def build_eo_month(cfg: dict, source: str, year: int, month: int, force: bool = False) -> Path:
    if source not in {"s2", "s5p"}:
        raise ValueError("source must be 's2' or 's5p'")
    destination = eo_cache_path(cfg, source, year, month)
    if destination.exists() and not force:
        return destination

    prefix = source_prefix(cfg, source, year).rstrip("/")
    pattern = f"{prefix}/year={year}/month={month:02d}/window=*/features.csv"
    allow_missing = (source == "s5p" and cfg["execution"]["allow_missing_s5p"]) or (
        source == "s2" and not cfg["execution"]["fail_on_missing_s2"]
    )
    try:
        objects = gcs_list(pattern)
    except PermissionError:
        if allow_missing:
            logger.warning(
                "%s unavailable for %04d-%02d; writing empty fallback shard",
                source.upper(),
                year,
                month,
            )
            empty = pd.DataFrame(columns=["cell_id", "window_end", f"{source}_data_available"])
            atomic_parquet(empty, destination)
            return destination
        raise
    if not objects:
        if allow_missing:
            logger.warning(
                "No %s objects for %04d-%02d; writing empty fallback shard",
                source.upper(),
                year,
                month,
            )
            empty = pd.DataFrame(columns=["cell_id", "window_end", f"{source}_data_available"])
            atomic_parquet(empty, destination)
            return destination
        raise FileNotFoundError(f"No objects matched {pattern}")

    selected_features = list(cfg["features"][source])
    wanted_columns = {
        "latitude",
        "longitude",
        "window_end",
        *selected_features,
    }
    if cfg["execution"].get("eo_read_mode", "stream") != "stream":
        raise ValueError("Only execution.eo_read_mode=stream is currently supported")

    stream_count = min(
        len(objects),
        int(cfg["execution"].get("eo_parallel_streams", 1)),
    )
    frames_by_position: dict[int, pd.DataFrame] = {}
    logger.info(
        "Streaming %d %s object(s) for %04d-%02d with %d parallel stream(s)",
        len(objects),
        source.upper(),
        year,
        month,
        stream_count,
    )
    with ThreadPoolExecutor(max_workers=stream_count) as pool:
        pending = {
            pool.submit(
                _stream_and_aggregate,
                uri,
                cfg,
                source,
                selected_features,
                wanted_columns,
            ): position
            for position, uri in enumerate(objects)
        }
        for completed, future in enumerate(as_completed(pending), start=1):
            position = pending[future]
            frames_by_position[position] = future.result()
            logger.info(
                "Completed %s object %d/%d for %04d-%02d",
                source.upper(),
                completed,
                len(objects),
                year,
                month,
            )
    frames = [frames_by_position[position] for position in range(len(objects))]
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.groupby(["cell_id", "window_end"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["window_end", "cell_id"])
    )
    atomic_parquet(combined, destination)
    logger.info("Built %s cell shard %s (%d rows)", source.upper(), destination, len(combined))
    return destination


def read_eo_for_year(cfg: dict, source: str, year: int) -> pd.DataFrame:
    paths = []
    previous_december = eo_cache_path(cfg, source, year - 1, 12)
    # S2 five-day composites can validly bridge the year boundary. S5P is daily;
    # carrying the previous year's last observation would hide a missing-year outage.
    if source == "s2" and previous_december.exists():
        paths.append(previous_december)
    paths.extend(
        eo_cache_path(cfg, source, year, month)
        for month in range(1, 13)
        if eo_cache_path(cfg, source, year, month).exists()
    )
    if not paths:
        return pd.DataFrame(columns=["cell_id", "window_end"])
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["window_end"] = pd.to_datetime(frame["window_end"]).dt.normalize()
    return frame.sort_values(["window_end", "cell_id"]).reset_index(drop=True)


def causal_attach(
    base: pd.DataFrame,
    eo: pd.DataFrame,
    source: str,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    """Attach the latest per-cell EO record whose window ended by feature day D."""
    left = base.copy()
    left["_row_order"] = np.arange(len(left))
    if eo.empty:
        left[f"{source}_window_end"] = pd.NaT
        left[f"{source}_age_days"] = np.nan
        if source == "s5p":
            left["s5p_data_available"] = 0.0
        return left.drop(columns="_row_order")

    right = eo.rename(columns={"window_end": f"{source}_window_end"}).copy()
    left = left.sort_values(["feature_end_date", "cell_id"])
    right = right.sort_values([f"{source}_window_end", "cell_id"])
    joined = pd.merge_asof(
        left,
        right,
        left_on="feature_end_date",
        right_on=f"{source}_window_end",
        by="cell_id",
        direction="backward",
        allow_exact_matches=True,
    )
    joined[f"{source}_age_days"] = (
        joined["feature_end_date"] - joined[f"{source}_window_end"]
    ).dt.days.astype("float32")
    if (joined[f"{source}_window_end"] > joined["feature_end_date"]).fillna(False).any():
        raise ValueError(f"{source.upper()} causal join attached future data")
    if max_age_days is not None:
        stale = joined[f"{source}_age_days"] > max_age_days
        source_columns = [
            column
            for column in joined
            if column.startswith(f"{source}_") and column not in {f"{source}_age_days"}
        ]
        joined.loc[stale, source_columns] = np.nan
        joined.loc[stale, f"{source}_age_days"] = np.nan
    if source == "s5p":
        value_columns = [
            column
            for column in joined
            if column.startswith("s5p_")
            and column not in {"s5p_window_end", "s5p_age_days", "s5p_data_available"}
        ]
        joined[value_columns] = joined[value_columns].fillna(0.0)
        joined["s5p_data_available"] = joined.get(
            "s5p_data_available", pd.Series(index=joined.index, dtype=float)
        ).fillna(0.0)
    return joined.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
