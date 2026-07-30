from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from numerical_nextday.io import atomic_json, atomic_parquet  # noqa: E402


IDENTITY_COLUMNS = [
    "feature_end_date",
    "eo_asof_date",
    "label_date",
    "cell_id",
    "latitude",
    "longitude",
    "y_fire",
]
CONSTANT_FEATURES = {"s5n_s5p_aai_std", "s5n_s5p_co_std"}
S2_CONTROL_COLUMNS = {"s2n_available"}
S5_CONTROL_COLUMNS = {"s5n_available"}
SOIL_COLUMNS = [
    "swvl1_mean",
    "swvl2_mean",
    "soil_moisture_index",
    "swvl1_mean_7d",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rolling_matrix(values: np.ndarray, window: int, operation: str) -> np.ndarray:
    table = pd.DataFrame(values.T)
    rolling = table.rolling(window=window, min_periods=1)
    result = getattr(rolling, operation)().to_numpy(dtype="float32").T
    return result


def add_weather_history(
    frame: pd.DataFrame, cells: int, days: int
) -> tuple[pd.DataFrame, list[str]]:
    engineered: dict[str, np.ndarray] = {}
    temperature_c = frame["t2m_mean"].to_numpy(dtype="float64") - 273.15
    dewpoint_c = frame["d2m_mean"].to_numpy(dtype="float64") - 273.15
    saturation = 0.6108 * np.exp(
        17.27 * temperature_c / np.maximum(temperature_c + 237.3, 1e-6)
    )
    actual = 0.6108 * np.exp(
        17.27 * dewpoint_c / np.maximum(dewpoint_c + 237.3, 1e-6)
    )
    vpd = np.maximum(saturation - actual, 0).astype("float32")
    engineered["vpd_kpa"] = vpd
    engineered["vpd_wind_interaction"] = (
        vpd * frame["wind_speed_mean"].to_numpy(dtype="float32")
    )
    engineered["vpd_soil_deficit_interaction"] = (
        vpd
        * (
            1
            - np.clip(
                frame["soil_moisture_index"].to_numpy(dtype="float32"), 0, 1
            )
        )
    )
    engineered["heat_soil_deficit_interaction"] = (
        np.maximum(frame["t2m_max"].to_numpy(dtype="float32") - 273.15, 0)
        * (
            1
            - np.clip(frame["swvl1_mean"].to_numpy(dtype="float32"), 0, 1)
        )
    )
    engineered["wind_gust_ratio"] = (
        frame["i10fg_max"].to_numpy(dtype="float32")
        / (frame["wind_speed_mean"].to_numpy(dtype="float32") + 0.1)
    )

    rolling_specs = {
        "t2m_max": ("max",),
        "rh_mean": ("min",),
        "tp_sum_mm": ("sum",),
        "wind_speed_mean": ("max",),
        "i10fg_max": ("max",),
        "swvl1_mean": ("mean",),
        "vpd_kpa": ("max", "mean"),
    }
    source_arrays = {
        column: (
            engineered[column]
            if column in engineered
            else frame[column].to_numpy(dtype="float32")
        ).reshape(cells, days)
        for column in rolling_specs
    }
    for column, operations in rolling_specs.items():
        for window in (14, 30):
            for operation in operations:
                name = f"{column}_{operation}_{window}d"
                engineered[name] = rolling_matrix(
                    source_arrays[column], window, operation
                ).reshape(-1)

    t2m_matrix = frame["t2m_max"].to_numpy(dtype="float32").reshape(cells, days)
    swvl_matrix = frame["swvl1_mean"].to_numpy(dtype="float32").reshape(cells, days)
    engineered["t2m_max_anomaly_30d"] = (
        t2m_matrix - rolling_matrix(t2m_matrix, 30, "mean")
    ).reshape(-1)
    engineered["swvl1_anomaly_30d"] = (
        swvl_matrix - rolling_matrix(swvl_matrix, 30, "mean")
    ).reshape(-1)
    additions = pd.DataFrame(engineered, index=frame.index).astype("float32")
    return pd.concat([frame, additions], axis=1), list(engineered)


def neighbor_lists(frame: pd.DataFrame, cells: int, days: int) -> list[np.ndarray]:
    first_rows = frame.iloc[np.arange(cells) * days]
    coordinates = first_rows[["latitude", "longitude"]].to_numpy(dtype="float64")
    result = []
    for index, (latitude, longitude) in enumerate(coordinates):
        delta_lat = np.abs(coordinates[:, 0] - latitude)
        delta_lon = np.abs(coordinates[:, 1] - longitude)
        mask = (
            (delta_lat <= 0.251)
            & (delta_lon <= 0.251)
            & ~((delta_lat < 1e-9) & (delta_lon < 1e-9))
        )
        result.append(np.flatnonzero(mask))
    return result


def sum_neighbors(values: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    output = np.zeros_like(values, dtype="float32")
    for index, adjacent in enumerate(neighbors):
        if adjacent.size:
            output[index] = values[adjacent].sum(axis=0)
    return output


def add_causal_fire_history(
    frame: pd.DataFrame, cells: int, days: int
) -> tuple[pd.DataFrame, list[str]]:
    target = frame["y_fire"].to_numpy(dtype="float32").reshape(cells, days)
    safe_lag2 = np.zeros_like(target, dtype="float32")
    safe_lag2[:, 2:] = target[:, :-2]
    history_7d = rolling_matrix(safe_lag2, 7, "sum")
    history_30d = rolling_matrix(safe_lag2, 30, "sum")
    neighbors = neighbor_lists(frame, cells, days)

    last_positive = np.full(cells, -10_000, dtype="int32")
    days_since = np.full_like(target, 365, dtype="float32")
    for day in range(days):
        positive = safe_lag2[:, day] > 0
        last_positive[positive] = day
        seen = last_positive > -10_000
        days_since[seen, day] = np.minimum(
            day - last_positive[seen], 365
        ).astype("float32")

    observed_count = np.maximum(np.arange(days, dtype="float32") - 1, 0)
    cumulative = np.cumsum(safe_lag2, axis=1, dtype="float32")
    smoothed_rate = (cumulative + 1.0) / (observed_count[None, :] + 100.0)
    statewide_7d = history_7d.sum(axis=0, dtype="float32")
    additions = {
        "fire_cell_lag2": safe_lag2.reshape(-1),
        "fire_cell_count_7d_lag2": history_7d.reshape(-1),
        "fire_cell_count_30d_lag2": history_30d.reshape(-1),
        "fire_cell_any_7d_lag2": (history_7d > 0).astype("float32").reshape(-1),
        "fire_cell_days_since_lag2": days_since.reshape(-1),
        "fire_cell_expanding_rate_lag2": smoothed_rate.astype("float32").reshape(-1),
        "fire_neighbor_count_lag2": sum_neighbors(
            safe_lag2, neighbors
        ).reshape(-1),
        "fire_neighbor_count_7d_lag2": sum_neighbors(
            history_7d, neighbors
        ).reshape(-1),
        "fire_neighbor_any_7d_lag2": (
            sum_neighbors(history_7d, neighbors) > 0
        )
        .astype("float32")
        .reshape(-1),
        "fire_statewide_cells_7d_lag2": np.tile(
            statewide_7d, cells
        ).astype("float32"),
    }
    table = pd.DataFrame(additions, index=frame.index)
    return pd.concat([frame, table], axis=1), list(additions)


def load_clean_stage_c(archive: Path) -> tuple[pd.DataFrame, list[str], dict]:
    stage = archive / "stage_c"
    features = json.loads(
        (stage / "metadata" / "feature_columns.json").read_text(encoding="utf-8")
    )
    features = [column for column in features if column not in CONSTANT_FEATURES]
    path = stage / "all.parquet"
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    columns = list(
        dict.fromkeys(
            [
                *IDENTITY_COLUMNS,
                *features,
                *[
                    column
                    for column in ("s2n_lag_days", "s5n_lag_days")
                    if column in schema
                ],
            ]
        )
    )
    frame = pd.read_parquet(path, columns=columns)
    for column in ("feature_end_date", "eo_asof_date", "label_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    if not (
        (frame["label_date"] - frame["eo_asof_date"]).dt.days.eq(1).all()
        and (frame["eo_asof_date"] - frame["feature_end_date"]).dt.days.eq(5).all()
    ):
        raise ValueError("Archive time relation is not label=D+1 and ERA5=D-5")
    frame = frame.sort_values(["cell_id", "label_date"]).reset_index(drop=True)
    cleanup = {}
    s2_invalid = (
        frame["s2n_available"].ne(1)
        | frame["s2n_lag_days"].isna()
        | frame["s2n_lag_days"].gt(15)
        | frame["s2n_lag_days"].lt(0)
    )
    s2_measurements = [
        column
        for column in features
        if column.startswith("s2n_") and column not in S2_CONTROL_COLUMNS
    ]
    frame.loc[s2_invalid, s2_measurements] = 0.0
    frame.loc[s2_invalid, "s2n_available"] = 0
    cleanup["s2_rows_zeroed"] = int(s2_invalid.sum())

    s5_invalid = (
        frame["s5n_available"].ne(1)
        | frame["s5n_lag_days"].isna()
        | frame["s5n_lag_days"].gt(2)
        | frame["s5n_lag_days"].lt(0)
    )
    s5_measurements = [
        column
        for column in features
        if column.startswith("s5n_") and column not in S5_CONTROL_COLUMNS
    ]
    frame.loc[s5_invalid, s5_measurements] = 0.0
    frame.loc[s5_invalid, "s5n_available"] = 0
    cleanup["s5_rows_zeroed"] = int(s5_invalid.sum())

    for column in SOIL_COLUMNS:
        count = int(frame[column].lt(0).sum())
        frame[column] = frame[column].clip(lower=0)
        cleanup[f"{column}_negative_rows_clipped"] = count
    frame[features] = frame[features].astype("float32")
    return frame, features, cleanup


def build(archive: Path, output: Path) -> None:
    print("[v2] loading and cleaning Stage C", flush=True)
    frame, base_features, cleanup = load_clean_stage_c(archive)
    cells = int(frame["cell_id"].nunique())
    days = int(frame["label_date"].nunique())
    if len(frame) != cells * days:
        raise ValueError("V2 construction requires a complete cell-by-day grid")
    for _, group in frame.groupby("cell_id", sort=False):
        if not group["label_date"].is_monotonic_increasing:
            raise ValueError("Rows are not time ordered within cells")

    day_of_year = frame["eo_asof_date"].dt.dayofyear.to_numpy(dtype="float32")
    calendar = pd.DataFrame(
        {
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "month_sin": np.sin(
                2
                * np.pi
                * frame["eo_asof_date"].dt.month.to_numpy(dtype="float32")
                / 12
            ),
            "month_cos": np.cos(
                2
                * np.pi
                * frame["eo_asof_date"].dt.month.to_numpy(dtype="float32")
                / 12
            ),
        },
        index=frame.index,
    ).astype("float32")
    frame = pd.concat([frame, calendar], axis=1)
    calendar_features = list(calendar)
    print("[v2] adding ERA5-derived rolling and interaction features", flush=True)
    frame, weather_features = add_weather_history(frame, cells, days)
    print("[v2] adding strictly lagged fire-history features", flush=True)
    frame, fire_features = add_causal_fire_history(frame, cells, days)

    geographic_features = ["latitude", "longitude"]
    full_features = list(
        dict.fromkeys(
            [
                *base_features,
                *geographic_features,
                *calendar_features,
                *weather_features,
                *fire_features,
            ]
        )
    )
    no_fire_features = [
        column for column in full_features if column not in set(fire_features)
    ]
    no_s5p_features = [
        column for column in full_features if not column.startswith("s5n_")
    ]
    no_s5p_availability_features = [
        column for column in full_features if column != "s5n_available"
    ]
    if not np.isfinite(frame[full_features].to_numpy(dtype="float32")).all():
        raise ValueError("Non-finite V2 feature detected")

    output.mkdir(parents=True, exist_ok=True)
    data_columns = list(
        dict.fromkeys(
            [
                *IDENTITY_COLUMNS,
                "s2n_lag_days",
                "s5n_lag_days",
                *full_features,
            ]
        )
    )
    split_years = {
        "train": {2019, 2020, 2021, 2022},
        "val": {2023, 2024},
        "test": {2025},
    }
    outputs = {}
    for split, years in split_years.items():
        part = frame.loc[frame["label_date"].dt.year.isin(years), data_columns]
        print(f"[v2] writing {split}: {len(part):,} rows", flush=True)
        outputs[split] = atomic_parquet(part, output / f"{split}.parquet")
    feature_sets = {
        "v2_full": full_features,
        "v2_no_fire_history": no_fire_features,
        "v2_no_s5p": no_s5p_features,
        "v2_without_s5p_availability": no_s5p_availability_features,
    }
    atomic_json(feature_sets, output / "feature_sets.json")
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": str((archive / "stage_c" / "all.parquet").resolve()),
        "source_sha256": sha256(archive / "stage_c" / "all.parquet"),
        "rows": len(frame),
        "cells": cells,
        "days": days,
        "time_contract": {
            "forecast_day_D": "eo_asof_date",
            "target": "y_fire on D+1",
            "era5_cutoff": "feature_end_date = D-5",
            "fire_history_cutoff": (
                "All fire-history features end at label_date-2, which is D-1; "
                "neither target day D+1 nor forecast day D FIRMS labels are used."
            ),
            "weather_rolling_cutoff": (
                "All weather rolling features end at feature_end_date, which is D-5."
            ),
        },
        "cleanup": cleanup,
        "feature_counts": {
            name: len(columns) for name, columns in feature_sets.items()
        },
        "split_outputs": outputs,
    }
    atomic_json(metadata, output / "metadata.json")
    print(f"[v2] complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.archive.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
