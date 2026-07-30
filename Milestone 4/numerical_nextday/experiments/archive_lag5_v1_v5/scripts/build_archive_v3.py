from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = [
    "feature_end_date",
    "eo_asof_date",
    "label_date",
    "cell_id",
    "latitude",
    "longitude",
    "y_fire",
]
ROUTER_COLUMN = "v3_recent_fire_context"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def neighbor_geometry(
    coordinates: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    result = []
    for latitude, longitude in coordinates:
        delta_lat = latitude - coordinates[:, 0]
        delta_lon = (longitude - coordinates[:, 1]) * np.cos(
            np.deg2rad(latitude)
        )
        distance = np.sqrt(delta_lat**2 + delta_lon**2)
        adjacent = np.flatnonzero((distance > 1e-9) & (distance <= 0.36))
        result.append(
            (
                adjacent,
                (delta_lon[adjacent] / distance[adjacent]).astype("float32"),
                (delta_lat[adjacent] / distance[adjacent]).astype("float32"),
                (1.0 / (distance[adjacent] + 0.05)).astype("float32"),
            )
        )
    return result


def directional_neighbor_features(
    fire_values: np.ndarray,
    wind_from_sin: np.ndarray,
    wind_from_cos: np.ndarray,
    geometry: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cells, days = fire_values.shape
    upwind = np.zeros((cells, days), dtype="float32")
    downwind = np.zeros((cells, days), dtype="float32")
    crosswind = np.zeros((cells, days), dtype="float32")
    distance_weighted = np.zeros((cells, days), dtype="float32")
    wind_to_east = -wind_from_sin
    wind_to_north = -wind_from_cos
    for index, (adjacent, east, north, inverse_distance) in enumerate(geometry):
        if not adjacent.size:
            continue
        neighbor_fire = fire_values[adjacent]
        alignment = (
            wind_to_east[index][None, :] * east[:, None]
            + wind_to_north[index][None, :] * north[:, None]
        )
        upwind[index] = (
            neighbor_fire * np.maximum(alignment, 0) * inverse_distance[:, None]
        ).sum(axis=0)
        downwind[index] = (
            neighbor_fire * np.maximum(-alignment, 0) * inverse_distance[:, None]
        ).sum(axis=0)
        cross_alignment = np.sqrt(np.maximum(1.0 - alignment**2, 0))
        crosswind[index] = (
            neighbor_fire * cross_alignment * inverse_distance[:, None]
        ).sum(axis=0)
        distance_weighted[index] = (
            neighbor_fire * inverse_distance[:, None]
        ).sum(axis=0)
    return upwind, downwind, crosswind, distance_weighted


def add_v3_features(
    frame: pd.DataFrame, cells: int, days: int
) -> tuple[pd.DataFrame, list[str]]:
    coordinates = frame.iloc[np.arange(cells) * days][
        ["latitude", "longitude"]
    ].to_numpy(dtype="float64")
    geometry = neighbor_geometry(coordinates)
    wind_sin = frame["wind_dir_sin"].to_numpy(dtype="float32").reshape(
        cells, days
    )
    wind_cos = frame["wind_dir_cos"].to_numpy(dtype="float32").reshape(
        cells, days
    )
    lag2 = frame["fire_cell_lag2"].to_numpy(dtype="float32").reshape(cells, days)
    history7 = frame["fire_cell_count_7d_lag2"].to_numpy(
        dtype="float32"
    ).reshape(cells, days)
    upwind_lag2, _, _, distance_lag2 = directional_neighbor_features(
        lag2, wind_sin, wind_cos, geometry
    )
    (
        upwind_history7,
        downwind_history7,
        crosswind_history7,
        distance_history7,
    ) = directional_neighbor_features(history7, wind_sin, wind_cos, geometry)

    wind = frame["wind_speed_mean"].to_numpy(dtype="float32")
    vpd = frame["vpd_kpa"].to_numpy(dtype="float32")
    soil_deficit = 1.0 - np.clip(
        frame["soil_moisture_index"].to_numpy(dtype="float32"), 0, 1
    )
    vegetation = np.clip(
        frame["cvh_mean"].to_numpy(dtype="float32")
        + frame["cvl_mean"].to_numpy(dtype="float32"),
        0,
        1,
    )
    neighbor_history = frame["fire_neighbor_count_7d_lag2"].to_numpy(
        dtype="float32"
    )
    recent_context = np.maximum(
        frame["fire_cell_any_7d_lag2"].to_numpy(dtype="float32"),
        frame["fire_neighbor_any_7d_lag2"].to_numpy(dtype="float32"),
    )
    additions = {
        "fire_upwind_count_lag2": upwind_lag2.reshape(-1),
        "fire_upwind_count_7d_lag2": upwind_history7.reshape(-1),
        "fire_downwind_count_7d_lag2": downwind_history7.reshape(-1),
        "fire_crosswind_count_7d_lag2": crosswind_history7.reshape(-1),
        "fire_distance_weighted_count_lag2": distance_lag2.reshape(-1),
        "fire_distance_weighted_count_7d_lag2": distance_history7.reshape(-1),
        "fire_wind_spread_potential_lag2": upwind_lag2.reshape(-1) * wind,
        "fire_wind_spread_potential_7d_lag2": (
            upwind_history7.reshape(-1) * wind
        ),
        "fire_context_vpd_interaction": neighbor_history * vpd,
        "fire_context_dry_windy_interaction": (
            recent_context * vpd * wind * soil_deficit
        ),
        "ignition_dry_windy_index": vpd * wind * soil_deficit,
        "fuel_dryness_index": vpd * soil_deficit * vegetation,
        "vpd_short_long_trend": (
            frame["vpd_kpa_mean_14d"].to_numpy(dtype="float32")
            - frame["vpd_kpa_mean_30d"].to_numpy(dtype="float32")
        ),
        ROUTER_COLUMN: recent_context,
    }
    table = pd.DataFrame(additions, index=frame.index).astype("float32")
    return pd.concat([frame, table], axis=1), list(additions)


def atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def build(v2_directory: Path, output: Path) -> None:
    feature_sets = json.loads(
        (v2_directory / "feature_sets.json").read_text(encoding="utf-8")
    )
    v2_features = feature_sets["v2_without_s5p_availability"]
    load_columns = list(
        dict.fromkeys(
            [
                *IDENTITY_COLUMNS,
                "s2n_lag_days",
                "s5n_lag_days",
                *feature_sets["v2_full"],
            ]
        )
    )
    print("[v3] loading V2 train/val/test tables", flush=True)
    frame = pd.concat(
        [
            pd.read_parquet(v2_directory / f"{split}.parquet", columns=load_columns)
            for split in ("train", "val", "test")
        ],
        ignore_index=True,
    )
    for column in ("feature_end_date", "eo_asof_date", "label_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    frame = frame.sort_values(["cell_id", "label_date"]).reset_index(drop=True)
    cells = int(frame["cell_id"].nunique())
    days = int(frame["label_date"].nunique())
    if len(frame) != cells * days:
        raise ValueError("V3 construction requires a complete cell-by-day grid")
    if not (
        (frame["label_date"] - frame["eo_asof_date"]).dt.days.eq(1).all()
        and (frame["eo_asof_date"] - frame["feature_end_date"]).dt.days.eq(5).all()
    ):
        raise ValueError("V3 source violates D+1 target / D-5 ERA5 contract")

    print("[v3] adding direction-aware spread and routing features", flush=True)
    frame, v3_features = add_v3_features(frame, cells, days)
    all_features = list(dict.fromkeys([*v2_features, *v3_features]))
    no_s5p_features = [
        column for column in all_features if not column.startswith("s5n_")
    ]
    if not np.isfinite(frame[all_features].to_numpy(dtype="float32")).all():
        raise ValueError("Non-finite V3 feature detected")

    output.mkdir(parents=True, exist_ok=True)
    output_columns = list(
        dict.fromkeys(
            [
                *IDENTITY_COLUMNS,
                "s2n_lag_days",
                "s5n_lag_days",
                *all_features,
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
        part = frame.loc[frame["label_date"].dt.year.isin(years), output_columns]
        destination = output / f"{split}.parquet"
        print(f"[v3] writing {split}: {len(part):,} rows", flush=True)
        part.to_parquet(destination, index=False)
        outputs[split] = {
            "path": str(destination.resolve()),
            "rows": len(part),
            "sha256": sha256(destination),
        }
    v3_sets = {
        "v2_selected_baseline": v2_features,
        "v3_directional": all_features,
        "v3_directional_no_s5p": no_s5p_features,
    }
    atomic_json(v3_sets, output / "feature_sets.json")
    atomic_json(
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "version": "V3",
            "source": str(v2_directory.resolve()),
            "source_feature_sets_sha256": sha256(
                v2_directory / "feature_sets.json"
            ),
            "rows": len(frame),
            "cells": cells,
            "days": days,
            "router": {
                "column": ROUTER_COLUMN,
                "context": (
                    "Same-cell or neighboring-cell fire in the seven-day "
                    "window ending T-2=D-1."
                ),
                "ignition": "No fire context in that causal window.",
            },
            "time_contract": {
                "target": "y_fire on T=D+1",
                "era5_cutoff": "D-5",
                "fire_history_cutoff": "T-2=D-1",
            },
            "feature_counts": {
                name: len(columns) for name, columns in v3_sets.items()
            },
            "new_features": v3_features,
            "split_outputs": outputs,
        },
        output / "metadata.json",
    )
    print(f"[v3] complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.v2_data.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
