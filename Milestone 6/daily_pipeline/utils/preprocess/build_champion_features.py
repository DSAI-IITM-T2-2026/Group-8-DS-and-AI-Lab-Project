"""Champion feature engineering (ported from Wildfire_Training_final / V2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONSTANT_FEATURES = frozenset({"s5n_s5p_aai_std", "s5n_s5p_co_std"})

ID_COLUMNS = [
    "cell_id",
    "latitude",
    "longitude",
    "feature_end_date",
    "label_date",
    "eo_asof_date",
    "y_fire",
    "firms_n_pixels",
    "firms_max_confidence",
    "region",
    "era5_lag_days",
    "s2n_lag_days",
    "s5n_lag_days",
    "s5p_2021_status",
]

STAGE_C_FEATURE_PREFIXES = ("t2m_", "d2m_", "rh_", "sp_", "wind_", "i10fg_", "tp_", "swvl", "soil_", "cvh_", "cvl_", "lai_", "blh_", "elevation", "slope", "aspect_", "tri", "tpi", "orographic_", "hillshade", "s2n_", "s5n_")


def infer_base_features(columns: list[str]) -> list[str]:
    """Infer model base features from Stage C columns."""
    skip = set(ID_COLUMNS) | {"s2n_knn_imputed", "s5n_available"}
    base = [
        c
        for c in columns
        if c not in skip
        and (
            c.startswith(STAGE_C_FEATURE_PREFIXES)
            or c in {"soil_moisture_index", "t2m_max_7d", "tp_sum_7d", "wind_speed_max_7d", "rh_min_7d", "swvl1_mean_7d", "i10fg_max_7d"}
        )
    ]
    return [c for c in base if c not in CONSTANT_FEATURES]


def rolling_matrix(values: np.ndarray, window: int, operation: str) -> np.ndarray:
    rolling = pd.DataFrame(values.T).rolling(window=window, min_periods=1)
    return getattr(rolling, operation)().to_numpy(dtype="float32").T


def simple_neighbors(frame: pd.DataFrame, cells: int, days: int) -> list[np.ndarray]:
    coordinates = frame.iloc[np.arange(cells) * days][["latitude", "longitude"]].to_numpy(dtype="float64")
    result: list[np.ndarray] = []
    for latitude, longitude in coordinates:
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


def directional_geometry(coordinates: np.ndarray) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = []
    for latitude, longitude in coordinates:
        delta_lat = latitude - coordinates[:, 0]
        delta_lon = (longitude - coordinates[:, 1]) * np.cos(np.deg2rad(latitude))
        distance = np.sqrt(delta_lat**2 + delta_lon**2)
        adjacent = np.flatnonzero((distance > 1e-9) & (distance <= 0.36))
        result.append((
            adjacent,
            (delta_lon[adjacent] / distance[adjacent]).astype("float32"),
            (delta_lat[adjacent] / distance[adjacent]).astype("float32"),
            (1.0 / (distance[adjacent] + 0.05)).astype("float32"),
        ))
    return result


def directional_counts(
    fire: np.ndarray,
    wind_sin: np.ndarray,
    wind_cos: np.ndarray,
    geometry: list[tuple[Any, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cells, days = fire.shape
    upwind = np.zeros((cells, days), dtype="float32")
    downwind = np.zeros((cells, days), dtype="float32")
    crosswind = np.zeros((cells, days), dtype="float32")
    distance_weighted = np.zeros((cells, days), dtype="float32")
    wind_east, wind_north = -wind_sin, -wind_cos
    for index, (adjacent, east, north, inverse_distance) in enumerate(geometry):
        if not adjacent.size:
            continue
        neighbor_fire = fire[adjacent]
        alignment = wind_east[index][None, :] * east[:, None] + wind_north[index][None, :] * north[:, None]
        upwind[index] = (neighbor_fire * np.maximum(alignment, 0) * inverse_distance[:, None]).sum(axis=0)
        downwind[index] = (neighbor_fire * np.maximum(-alignment, 0) * inverse_distance[:, None]).sum(axis=0)
        crosswind[index] = (
            neighbor_fire * np.sqrt(np.maximum(1.0 - alignment**2, 0)) * inverse_distance[:, None]
        ).sum(axis=0)
        distance_weighted[index] = (neighbor_fire * inverse_distance[:, None]).sum(axis=0)
    return upwind, downwind, crosswind, distance_weighted


def build_features(
    frame: pd.DataFrame,
    raw_base_features: list[str],
    *,
    use_neighbor_fire_features: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Engineer 107 features (neighbor ON) from Stage C history panel."""
    frame = frame.copy()
    for col in ("label_date", "eo_asof_date", "feature_end_date"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col]).dt.normalize()

    frame = frame.sort_values(["cell_id", "label_date"]).reset_index(drop=True)
    cells = int(frame["cell_id"].nunique())
    days = int(frame["label_date"].nunique())

    day_of_year = frame["eo_asof_date"].dt.dayofyear.to_numpy(dtype="float32")
    month = frame["eo_asof_date"].dt.month.to_numpy(dtype="float32")
    calendar = pd.DataFrame({
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }, index=frame.index).astype("float32")
    frame = pd.concat([frame, calendar], axis=1)

    temperature_c = frame["t2m_mean"].to_numpy(dtype="float64") - 273.15
    dewpoint_c = frame["d2m_mean"].to_numpy(dtype="float64") - 273.15
    saturation = 0.6108 * np.exp(17.27 * temperature_c / np.maximum(temperature_c + 237.3, 1e-6))
    actual = 0.6108 * np.exp(17.27 * dewpoint_c / np.maximum(dewpoint_c + 237.3, 1e-6))
    vpd_arr = np.maximum(saturation - actual, 0).astype("float32")
    weather: dict[str, np.ndarray] = {
        "vpd_kpa": vpd_arr,
        "vpd_wind_interaction": vpd_arr * frame["wind_speed_mean"].to_numpy(dtype="float32"),
        "vpd_soil_deficit_interaction": vpd_arr * (1 - np.clip(frame["soil_moisture_index"], 0, 1)),
        "heat_soil_deficit_interaction": (
            np.maximum(frame["t2m_max"].to_numpy(dtype="float32") - 273.15, 0)
            * (1 - np.clip(frame["swvl1_mean"], 0, 1))
        ),
        "wind_gust_ratio": frame["i10fg_max"].to_numpy(dtype="float32")
        / (frame["wind_speed_mean"].to_numpy(dtype="float32") + 0.1),
    }
    rolling_specs = {
        "t2m_max": ("max",),
        "rh_mean": ("min",),
        "tp_sum_mm": ("sum",),
        "wind_speed_mean": ("max",),
        "i10fg_max": ("max",),
        "swvl1_mean": ("mean",),
        "vpd_kpa": ("max", "mean"),
    }
    arrays = {
        name: (weather[name] if name in weather else frame[name].to_numpy(dtype="float32")).reshape(cells, days)
        for name in rolling_specs
    }
    for name, operations in rolling_specs.items():
        for window in (14, 30):
            for operation in operations:
                weather[f"{name}_{operation}_{window}d"] = rolling_matrix(
                    arrays[name], window, operation
                ).reshape(-1)
    temperature = frame["t2m_max"].to_numpy(dtype="float32").reshape(cells, days)
    soil = frame["swvl1_mean"].to_numpy(dtype="float32").reshape(cells, days)
    weather["t2m_max_anomaly_30d"] = (temperature - rolling_matrix(temperature, 30, "mean")).reshape(-1)
    weather["swvl1_anomaly_30d"] = (soil - rolling_matrix(soil, 30, "mean")).reshape(-1)
    weather_frame = pd.DataFrame(weather, index=frame.index).astype("float32")
    frame = pd.concat([frame, weather_frame], axis=1)

    wind = frame["wind_speed_mean"].to_numpy(dtype="float32")
    vpd = frame["vpd_kpa"].to_numpy(dtype="float32")
    soil_deficit = 1 - np.clip(frame["soil_moisture_index"].to_numpy(dtype="float32"), 0, 1)
    vegetation = np.clip(
        frame["cvh_mean"].to_numpy(dtype="float32") + frame["cvl_mean"].to_numpy(dtype="float32"), 0, 1
    )
    context = {
        "ignition_dry_windy_index": vpd * wind * soil_deficit,
        "fuel_dryness_index": vpd * soil_deficit * vegetation,
        "vpd_short_long_trend": frame["vpd_kpa_mean_14d"] - frame["vpd_kpa_mean_30d"],
    }
    context_frame = pd.DataFrame(context, index=frame.index).astype("float32")
    frame = pd.concat([frame, context_frame], axis=1)

    neighbor_frame = pd.DataFrame(index=frame.index)
    wind_fire_frame = pd.DataFrame(index=frame.index)

    if use_neighbor_fire_features:
        target = frame["y_fire"].fillna(0).to_numpy(dtype="float32").reshape(cells, days)
        lag2 = np.zeros_like(target, dtype="float32")
        lag2[:, 2:] = target[:, :-2]
        history7 = rolling_matrix(lag2, 7, "sum")
        neighbors = simple_neighbors(frame, cells, days)
        neighbor_lag2 = sum_neighbors(lag2, neighbors)
        neighbor_7d = sum_neighbors(history7, neighbors)
        neighbor = {
            "fire_neighbor_count_lag2": neighbor_lag2.reshape(-1),
            "fire_neighbor_count_7d_lag2": neighbor_7d.reshape(-1),
            "fire_neighbor_any_7d_lag2": (neighbor_7d > 0).astype("float32").reshape(-1),
        }
        neighbor_frame = pd.DataFrame(neighbor, index=frame.index).astype("float32")
        frame = pd.concat([frame, neighbor_frame], axis=1)

        coordinates = frame.iloc[np.arange(cells) * days][["latitude", "longitude"]].to_numpy(dtype="float64")
        geometry = directional_geometry(coordinates)
        wind_sin = frame["wind_dir_sin"].to_numpy(dtype="float32").reshape(cells, days)
        wind_cos = frame["wind_dir_cos"].to_numpy(dtype="float32").reshape(cells, days)
        upwind_lag2, _, _, distance_lag2 = directional_counts(lag2, wind_sin, wind_cos, geometry)
        upwind_7d, downwind_7d, crosswind_7d, distance_7d = directional_counts(
            history7, wind_sin, wind_cos, geometry
        )
        recent_neighbor = frame["fire_neighbor_any_7d_lag2"].to_numpy(dtype="float32")
        wind_fire = {
            "fire_upwind_count_lag2": upwind_lag2.reshape(-1),
            "fire_upwind_count_7d_lag2": upwind_7d.reshape(-1),
            "fire_downwind_count_7d_lag2": downwind_7d.reshape(-1),
            "fire_crosswind_count_7d_lag2": crosswind_7d.reshape(-1),
            "fire_distance_weighted_count_lag2": distance_lag2.reshape(-1),
            "fire_distance_weighted_count_7d_lag2": distance_7d.reshape(-1),
            "fire_wind_spread_potential_lag2": upwind_lag2.reshape(-1) * wind,
            "fire_wind_spread_potential_7d_lag2": upwind_7d.reshape(-1) * wind,
            "fire_context_vpd_interaction": frame["fire_neighbor_count_7d_lag2"].to_numpy(dtype="float32") * vpd,
            "fire_context_dry_windy_interaction": recent_neighbor * vpd * wind * soil_deficit,
            "recent_neighbor_fire_context": recent_neighbor,
        }
        wind_fire_frame = pd.DataFrame(wind_fire, index=frame.index).astype("float32")
        frame = pd.concat([frame, wind_fire_frame], axis=1)

    selected_base = [name for name in raw_base_features if name != "s5n_available"]
    feature_columns = list(dict.fromkeys([
        *selected_base,
        "latitude",
        "longitude",
        *calendar.columns,
        *weather_frame.columns,
        *context_frame.columns,
        *neighbor_frame.columns,
        *wind_fire_frame.columns,
    ]))
    groups = {
        "source_used_by_model": len(selected_base),
        "total": len(feature_columns),
        "use_neighbor_fire_features": int(use_neighbor_fire_features),
    }
    return frame, feature_columns, groups


def apply_champion_prune(
    frame: pd.DataFrame,
    feature_columns: list[str],
    kept_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in kept_features if c not in frame.columns]
    if missing:
        raise ValueError(f"Champion features missing from engineered frame: {missing[:5]}...")
    if len(kept_features) != 86:
        raise ValueError(f"Expected 86 kept features, got {len(kept_features)}")
    out_cols = [
        c for c in [
            "cell_id", "label_date", "eo_asof_date", "feature_end_date",
            "latitude", "longitude", "y_fire",
        ] if c in frame.columns
    ]
    # lat/lon are also in the 86-feature keep list; write each column once.
    pruned = frame[list(dict.fromkeys([*out_cols, *kept_features]))].copy()
    return pruned, kept_features


def load_cell_subset(
    fire_region_csv: str | None,
    categories: list[str] | None = None,
    *,
    mode: str | None = None,
) -> set[str] | None:
    """Return cell_ids for a fire-region subset, or None for all cells.

    Modes (from config preprocess.cell_subset):
      - all | none | off  → no filter (full ERA5 panel for that day)
      - high_medium_fire / high_medium → High Outlier + High + Medium (~439)
      - high_only / high → High Outlier + High (~146)
      - or pass explicit categories=
    """
    if not fire_region_csv:
        return None

    mode_norm = (mode or "").strip().lower().replace("-", "_")
    if mode_norm in {"all", "none", "off", ""}:
        if categories is None:
            return None

    presets = {
        "high_medium_fire": ["High Outlier", "High", "Medium"],
        "high_medium": ["High Outlier", "High", "Medium"],
        "high_only": ["High Outlier", "High"],
        "high": ["High Outlier", "High"],
    }
    cats = categories
    if cats is None:
        cats = presets.get(mode_norm, ["High Outlier", "High", "Medium"])

    path = Path(fire_region_csv)
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    cat_col = "Category" if "Category" in df.columns else (
        "fire_region_category" if "fire_region_category" in df.columns else None
    )
    if cat_col is None or "cell_id" not in df.columns:
        return None
    return set(df.loc[df[cat_col].isin(cats), "cell_id"].astype(str))
