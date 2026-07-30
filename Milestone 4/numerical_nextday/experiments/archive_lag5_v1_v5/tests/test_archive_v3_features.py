from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_archive_v3.py"
    spec = importlib.util.spec_from_file_location("build_archive_v3", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _two_cell_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["west"] * 3 + ["east"] * 3,
            "latitude": [35.0] * 6,
            "longitude": [-120.25] * 3 + [-120.0] * 3,
            "wind_dir_sin": [-1.0] * 6,
            "wind_dir_cos": [0.0] * 6,
            "wind_speed_mean": [5.0] * 6,
            "fire_cell_lag2": [0, 1, 0, 0, 0, 0],
            "fire_cell_count_7d_lag2": [0, 1, 1, 0, 0, 0],
            "fire_neighbor_count_7d_lag2": [0, 0, 0, 0, 1, 1],
            "fire_cell_any_7d_lag2": [0, 1, 1, 0, 0, 0],
            "fire_neighbor_any_7d_lag2": [0, 0, 0, 0, 1, 1],
            "vpd_kpa": [1.0] * 6,
            "soil_moisture_index": [0.2] * 6,
            "cvh_mean": [0.3] * 6,
            "cvl_mean": [0.4] * 6,
            "vpd_kpa_mean_14d": [1.2] * 6,
            "vpd_kpa_mean_30d": [1.0] * 6,
        }
    )


def test_upwind_fire_is_projected_toward_downwind_cell() -> None:
    module = _module()
    result, _ = module.add_v3_features(_two_cell_frame(), cells=2, days=3)
    east_day_two = result.iloc[4]
    assert east_day_two["fire_upwind_count_lag2"] > 0
    assert east_day_two["fire_downwind_count_7d_lag2"] == 0


def test_router_uses_only_already_lagged_fire_context() -> None:
    module = _module()
    result, _ = module.add_v3_features(_two_cell_frame(), cells=2, days=3)
    assert result[module.ROUTER_COLUMN].tolist() == [0, 1, 1, 0, 1, 1]


def test_directional_features_do_not_read_target_column() -> None:
    module = _module()
    base = _two_cell_frame()
    base["y_fire"] = 0
    changed = base.copy()
    changed.loc[4, "y_fire"] = 1
    left, features = module.add_v3_features(base, cells=2, days=3)
    right, _ = module.add_v3_features(changed, cells=2, days=3)
    pd.testing.assert_frame_equal(left[features], right[features])


def test_directional_feature_values_are_finite() -> None:
    module = _module()
    result, features = module.add_v3_features(_two_cell_frame(), cells=2, days=3)
    assert np.isfinite(result[features].to_numpy()).all()
