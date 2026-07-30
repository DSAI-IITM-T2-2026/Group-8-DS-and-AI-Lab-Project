from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_archive_v2.py"
    spec = importlib.util.spec_from_file_location("build_archive_v2", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weather_rolling_does_not_look_forward() -> None:
    module = _module()
    values = np.array([[1.0, 2.0, 100.0]], dtype="float32")
    result = module.rolling_matrix(values, window=2, operation="sum")
    np.testing.assert_allclose(result, [[1.0, 3.0, 102.0]])


def test_fire_history_uses_two_day_label_lag() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "cell_id": ["a"] * 6,
            "latitude": [35.0] * 6,
            "longitude": [-120.0] * 6,
            "y_fire": [0, 0, 1, 0, 0, 0],
        }
    )
    result, _ = module.add_causal_fire_history(frame, cells=1, days=6)
    assert result["fire_cell_lag2"].tolist() == [0, 0, 0, 0, 1, 0]
    assert result.loc[:3, "fire_cell_count_7d_lag2"].sum() == 0


def test_future_target_mutation_cannot_change_earlier_fire_features() -> None:
    module = _module()
    base = pd.DataFrame(
        {
            "cell_id": ["a"] * 8,
            "latitude": [35.0] * 8,
            "longitude": [-120.0] * 8,
            "y_fire": [0] * 8,
        }
    )
    changed = base.copy()
    changed.loc[4, "y_fire"] = 1
    left, features = module.add_causal_fire_history(base, cells=1, days=8)
    right, _ = module.add_causal_fire_history(changed, cells=1, days=8)
    pd.testing.assert_frame_equal(left.loc[:5, features], right.loc[:5, features])
    assert right.loc[6, "fire_cell_lag2"] == 1
