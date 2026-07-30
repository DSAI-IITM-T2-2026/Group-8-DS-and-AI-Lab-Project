from __future__ import annotations

import numpy as np
import pandas as pd

from numerical_nextday.train.v5_model import (
    META_FEATURES,
    add_retrieval_meta,
    candidate_positions,
)


def test_candidate_pool_is_selected_separately_per_day() -> None:
    frame = pd.DataFrame(
        {
            "label_date": pd.to_datetime(
                ["2024-01-01"] * 3 + ["2024-01-02"] * 3
            ),
            "v4_retrieval_score": [0.1, 0.9, 0.5, 0.8, 0.2, 0.7],
        }
    )
    positions = candidate_positions(frame, "v4_retrieval_score", pool_size=2)
    assert set(positions) == {1, 2, 3, 5}


def test_retrieval_meta_contains_all_declared_features() -> None:
    frame = pd.DataFrame(
        {
            "label_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            )
        }
    )
    predictions = {
        "p_fire_raw": np.array([0.1, 0.2, 0.3, 0.4]),
        "rank_score": np.array([1.0, 0.0, 2.0, 3.0]),
        "alert_score": np.array([0.4, 0.6, 0.2, 0.8]),
    }
    result = add_retrieval_meta(frame, predictions)
    assert set(META_FEATURES).issubset(result.columns)
    assert np.isfinite(result[META_FEATURES].to_numpy()).all()
