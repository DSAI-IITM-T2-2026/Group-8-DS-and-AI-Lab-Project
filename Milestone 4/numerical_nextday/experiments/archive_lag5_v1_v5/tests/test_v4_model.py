from __future__ import annotations

import numpy as np
import pandas as pd

from numerical_nextday.train.v4_model import within_day_percentile


def test_within_day_percentile_does_not_mix_days() -> None:
    dates = pd.Series(
        pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3)
    )
    scores = np.array([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    result = within_day_percentile(scores, dates)
    np.testing.assert_allclose(
        result, [1 / 3, 2 / 3, 1.0, 1 / 3, 2 / 3, 1.0]
    )


def test_within_day_percentile_preserves_row_order() -> None:
    dates = pd.Series(
        pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"])
    )
    scores = np.array([5.0, 99.0, 1.0])
    result = within_day_percentile(scores, dates)
    np.testing.assert_allclose(result, [1.0, 1.0, 0.5])
