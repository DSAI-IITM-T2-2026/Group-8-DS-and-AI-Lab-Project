from __future__ import annotations

import pandas as pd

from numerical_nextday.evaluation.metrics import complete_metrics


def test_top_k_metrics_are_daily_and_full_population() -> None:
    scored = pd.DataFrame(
        {
            "label_date": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"] * 3),
            "y_fire": [1, 0, 0, 0, 1, 0],
            "p_fire": [0.9, 0.8, 0.1, 0.8, 0.7, 0.2],
        }
    )
    metrics = complete_metrics(scored, top_k=1)
    assert metrics["alert_count"] == 2
    assert metrics["alert_precision"] == 0.5
    assert metrics["positive_recall"] == 0.5
    assert metrics["false_alerts_per_day"] == 0.5
