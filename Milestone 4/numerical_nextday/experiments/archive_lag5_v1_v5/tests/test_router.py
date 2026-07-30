from __future__ import annotations

import numpy as np
import pandas as pd

from numerical_nextday.train.router import MonthRouter


class ConstantModel:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.probability)


def test_month_router_preserves_row_order() -> None:
    frame = pd.DataFrame({"model_bucket": ["jan", "fire_season", "jan", "dec"]})
    router = MonthRouter(
        {
            "jan": ConstantModel(0.1),
            "fire_season": ConstantModel(0.8),
            "dec": ConstantModel(0.3),
        }
    )
    assert router.predict_proba(frame).tolist() == [0.1, 0.8, 0.1, 0.3]
