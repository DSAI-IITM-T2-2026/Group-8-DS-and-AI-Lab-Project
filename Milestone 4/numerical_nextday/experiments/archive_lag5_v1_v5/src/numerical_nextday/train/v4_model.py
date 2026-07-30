from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .lightgbm_model import LightGBMBundle
from .v3_model import V3RankerBundle


def within_day_percentile(
    scores: np.ndarray, label_dates: pd.Series
) -> np.ndarray:
    table = pd.DataFrame(
        {
            "label_date": pd.to_datetime(label_dates).to_numpy(),
            "score": np.asarray(scores, dtype=float),
            "position": np.arange(len(scores)),
        }
    )
    table["percentile"] = table.groupby("label_date", sort=False)["score"].rank(
        method="average", pct=True
    )
    return table.sort_values("position")["percentile"].to_numpy(dtype=float)


@dataclass
class V4AlertBundle:
    classifier: LightGBMBundle
    ranker: V3RankerBundle | None
    alert_head: str
    classifier_weight: float = 1.0

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        raw = self.classifier.predict_raw(frame)
        calibrated = self.classifier.predict_proba(frame)
        if self.ranker is None:
            rank_score = np.full(len(frame), np.nan, dtype=float)
        else:
            rank_score = self.ranker.predict_score(frame)
        if self.alert_head == "classifier":
            alert_score = raw
        elif self.alert_head == "ranker":
            alert_score = rank_score
        elif self.alert_head == "blend":
            if self.ranker is None:
                raise RuntimeError("V4 blend requires a fitted ranker")
            classifier_rank = within_day_percentile(raw, frame["label_date"])
            ranker_rank = within_day_percentile(
                rank_score, frame["label_date"]
            )
            alert_score = (
                self.classifier_weight * classifier_rank
                + (1 - self.classifier_weight) * ranker_rank
            )
        else:
            raise ValueError(f"Unknown V4 alert head: {self.alert_head}")
        return {
            "p_fire_raw": raw,
            "p_fire": calibrated,
            "rank_score": rank_score,
            "alert_score": np.asarray(alert_score, dtype=float),
        }

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: Path) -> V4AlertBundle:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} is not a V4AlertBundle")
        return loaded
