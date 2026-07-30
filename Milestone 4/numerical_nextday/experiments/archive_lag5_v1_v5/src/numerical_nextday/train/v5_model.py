from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .v3_model import V3RankerBundle
from .v4_model import V4AlertBundle, within_day_percentile


META_FEATURES = [
    "v4_classifier_score",
    "v4_ranker_score",
    "v4_retrieval_score",
    "v4_classifier_percentile",
    "v4_ranker_percentile",
    "v4_retrieval_percentile",
]


def add_retrieval_meta(
    frame: pd.DataFrame, base_predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    result = frame.copy()
    result["v4_classifier_score"] = base_predictions["p_fire_raw"]
    result["v4_ranker_score"] = base_predictions["rank_score"]
    result["v4_retrieval_score"] = base_predictions["alert_score"]
    result["v4_classifier_percentile"] = within_day_percentile(
        base_predictions["p_fire_raw"], result["label_date"]
    )
    result["v4_ranker_percentile"] = within_day_percentile(
        base_predictions["rank_score"], result["label_date"]
    )
    result["v4_retrieval_percentile"] = within_day_percentile(
        base_predictions["alert_score"], result["label_date"]
    )
    return result


def candidate_positions(
    frame: pd.DataFrame, score_column: str, pool_size: int
) -> np.ndarray:
    ranking = pd.DataFrame(
        {
            "label_date": pd.to_datetime(frame["label_date"]).to_numpy(),
            "score": frame[score_column].to_numpy(dtype=float),
            "position": np.arange(len(frame)),
        }
    )
    selected = (
        ranking.sort_values(
            ["label_date", "score"], ascending=[True, False], kind="stable"
        )
        .groupby("label_date", sort=False)
        .head(pool_size)
    )
    return selected["position"].to_numpy(dtype=int)


@dataclass
class V5TwoStageBundle:
    retrieval_bundle: V4AlertBundle
    reranker: V3RankerBundle
    candidate_pool_size: int

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        base = self.retrieval_bundle.predict(frame)
        enriched = add_retrieval_meta(frame, base)
        positions = candidate_positions(
            enriched, "v4_retrieval_score", self.candidate_pool_size
        )
        candidate_score = self.reranker.predict_score(enriched.iloc[positions])
        rerank_score = np.full(len(frame), -1e9, dtype=float)
        rerank_score[positions] = candidate_score
        return {
            **base,
            "rerank_score": rerank_score,
            "alert_score": rerank_score,
            "candidate_position": np.isin(
                np.arange(len(frame)), positions
            ).astype("int8"),
        }

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: Path) -> V5TwoStageBundle:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} is not a V5TwoStageBundle")
        return loaded
