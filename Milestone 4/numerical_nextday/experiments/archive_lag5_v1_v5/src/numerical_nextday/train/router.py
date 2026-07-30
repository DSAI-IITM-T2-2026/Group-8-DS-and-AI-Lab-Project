from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonthRouter:
    models: dict[str, object]
    fallback_bucket: str = "fire_season"

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if "model_bucket" not in frame:
            raise ValueError("Inference frame lacks model_bucket")
        probability = np.full(len(frame), np.nan, dtype=float)
        for bucket, indices in frame.groupby("model_bucket").groups.items():
            model = self.models.get(bucket) or self.models.get(self.fallback_bucket)
            if model is None:
                raise KeyError(f"No model or fallback is available for bucket={bucket}")
            positions = frame.index.get_indexer(indices)
            probability[positions] = model.predict_proba(frame.loc[indices])
        if not np.isfinite(probability).all():
            raise RuntimeError("Router failed to score every row")
        return probability
