from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularPreprocessor:
    feature_columns: list[str]
    medians: pd.Series | None = None
    scaler: StandardScaler | None = None

    def fit(self, frame: pd.DataFrame, scale: bool = False) -> TabularPreprocessor:
        numeric = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median(axis=0).fillna(0.0)
        self.medians = medians
        values = numeric.fillna(medians).to_numpy(dtype="float64")
        self.scaler = StandardScaler().fit(values) if scale else None
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("Preprocessor has not been fitted")
        numeric = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        values = numeric.fillna(self.medians).to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite feature remains after preprocessing")
        return self.scaler.transform(values) if self.scaler is not None else values
