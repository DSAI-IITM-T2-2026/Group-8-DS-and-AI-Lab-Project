from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .lightgbm_model import LightGBMBundle
from .preprocessing import TabularPreprocessor


@dataclass
class V3ClassifierBundle:
    architecture: str
    router_column: str
    global_model: LightGBMBundle | None = None
    context_model: LightGBMBundle | None = None
    ignition_model: LightGBMBundle | None = None

    def _expert_masks(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        context = frame[self.router_column].to_numpy(dtype=float) > 0.5
        return context, ~context

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        if self.architecture == "global":
            if self.global_model is None:
                raise RuntimeError("V3 global model is missing")
            return self.global_model.predict_raw(frame)
        if self.context_model is None or self.ignition_model is None:
            raise RuntimeError("V3 mixture experts are missing")
        result = np.empty(len(frame), dtype=float)
        context, ignition = self._expert_masks(frame)
        result[context] = self.context_model.predict_raw(frame.loc[context])
        result[ignition] = self.ignition_model.predict_raw(frame.loc[ignition])
        return result

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.architecture == "global":
            if self.global_model is None:
                raise RuntimeError("V3 global model is missing")
            return self.global_model.predict_proba(frame)
        if self.context_model is None or self.ignition_model is None:
            raise RuntimeError("V3 mixture experts are missing")
        result = np.empty(len(frame), dtype=float)
        context, ignition = self._expert_masks(frame)
        result[context] = self.context_model.predict_proba(frame.loc[context])
        result[ignition] = self.ignition_model.predict_proba(frame.loc[ignition])
        return result

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: Path) -> V3ClassifierBundle:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} is not a V3ClassifierBundle")
        return loaded


@dataclass
class V3RankerBundle:
    booster: lgb.Booster
    preprocessor: TabularPreprocessor
    feature_columns: list[str]
    best_iteration: int

    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame[self.feature_columns])
        return np.asarray(
            self.booster.predict(matrix, num_iteration=self.best_iteration),
            dtype=float,
        )

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: Path) -> V3RankerBundle:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} is not a V3RankerBundle")
        return loaded
