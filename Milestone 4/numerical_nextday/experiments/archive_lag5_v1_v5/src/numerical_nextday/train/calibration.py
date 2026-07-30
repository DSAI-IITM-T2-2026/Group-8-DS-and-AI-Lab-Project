from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class IdentityCalibrator:
    reason: str = "calibration skipped"

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probability, dtype=float), 0, 1)


@dataclass
class ConstantCalibrator:
    probability: float
    reason: str

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return np.full(len(probability), self.probability, dtype=float)


@dataclass
class PlattCalibrator:
    model: LogisticRegression
    reason: str = "platt logistic calibration"

    def predict(self, probability: np.ndarray) -> np.ndarray:
        raw = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
        logit = np.log(raw / (1 - raw)).reshape(-1, 1)
        return self.model.predict_proba(logit)[:, 1]


def fit_calibrator(raw_probability: np.ndarray, y_true: np.ndarray, min_positives: int):
    y = np.asarray(y_true, dtype=int)
    raw = np.asarray(raw_probability, dtype=float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if len(y) == 0:
        return IdentityCalibrator("empty calibration split")
    if positives < min_positives or negatives < min_positives:
        return ConstantCalibrator(
            probability=float(y.mean()),
            reason=(
                f"insufficient calibration classes: positives={positives}, negatives={negatives}"
            ),
        )
    if np.unique(raw).size < 3:
        return IdentityCalibrator("fewer than three distinct raw probabilities")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw, y)
    return calibrator


def fit_platt_calibrator(
    raw_probability: np.ndarray, y_true: np.ndarray, min_positives: int
):
    y = np.asarray(y_true, dtype=int)
    raw = np.clip(np.asarray(raw_probability, dtype=float), 1e-7, 1 - 1e-7)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if len(y) == 0:
        return IdentityCalibrator("empty calibration split")
    if positives < min_positives or negatives < min_positives:
        return ConstantCalibrator(
            probability=float(y.mean()),
            reason=(
                f"insufficient calibration classes: positives={positives}, "
                f"negatives={negatives}"
            ),
        )
    logit = np.log(raw / (1 - raw)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit, y)
    return PlattCalibrator(model=model)
