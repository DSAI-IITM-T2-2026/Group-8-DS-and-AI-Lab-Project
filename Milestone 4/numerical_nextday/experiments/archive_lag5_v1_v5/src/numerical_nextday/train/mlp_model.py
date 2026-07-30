from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from .calibration import fit_calibrator
from .preprocessing import TabularPreprocessor


@dataclass
class MLPBundle:
    model: MLPClassifier
    calibrator: Any
    preprocessor: TabularPreprocessor
    feature_columns: list[str]
    stage: str
    model_bucket: str
    random_seed: int

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(self.preprocessor.transform(frame[self.feature_columns]))[
            :, 1
        ]
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)


def fit_mlp(
    cfg: dict,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: list[str],
    stage: str,
    model_bucket: str,
) -> MLPBundle:
    mlp_cfg = cfg["training"]["mlp"]
    preprocessor = TabularPreprocessor(feature_columns).fit(train, scale=True)
    model = MLPClassifier(
        hidden_layer_sizes=tuple(int(v) for v in mlp_cfg["hidden_layers"]),
        max_iter=int(mlp_cfg["max_epochs"]),
        batch_size=int(mlp_cfg["batch_size"]),
        n_iter_no_change=int(mlp_cfg["patience"]),
        learning_rate_init=float(mlp_cfg["learning_rate_init"]),
        alpha=float(mlp_cfg["alpha"]),
        early_stopping=True,
        validation_fraction=0.1,
        random_state=int(cfg["project"]["random_seed"]),
    )
    model.fit(
        preprocessor.transform(train),
        train["y_fire"].astype(int).to_numpy(),
    )
    raw = model.predict_proba(preprocessor.transform(calibration))[:, 1]
    calibrator = fit_calibrator(
        raw,
        calibration["y_fire"].astype(int).to_numpy(),
        int(cfg["model_buckets"]["min_calibration_positives"]),
    )
    return MLPBundle(
        model=model,
        calibrator=calibrator,
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        stage=stage,
        model_bucket=model_bucket,
        random_seed=int(cfg["project"]["random_seed"]),
    )
