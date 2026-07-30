from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .calibration import fit_calibrator
from .preprocessing import TabularPreprocessor


@dataclass
class LightGBMBundle:
    booster: lgb.Booster
    calibrator: Any
    preprocessor: TabularPreprocessor
    feature_columns: list[str]
    stage: str
    model_bucket: str
    params: dict[str, Any]
    random_seed: int
    best_iteration: int
    calibration_note: str

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame[self.feature_columns])
        return np.asarray(
            self.booster.predict(
                matrix,
                num_iteration=self.best_iteration or self.booster.best_iteration,
            ),
            dtype=float,
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.calibrator.predict(self.predict_raw(frame)), dtype=float)

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: Path) -> LightGBMBundle:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} is not a LightGBMBundle")
        return loaded


def lightgbm_params(cfg: dict, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    training = cfg["training"]
    params = {
        "objective": "binary",
        "metric": ["average_precision", "auc"],
        "learning_rate": float(training["learning_rate"]),
        "num_leaves": int(training["num_leaves"]),
        "min_data_in_leaf": int(training["min_data_in_leaf"]),
        "feature_fraction": float(training["feature_fraction"]),
        "bagging_fraction": float(training["bagging_fraction"]),
        "bagging_freq": int(training["bagging_freq"]),
        "lambda_l2": float(training["lambda_l2"]),
        "verbosity": -1,
        "seed": int(cfg["project"]["random_seed"]),
        "feature_pre_filter": False,
        "num_threads": -1,
    }
    if overrides:
        params.update(overrides)
    return params


def fit_lightgbm(
    cfg: dict,
    train: pd.DataFrame,
    tune: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: list[str],
    stage: str,
    model_bucket: str,
    overrides: dict[str, Any] | None = None,
) -> LightGBMBundle:
    if train.empty or train["y_fire"].nunique() < 2:
        raise ValueError(f"{model_bucket} training data must contain both target classes")
    if tune.empty or tune["y_fire"].nunique() < 2:
        raise ValueError(f"{model_bucket} tune data must contain both target classes")
    preprocessor = TabularPreprocessor(feature_columns).fit(train, scale=False)
    x_train = preprocessor.transform(train)
    x_tune = preprocessor.transform(tune)
    y_train = train["y_fire"].astype(int).to_numpy()
    y_tune = tune["y_fire"].astype(int).to_numpy()
    positives = max(int(y_train.sum()), 1)
    negatives = max(int(len(y_train) - positives), 1)
    params = lightgbm_params(cfg, overrides)
    params.setdefault("scale_pos_weight", negatives / positives)
    training_data = lgb.Dataset(
        x_train, label=y_train, feature_name=feature_columns, free_raw_data=False
    )
    tune_data = lgb.Dataset(
        x_tune,
        label=y_tune,
        reference=training_data,
        feature_name=feature_columns,
        free_raw_data=False,
    )
    booster = lgb.train(
        params,
        training_data,
        num_boost_round=int(cfg["training"]["num_boost_round"]),
        valid_sets=[tune_data],
        valid_names=["tune"],
        callbacks=[
            lgb.early_stopping(int(cfg["training"]["early_stopping_rounds"]), verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    raw_calibration = np.asarray(
        booster.predict(
            preprocessor.transform(calibration),
            num_iteration=booster.best_iteration,
        )
    )
    calibrator = fit_calibrator(
        raw_calibration,
        calibration["y_fire"].astype(int).to_numpy(),
        int(cfg["model_buckets"]["min_calibration_positives"]),
    )
    note = getattr(calibrator, "reason", "isotonic")
    return LightGBMBundle(
        booster=booster,
        calibrator=calibrator,
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        stage=stage,
        model_bucket=model_bucket,
        params=params,
        random_seed=int(cfg["project"]["random_seed"]),
        best_iteration=int(booster.best_iteration),
        calibration_note=note,
    )
