"""Loads the real trained model artifact, if one has been configured.

The champion model (LightGBM classifier + XGBoost blend, calibrated,
86 features -- see Milestone 5/metrics_summary.json and
Milestone 5/Wildfire_Inference.ipynb) is produced by an external Kaggle
training notebook as ``wildfire_model.joblib`` and is **not checked into
this repository** (it is a multi-hundred-MB artifact). This module never
fabricates a substitute: if no artifact is configured/loadable, every
caller gets a clear "unavailable" signal instead of invented numbers.

To enable real inference, set ``WILDFIRE_MODEL_ARTIFACT`` to a local path
of a ``wildfire_model.joblib`` produced by that notebook and restart the
API. The expected artifact shape (verified against the inference
notebook) is a dict with at least:

    {
      "classifier_pipeline": sklearn Pipeline,      # named_steps: feature_preprocessor, fire_probability_model
      "ranker_pipeline": sklearn Pipeline,
      "feature_columns": list[str],                 # the 86 champion features, in order
      "source_stage": "stage_c_knn",
      "imputation_method": "precomputed KNN",
    }
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_settings


class ModelUnavailable(RuntimeError):
    """Raised whenever inference/explanation is requested but no real model is loaded."""


@dataclass
class ScoredBatch:
    probability: np.ndarray  # calibrated P(fire) from the classifier pipeline, one per row
    rank_score: np.ndarray | None  # ranker pipeline output, for percentile-based riskClass bucketing


class ModelRegistry:
    def __init__(self, artifact_path: Path | None) -> None:
        self.artifact_path = artifact_path
        self._artifact: dict[str, Any] | None = None
        self._error: str | None = None
        self._load()

    def _load(self) -> None:
        if self.artifact_path is None:
            self._error = (
                "WILDFIRE_MODEL_ARTIFACT is not configured. Point it at a wildfire_model.joblib "
                "produced by Milestone 5/Wildfire_Training_final.ipynb to enable inference endpoints."
            )
            return
        if not self.artifact_path.is_file():
            self._error = f"Configured model artifact does not exist: {self.artifact_path}"
            return
        try:
            import joblib

            artifact = joblib.load(self.artifact_path)
        except Exception as exc:  # pragma: no cover - depends on optional heavy deps
            self._error = f"Failed to load model artifact ({self.artifact_path.name}): {exc}"
            return

        missing = [k for k in ("classifier_pipeline", "feature_columns") if k not in artifact]
        if missing:
            self._error = f"Model artifact is missing required keys: {missing}"
            return
        self._artifact = artifact

    @property
    def is_loaded(self) -> bool:
        return self._artifact is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._error

    def require_loaded(self) -> dict[str, Any]:
        if self._artifact is None:
            raise ModelUnavailable(self._error or "Model is not loaded.")
        return self._artifact

    @property
    def feature_columns(self) -> list[str]:
        return list(self.require_loaded()["feature_columns"])

    @property
    def version(self) -> str:
        """Architecture identity + artifact content hash (real, not invented)."""
        base = "champion-lgbm-xgb-blend-86f"
        if self._artifact is None or self.artifact_path is None:
            return f"{base} (unloaded)"
        digest = hashlib.sha256(self.artifact_path.read_bytes()).hexdigest()[:12]
        return f"{base}-{digest}"

    @property
    def explanation_capability(self) -> str:
        if not self.is_loaded:
            return "unavailable"
        pipe = self._artifact.get("classifier_pipeline")
        try:
            model_step = pipe.named_steps.get("fire_probability_model")
            booster = getattr(model_step, "lgbm_model", model_step)
            return "available" if hasattr(booster, "booster_") else "unknown"
        except Exception:
            return "unknown"

    def score(self, frame: pd.DataFrame) -> ScoredBatch:
        """Real classifier probability + (optional) ranker score for a feature frame.

        Uses the artifact's own fitted sklearn Pipelines exactly as
        Milestone 5/Wildfire_Inference.ipynb does -- no re-derivation of
        training-time math. The classifier pipeline's calibrated
        ``predict_proba`` is reported to the frontend as ``probability``;
        the ranker pipeline (when present) is used only to bucket
        ``riskClass`` by daily percentile, matching the "daily_blend"
        percentiles recorded in the champion feature contract.
        """
        artifact = self.require_loaded()
        feature_cols = artifact["feature_columns"]
        missing = [c for c in feature_cols if c not in frame.columns]
        if missing:
            raise ModelUnavailable(f"Feature snapshot is missing columns required by the model: {missing[:5]}")
        x = frame[feature_cols]

        clf_pipe = artifact["classifier_pipeline"]
        probability = np.asarray(clf_pipe.predict_proba(x))[:, 1]

        rank_score = None
        ranker_pipe = artifact.get("ranker_pipeline")
        if ranker_pipe is not None:
            try:
                if hasattr(ranker_pipe, "predict_proba"):
                    rank_score = np.asarray(ranker_pipe.predict_proba(x))[:, 1]
                else:
                    rank_score = np.asarray(ranker_pipe.predict(x), dtype=float)
            except Exception:
                rank_score = None
        return ScoredBatch(probability=probability, rank_score=rank_score)

    def explain(self, frame: pd.DataFrame) -> dict[str, Any] | None:
        """TreeSHAP feature contributions, mirroring the inference notebook's SHAP cell.

        Returns None (never a fabricated explanation) when the loaded
        artifact's classifier step is not a LightGBM booster.
        """
        if self.explanation_capability != "available":
            return None
        artifact = self.require_loaded()
        feature_cols = artifact["feature_columns"]
        clf_pipe = artifact["classifier_pipeline"]
        pre = clf_pipe.named_steps.get("feature_preprocessor")
        model_step = clf_pipe.named_steps["fire_probability_model"]
        lgbm = getattr(model_step, "lgbm_model", model_step)

        matrix = pre.transform(frame[feature_cols]) if pre is not None else frame[feature_cols].to_numpy()
        contrib = lgbm.booster_.predict(matrix, pred_contrib=True)
        contrib = np.asarray(contrib)[:, :-1]  # drop the bias/expected-value column
        return {"feature_columns": feature_cols, "contributions": contrib}


_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(get_settings().model_artifact_path)
    return _registry


def model_updated_at() -> datetime:
    settings = get_settings()
    if settings.model_artifact_path and settings.model_artifact_path.is_file():
        return datetime.fromtimestamp(settings.model_artifact_path.stat().st_mtime, tz=timezone.utc)
    # Fall back to the champion feature contract's own last-modified time --
    # a real filesystem fact about this deployment's checked-in contract.
    from .config import ensure_pipeline_on_path, get_pipeline_config

    ensure_pipeline_on_path()
    from paths import resolve_path

    path = resolve_path(get_pipeline_config(), "contracts")
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
