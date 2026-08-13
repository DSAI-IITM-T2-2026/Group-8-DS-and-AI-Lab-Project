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
import io
import logging
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
    raw_probability: np.ndarray
    probability: np.ndarray
    rank_score: np.ndarray
    alert_score: np.ndarray
    priority_rank: np.ndarray
    alert_top_25: np.ndarray


logger = logging.getLogger("wildfire_api.model_registry")


def within_day_percentile(score: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Match the Milestone 5 notebook's within-day percentile calculation."""
    table = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "score": np.asarray(score, dtype=float),
            "position": np.arange(len(score)),
        }
    )
    table["percentile"] = table.groupby("date", sort=False)["score"].rank(
        method="average", pct=True
    )
    return table.sort_values("position")["percentile"].to_numpy(dtype=float)


class ModelRegistry:
    def __init__(self, artifact_source: str | Path | None) -> None:
        self.artifact_source = str(artifact_source) if artifact_source is not None else None
        self.artifact_path = (
            Path(self.artifact_source).expanduser()
            if self.artifact_source and not self.artifact_source.startswith("gs://")
            else None
        )
        self._artifact: dict[str, Any] | None = None
        self._error: str | None = None
        self._version_digest: str | None = None
        self._updated_at: datetime | None = None
        self._load()

    def _read_artifact(self) -> bytes:
        if self.artifact_source is None:
            raise FileNotFoundError("No model artifact source is configured.")
        if self.artifact_source.startswith("gs://"):
            rest = self.artifact_source[5:]
            bucket_name, separator, blob_name = rest.partition("/")
            if not separator or not bucket_name or not blob_name:
                raise ValueError("WILDFIRE_MODEL_URI must identify a GCS object.")
            from google.cloud import storage

            blob = storage.Client().bucket(bucket_name).blob(blob_name)
            blob.reload()
            self._updated_at = blob.updated
            return blob.download_as_bytes()
        assert self.artifact_path is not None
        payload = self.artifact_path.read_bytes()
        self._updated_at = datetime.fromtimestamp(
            self.artifact_path.stat().st_mtime, tz=timezone.utc
        )
        return payload

    def _load(self) -> None:
        if self.artifact_source is None:
            self._error = (
                "WILDFIRE_MODEL_URI is not configured. Point it at the trusted champion_model.joblib "
                "GCS object to enable inference endpoints."
            )
            return
        try:
            import joblib

            payload = self._read_artifact()
            artifact = joblib.load(io.BytesIO(payload))
        except Exception:  # pragma: no cover - depends on optional heavy deps
            logger.exception("Failed to load configured wildfire model artifact")
            self._error = (
                "The configured model artifact could not be loaded. Review the server log "
                "and ensure its custom model classes are importable."
            )
            return

        required = (
            "classifier_pipeline",
            "ranker_pipeline",
            "probability_calibrator",
            "classifier_weight",
            "ranker_weight",
            "feature_columns",
        )
        missing = [k for k in required if k not in artifact]
        if missing:
            self._error = f"Model artifact is missing required inference components: {missing}."
            return
        feature_columns = list(artifact["feature_columns"])
        if len(feature_columns) != 86 or len(set(feature_columns)) != 86:
            self._error = "Model artifact must contain exactly 86 unique feature columns."
            return
        try:
            weights = np.asarray(
                [artifact["classifier_weight"], artifact["ranker_weight"]], dtype=float
            )
        except (TypeError, ValueError):
            self._error = "Model artifact contains invalid classifier/ranker weights."
            return
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            self._error = "Model artifact contains invalid classifier/ranker weights."
            return
        self._artifact = artifact
        self._version_digest = hashlib.sha256(payload).hexdigest()[:12]

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
        if self._artifact is None:
            return f"{base} (unloaded)"
        return f"{base}-{self._version_digest}"

    @property
    def updated_at(self) -> datetime | None:
        return self._updated_at

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
        """Score a complete day exactly as Milestone 5 inference does."""
        artifact = self.require_loaded()
        feature_cols = list(artifact["feature_columns"])
        missing = [c for c in ("label_date", *feature_cols) if c not in frame.columns]
        if missing:
            raise ModelUnavailable(
                f"Feature snapshot is missing columns required by the model: {missing[:5]}"
            )
        if frame.empty:
            raise ModelUnavailable("Inference input is empty.")

        table = frame.reset_index(drop=True).copy()
        table["label_date"] = pd.to_datetime(table["label_date"]).dt.normalize()
        try:
            table[feature_cols] = table[feature_cols].apply(pd.to_numeric, errors="raise")
        except Exception as exc:
            raise ModelUnavailable("Model features must be numeric.") from exc
        x = table[feature_cols]

        raw_probability = np.asarray(
            artifact["classifier_pipeline"].predict_proba(x), dtype=float
        )[:, 1]
        clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        probability = np.asarray(
            artifact["probability_calibrator"].predict_proba(logits), dtype=float
        )[:, 1]

        sort_columns = ["label_date"] + (["cell_id"] if "cell_id" in table.columns else [])
        rank_input = table.assign(_position=np.arange(len(table))).sort_values(
            [*sort_columns, "_position"], kind="stable"
        )
        rank_sorted = np.asarray(
            artifact["ranker_pipeline"].predict(rank_input[feature_cols]), dtype=float
        )
        rank_score = np.empty(len(table), dtype=float)
        rank_score[rank_input["_position"].to_numpy(dtype=int)] = rank_sorted

        classifier_percentile = within_day_percentile(raw_probability, table["label_date"])
        ranker_percentile = within_day_percentile(rank_score, table["label_date"])
        alert_score = (
            float(artifact["classifier_weight"]) * classifier_percentile
            + float(artifact["ranker_weight"]) * ranker_percentile
        )
        priority_rank = (
            pd.Series(alert_score)
            .groupby(table["label_date"].to_numpy(), sort=False)
            .rank(method="first", ascending=False)
            .to_numpy(dtype=int)
        )
        return ScoredBatch(
            raw_probability=raw_probability,
            probability=probability,
            rank_score=rank_score,
            alert_score=alert_score,
            priority_rank=priority_rank,
            alert_top_25=priority_rank <= 25,
        )

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
        _registry = ModelRegistry(get_settings().model_artifact_source)
    return _registry


def model_updated_at() -> datetime:
    registry = get_model_registry()
    if registry.updated_at is not None:
        return registry.updated_at
    # Fall back to the champion feature contract's own last-modified time --
    # a real filesystem fact about this deployment's checked-in contract.
    from .config import ensure_pipeline_on_path, get_pipeline_config

    ensure_pipeline_on_path()
    from paths import resolve_path

    path = resolve_path(get_pipeline_config(), "contracts")
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
