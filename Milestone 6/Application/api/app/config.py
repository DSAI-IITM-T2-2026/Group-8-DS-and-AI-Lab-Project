"""Settings for the Wildfire Prediction Studio API.

Everything here maps to a real, documented piece of the project:

- ``daily_pipeline`` config (bucket, grid, feature contract) is loaded from
  ``Milestone 6/Application/daily_pipeline/utils/config.yaml`` via the pipeline's own
  ``config_loader`` -- we do not duplicate those values.
- Deployment-specific knobs (which trained model artifact to serve, whether
  to allow GCS reads, CORS origin) come from environment variables / ``.env``
  so this package stays runnable without secrets for the endpoints that do
  not need them.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def api_root() -> Path:
    """Milestone 6/Application/api/"""
    return Path(__file__).resolve().parent.parent


def application_root() -> Path:
    return api_root().parent


def daily_pipeline_root() -> Path:
    return application_root() / "daily_pipeline"


def _load_dotenv() -> None:
    for env_path in (api_root() / ".env", daily_pipeline_root() / ".env"):
        if not env_path.is_file():
            continue
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
        except ImportError:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def ensure_pipeline_on_path() -> Path:
    """Add daily_pipeline/utils (+ vendored M4) to sys.path, mirroring bootstrap.py.

    This lets the API reuse the pipeline's own config loader, feature
    contract, and cell-subset logic instead of re-deriving them by hand.
    """
    _load_dotenv()
    uroot = daily_pipeline_root() / "utils"
    m4_src = uroot / "vendor" / "numerical_nextday" / "src"
    for path in (uroot, m4_src):
        s = str(path)
        if path.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    return daily_pipeline_root()


class Settings:
    """Deployment settings, read once and cached."""

    def __init__(self) -> None:
        _load_dotenv()

        # Trained model artifact (wildfire_model.joblib from Milestone 5's
        # training notebook). Not checked into the repo -- point this at a
        # real artifact to enable the endpoints that need inference.
        # Prefer a deployment-safe GCS registry URI. A local path remains
        # supported for offline development and tests.
        artifact = (
            os.environ.get("WILDFIRE_MODEL_URI", "").strip()
            or os.environ.get("WILDFIRE_MODEL_ARTIFACT", "").strip()
        )
        self.model_artifact_source: str | None = artifact or None
        self.model_artifact_path: Path | None = (
            Path(artifact).expanduser() if artifact and not artifact.startswith("gs://") else None
        )

        # Optional local copy of the multi-year archive
        # (final_processed/2019_2025/2019-2025.parquet) used only for
        # /validation/events. See daily_pipeline/README.md "Zero-download
        # 2025 replay" for how to fetch it.
        archive = os.environ.get("WILDFIRE_HISTORICAL_ARCHIVE", "").strip()
        self.historical_archive_path: Path | None = Path(archive).expanduser() if archive else None

        # Whether the API is allowed to reach GCS for final_processed/*.parquet
        # that are not already in the pipeline's local .cache. Defaults on
        # because the daily_pipeline is designed to run with GCS access, but
        # can be disabled for fully offline dev.
        self.allow_gcs_reads: bool = os.environ.get("WILDFIRE_ALLOW_GCS", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }

        self.frontend_origins: list[str] = [
            origin.strip()
            for origin in os.environ.get("WILDFIRE_FRONTEND_ORIGINS", "*").split(",")
            if origin.strip()
        ]

        self.api_title = "Wildfire Prediction Studio API"
        self.api_base_path = "/api/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_pipeline_config() -> dict[str, Any]:
    """Load daily_pipeline/utils/config.yaml through the pipeline's own loader."""
    ensure_pipeline_on_path()
    from config_loader import load_daily_config  # type: ignore

    return load_daily_config()


@lru_cache(maxsize=1)
def get_feature_contract() -> dict[str, Any]:
    """Load the frozen 86-feature champion contract through the pipeline's own loader."""
    ensure_pipeline_on_path()
    from config_loader import load_feature_contract  # type: ignore

    return load_feature_contract(get_pipeline_config())
