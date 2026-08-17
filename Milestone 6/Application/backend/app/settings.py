from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

import yaml


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
    except ImportError:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    application_root: Path
    pipeline_root: Path
    state_dir: Path
    python_executable: str
    cors_origins: tuple[str, ...]
    timezone: str
    lookback_days: int
    cutoff_local_time: str = "06:30"
    expected_feature_count: int = 86
    min_prediction_date: str = "2019-01-01"

    @property
    def database_path(self) -> Path:
        return self.state_dir / "pipeline-runs.sqlite3"

    @property
    def logs_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def max_prediction_date(self) -> str:
        return (datetime.now(ZoneInfo(self.timezone)).date() + timedelta(days=1)).isoformat()


def load_settings() -> Settings:
    backend_root = Path(__file__).resolve().parents[1]
    _load_dotenv(backend_root / ".env")
    application_root = backend_root.parent
    pipeline_root = Path(
        os.environ.get(
            "PIPELINE_ROOT",
            application_root / "daily_pipeline",
        )
    ).resolve()
    config_path = pipeline_root / "utils" / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    state_dir = Path(
        os.environ.get("PIPELINE_STATE_DIR", backend_root / ".state")
    ).resolve()
    cors = tuple(
        value.strip()
        for value in os.environ.get(
            "PIPELINE_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if value.strip()
    )
    configured_python = os.environ.get("PIPELINE_PYTHON")
    pipeline_python = pipeline_root / ".venv" / "bin" / "python"
    application_python = application_root / ".venv" / "bin" / "python"
    python_executable = configured_python or str(
        pipeline_python if pipeline_python.is_file() else application_python if application_python.is_file() else "python3"
    )
    return Settings(
        application_root=application_root,
        pipeline_root=pipeline_root,
        state_dir=state_dir,
        python_executable=python_executable,
        cors_origins=cors,
        timezone=str(raw["task"].get("timezone", "America/Los_Angeles")),
        cutoff_local_time=str(raw["task"].get("forecast_cutoff_local_time", "06:30")),
        lookback_days=int(raw["task"].get("lookback_days", 30)),
    )
