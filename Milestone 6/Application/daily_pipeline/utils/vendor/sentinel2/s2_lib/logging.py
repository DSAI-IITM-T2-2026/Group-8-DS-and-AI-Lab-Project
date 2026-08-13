"""Structured JSON logging for the feature pipeline."""

from __future__ import annotations

import json
import logging as _logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredFormatter(_logging.Formatter):
    def format(self, record: _logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            "window": getattr(record, "window", None),
            "task_id": getattr(record, "task_id", None),
            "status": getattr(record, "status", None),
            "elapsed_time": getattr(record, "elapsed_time", None),
            "rows_exported": getattr(record, "rows_exported", None),
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_dir: Path, name: str = "s2_features") -> _logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = _logging.getLogger(name)
    logger.setLevel(_logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = StructuredFormatter()
    fh = _logging.FileHandler(log_dir / "pipeline.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = _logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def log_event(
    logger: _logging.Logger,
    event: str,
    *,
    window: str | None = None,
    task_id: str | None = None,
    status: str | None = None,
    elapsed_time: float | None = None,
    rows_exported: int | None = None,
    level: int = _logging.INFO,
    **extra: Any,
) -> None:
    parts = [event]
    if extra:
        parts.append(" ".join(f"{k}={v}" for k, v in extra.items()))
    logger.log(
        level,
        " | ".join(parts),
        extra={
            "event": event,
            "window": window,
            "task_id": task_id,
            "status": status,
            "elapsed_time": elapsed_time,
            "rows_exported": rows_exported,
        },
    )
