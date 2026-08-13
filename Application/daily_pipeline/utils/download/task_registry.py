"""Persistent reconciliation for Earth Engine tasks started by the daily pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PENDING_STATES = frozenset({"READY", "RUNNING"})
SUCCESS_STATES = frozenset({"COMPLETED"})
FAILED_STATES = frozenset({"FAILED", "CANCELLED", "CANCEL_REQUESTED"})

_PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = _PIPELINE_ROOT / ".cache" / "ee_daily_tasks.json"


def _load() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(registry: dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REGISTRY_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(REGISTRY_PATH)


def registry_key(source: str, key: str) -> str:
    return f"{source}:{key}"


def remember_task(source: str, key: str, task_id: str, description: str) -> None:
    registry = _load()
    registry[registry_key(source, key)] = {
        "source": source,
        "key": key,
        "task_id": task_id,
        "description": description,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _save(registry)


def clear_task(source: str, key: str) -> None:
    registry = _load()
    if registry.pop(registry_key(source, key), None) is not None:
        _save(registry)


def _task_status(task_id: str) -> dict | None:
    import ee

    try:
        rows = ee.data.getTaskStatus(task_id)
    except Exception:
        return None
    if isinstance(rows, list):
        return rows[0] if rows else None
    return rows if isinstance(rows, dict) else None


def find_task(source: str, key: str, description: str) -> dict | None:
    """Find the best known task without treating terminal failures as reusable."""
    import ee

    saved = _load().get(registry_key(source, key))
    failed: dict | None = None
    if saved and saved.get("task_id"):
        status = _task_status(str(saved["task_id"]))
        if status:
            state = str(status.get("state", "")).upper()
            status["state"] = state
            if state in PENDING_STATES | SUCCESS_STATES:
                return status
            if state in FAILED_STATES:
                failed = status

    try:
        for task in ee.batch.Task.list(count=500):
            status = task.status()
            if status.get("description") != description:
                continue
            state = str(status.get("state", "")).upper()
            status["state"] = state
            task_id = str(status.get("id") or getattr(task, "id", ""))
            if state in PENDING_STATES | SUCCESS_STATES:
                if task_id:
                    remember_task(source, key, task_id, description)
                return status
            if state in FAILED_STATES and failed is None:
                failed = status
    except Exception:
        pass
    return failed


def failure_message(status: dict) -> str:
    state = str(status.get("state", "FAILED")).upper()
    detail = status.get("error_message") or status.get("error_details") or "No detail supplied."
    return f"Earth Engine task {state}: {detail}"
