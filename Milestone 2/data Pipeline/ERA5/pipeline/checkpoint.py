"""Checkpoint and resume support for interrupted downloads."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CheckpointManager:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                return json.load(f)
        return {"completed": [], "failed": {}, "updated_at": None}

    def save(self) -> None:
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(self._state, f, indent=2)

    def is_completed(self, key: str) -> bool:
        with self._lock:
            return key in self._state["completed"]

    def mark_completed(self, key: str, account: str | None = None) -> None:
        with self._lock:
            if key not in self._state["completed"]:
                self._state["completed"].append(key)
            self._state["failed"].pop(key, None)
            if account:
                assignments = self._state.setdefault("assignments", {})
                assignments[key] = account
            self.save()

    def mark_failed(self, key: str, error: str, account: str | None = None) -> None:
        with self._lock:
            self._state["failed"][key] = {
                "error": error,
                "account": account,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.save()

    def pending_keys(self, all_keys: list[str]) -> list[str]:
        with self._lock:
            return [key for key in all_keys if key not in self._state["completed"]]

    @property
    def completed_count(self) -> int:
        with self._lock:
            return len(self._state["completed"])

    @property
    def failed(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state["failed"])
