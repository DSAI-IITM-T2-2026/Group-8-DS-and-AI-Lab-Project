"""Structured progress events consumed by the application worker."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

EVENT_PREFIX = "PIPELINE_EVENT "


def emit_event(
    stage: str,
    status: str,
    message: str,
    *,
    completed: int | None = None,
    total: int | None = None,
    inventory: dict[str, Any] | None = None,
    artifact: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if completed is not None:
        payload["progressCompleted"] = completed
    if total is not None:
        payload["progressTotal"] = total
    if inventory is not None:
        payload["sourceInventory"] = inventory
    if artifact is not None:
        payload["artifact"] = artifact
    print(EVENT_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)
