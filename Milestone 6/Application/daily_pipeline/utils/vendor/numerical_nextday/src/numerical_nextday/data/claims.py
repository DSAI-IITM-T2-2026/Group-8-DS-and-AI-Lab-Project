"""Append-only claim lock for multi-laptop shard coordination."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def claim_path(shared_cache: Path) -> Path:
    d = shared_cache / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d / "claim_lock.jsonl"


def _read_claims(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_done(shared_cache: Path, stage: str, year: int, month: int | None = None) -> bool:
    path = claim_path(shared_cache)
    key_month = month if month is not None else 0
    for row in _read_claims(path):
        if (
            row.get("stage") == stage
            and int(row.get("year", -1)) == year
            and int(row.get("month", 0)) == key_month
            and row.get("status") == "done"
        ):
            return True
    return False


def claim(
    shared_cache: Path,
    *,
    worker: str,
    stage: str,
    year: int,
    month: int | None = None,
    status: str = "running",
    force: bool = False,
) -> bool:
    """
    Append a claim. Returns False if already done and not force
    (caller should skip). Returns True if work should proceed.
    """
    path = claim_path(shared_cache)
    key_month = month if month is not None else 0
    if status == "running" and not force and is_done(shared_cache, stage, year, month):
        logger.info("Skip %s year=%s month=%s (already done)", stage, year, month)
        return False

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "worker": worker,
        "stage": stage,
        "year": year,
        "month": key_month,
        "status": status,
    }
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return True


def mark_done(
    shared_cache: Path,
    *,
    worker: str,
    stage: str,
    year: int,
    month: int | None = None,
) -> None:
    claim(shared_cache, worker=worker, stage=stage, year=year, month=month, status="done", force=True)
