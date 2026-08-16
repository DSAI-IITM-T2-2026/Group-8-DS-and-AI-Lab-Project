from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4


ACTIVE = ("queued", "running", "waiting_external")


class RunStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY, prediction_date TEXT NOT NULL, status TEXT NOT NULL,
                stage TEXT NOT NULL, message TEXT NOT NULL, progress_completed INTEGER NOT NULL DEFAULT 0,
                progress_total INTEGER NOT NULL DEFAULT 0, inventory_json TEXT NOT NULL DEFAULT '{}',
                artifact_json TEXT, error_code TEXT, created_at TEXT NOT NULL,
                started_at TEXT, finished_at TEXT, pid INTEGER)"""
            )

    def interrupt_orphans(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                "UPDATE pipeline_runs SET status='interrupted', message=?, error_code='worker_restarted', finished_at=? "
                "WHERE status IN ('running','waiting_external')",
                ("The API restarted while this run was active. Retry to reconcile completed cloud work.", now),
            )

    def create_or_reuse(self, prediction_date: str) -> tuple[dict, bool]:
        with self._lock, self._connect() as db:
            placeholders = ",".join("?" for _ in ACTIVE)
            row = db.execute(
                f"SELECT * FROM pipeline_runs WHERE prediction_date=? AND status IN ({placeholders}) ORDER BY created_at DESC LIMIT 1",
                (prediction_date, *ACTIVE),
            ).fetchone()
            if row:
                return self._row(row), True
            now = datetime.now(timezone.utc).isoformat()
            run_id = str(uuid4())
            db.execute(
                "INSERT INTO pipeline_runs (run_id,prediction_date,status,stage,message,created_at) VALUES (?,?,?,?,?,?)",
                (run_id, prediction_date, "queued", "validating", "Queued for preparation.", now),
            )
            row = db.execute("SELECT * FROM pipeline_runs WHERE run_id=?", (run_id,)).fetchone()
            return self._row(row), False

    def get(self, run_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM pipeline_runs WHERE run_id=?", (run_id,)).fetchone()
            return self._row(row) if row else None

    def list(self, prediction_date: str | None = None, limit: int = 20) -> list[dict]:
        with self._connect() as db:
            if prediction_date:
                rows = db.execute(
                    "SELECT * FROM pipeline_runs WHERE prediction_date=? ORDER BY created_at DESC LIMIT ?",
                    (prediction_date, limit),
                ).fetchall()
            else:
                rows = db.execute("SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [self._row(row) for row in rows]

    def claim_next(self) -> dict | None:
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM pipeline_runs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                db.commit()
                return None
            now = datetime.now(timezone.utc).isoformat()
            db.execute(
                "UPDATE pipeline_runs SET status='running', stage='validating', message=?, started_at=? WHERE run_id=?",
                ("Starting the preparation pipeline.", now, row["run_id"]),
            )
            db.commit()
            return self.get(str(row["run_id"]))

    def cancel(self, run_id: str) -> dict | None:
        """Atomically interrupt an active run so it cannot be claimed or updated."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM pipeline_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                db.commit()
                return None
            if row["status"] in ACTIVE:
                placeholders = ",".join("?" for _ in ACTIVE)
                db.execute(
                    f"UPDATE pipeline_runs SET status='interrupted', message=?, error_code=?, "
                    f"finished_at=?, pid=NULL WHERE run_id=? AND status IN ({placeholders})",
                    (
                        "Forecast preparation was stopped by the user.",
                        "cancelled_by_user",
                        now,
                        run_id,
                        *ACTIVE,
                    ),
                )
            db.commit()
            updated = db.execute("SELECT * FROM pipeline_runs WHERE run_id=?", (run_id,)).fetchone()
            return self._row(updated)

    def update(self, run_id: str, *, only_if_active: bool = False, **values) -> None:
        mapping = {
            "status": "status", "stage": "stage", "message": "message",
            "progressCompleted": "progress_completed", "progressTotal": "progress_total",
            "errorCode": "error_code", "pid": "pid",
        }
        columns: list[str] = []
        params: list[object] = []
        for key, value in values.items():
            if key == "sourceInventory":
                columns.append("inventory_json=?"); params.append(json.dumps(value))
            elif key == "artifact":
                columns.append("artifact_json=?"); params.append(json.dumps(value))
            elif key in mapping:
                columns.append(f"{mapping[key]}=?"); params.append(value)
        if values.get("status") in {"succeeded", "unavailable", "failed", "interrupted"}:
            columns.append("finished_at=?"); params.append(datetime.now(timezone.utc).isoformat())
        if not columns:
            return
        params.append(run_id)
        where = "run_id=?"
        if only_if_active:
            placeholders = ",".join("?" for _ in ACTIVE)
            where += f" AND status IN ({placeholders})"
            params.extend(ACTIVE)
        with self._connect() as db:
            db.execute(f"UPDATE pipeline_runs SET {','.join(columns)} WHERE {where}", params)

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        return {
            "runId": row["run_id"], "predictionDate": row["prediction_date"],
            "status": row["status"], "stage": row["stage"], "message": row["message"],
            "progressCompleted": row["progress_completed"], "progressTotal": row["progress_total"],
            "sourceInventory": json.loads(row["inventory_json"] or "{}"),
            "artifact": json.loads(row["artifact_json"]) if row["artifact_json"] else None,
            "errorCode": row["error_code"], "createdAt": row["created_at"],
            "startedAt": row["started_at"], "finishedAt": row["finished_at"],
        }
