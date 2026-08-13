"""SQLite state for feature export windows."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
RETRYING = "RETRYING"
CANCELLED = "CANCELLED"

ALL_STATUSES = (PENDING, RUNNING, COMPLETED, FAILED, RETRYING, CANCELLED)
UNFINISHED = (PENDING, RUNNING, RETRYING)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExportRecord:
    window_id: str
    start_date: str
    end_date: str
    task_id: str | None
    status: str
    attempt: int
    created_at: str
    updated_at: str
    error_message: str | None = None
    gcs_path: str | None = None
    window_index: int | None = None
    rows_exported: int | None = None


class StateDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            self._migrate_if_needed(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exports (
                    window_id TEXT PRIMARY KEY,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    task_id TEXT,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT,
                    gcs_path TEXT,
                    window_index INTEGER,
                    rows_exported INTEGER
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exports_status ON exports(status)"
            )

    def _migrate_if_needed(self, conn: sqlite3.Connection) -> None:
        """Collapse legacy (window_id, part_index) rows back to one row per window."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='exports'"
        ).fetchone()
        if row is None:
            return

        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(exports)").fetchall()
        }
        if "part_index" not in cols:
            return

        conn.execute(
            """
            CREATE TABLE exports_new (
                window_id TEXT PRIMARY KEY,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                task_id TEXT,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT,
                gcs_path TEXT,
                window_index INTEGER,
                rows_exported INTEGER
            )
            """
        )
        # Keep one row per window (prefer part_index = 0).
        conn.execute(
            """
            INSERT INTO exports_new (
                window_id, start_date, end_date, task_id, status,
                attempt, created_at, updated_at, error_message, gcs_path,
                window_index, rows_exported
            )
            SELECT
                e.window_id, e.start_date, e.end_date, e.task_id, e.status,
                e.attempt, e.created_at, e.updated_at, e.error_message, e.gcs_path,
                e.window_index, e.rows_exported
            FROM exports e
            INNER JOIN (
                SELECT window_id, MIN(part_index) AS part_index
                FROM exports
                GROUP BY window_id
            ) m
              ON e.window_id = m.window_id AND e.part_index = m.part_index
            """
        )
        conn.execute("DROP TABLE exports")
        conn.execute("ALTER TABLE exports_new RENAME TO exports")

    def upsert_window(
        self,
        window_id: str,
        start_date: str,
        end_date: str,
        window_index: int,
        gcs_path: str,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO exports (
                    window_id, start_date, end_date, task_id, status,
                    attempt, created_at, updated_at, gcs_path, window_index
                ) VALUES (?, ?, ?, NULL, ?, 0, ?, ?, ?, ?)
                ON CONFLICT(window_id) DO NOTHING
                """,
                (
                    window_id,
                    start_date,
                    end_date,
                    PENDING,
                    now,
                    now,
                    gcs_path,
                    window_index,
                ),
            )

    def get(self, window_id: str) -> ExportRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM exports WHERE window_id = ?", (window_id,)
            ).fetchone()
        return self._to_record(row) if row else None

    def list_by_status(self, *statuses: str) -> list[ExportRecord]:
        if not statuses:
            return []
        ph = ",".join("?" * len(statuses))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM exports
                WHERE status IN ({ph})
                ORDER BY start_date DESC, window_index DESC
                """,
                statuses,
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def next_submittable(self, window_id: str | None = None) -> ExportRecord | None:
        with self._connect() as conn:
            if window_id is None:
                row = conn.execute(
                    """
                    SELECT * FROM exports
                    WHERE status IN (?, ?)
                    ORDER BY start_date DESC, window_index DESC
                    LIMIT 1
                    """,
                    (PENDING, RETRYING),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT * FROM exports
                    WHERE status IN (?, ?) AND window_id = ?
                    ORDER BY start_date DESC, window_index DESC
                    LIMIT 1
                    """,
                    (PENDING, RETRYING, window_id),
                ).fetchone()
        return self._to_record(row) if row else None

    def count_by_status(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM exports GROUP BY status"
            ).fetchall()
        counts = {s: 0 for s in ALL_STATUSES}
        for row in rows:
            counts[row["status"]] = int(row["n"])
        return counts

    def total_windows(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM exports").fetchone()
        return int(row["n"]) if row else 0

    def unfinished_count(self, window_id: str | None = None) -> int:
        with self._connect() as conn:
            if window_id is None:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS n FROM exports
                    WHERE status IN ({",".join("?" * len(UNFINISHED))})
                    """,
                    UNFINISHED,
                ).fetchone()
            else:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS n FROM exports
                    WHERE window_id = ?
                      AND status IN ({",".join("?" * len(UNFINISHED))})
                    """,
                    (window_id, *UNFINISHED),
                ).fetchone()
        return int(row["n"]) if row else 0

    def reset_window_to_pending(self, window_id: str) -> None:
        """Force a window back to PENDING so it can be re-exported (test runs)."""
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE exports
                SET status = ?, task_id = NULL, attempt = 0,
                    error_message = NULL, rows_exported = NULL,
                    updated_at = ?
                WHERE window_id = ?
                """,
                (PENDING, now, window_id),
            )

    def mark_running(self, window_id: str, task_id: str) -> None:
        self._update(window_id, status=RUNNING, task_id=task_id, bump_attempt=True)

    def mark_completed(
        self, window_id: str, *, rows_exported: int | None = None
    ) -> None:
        self._update(
            window_id,
            status=COMPLETED,
            error_message=None,
            rows_exported=rows_exported,
        )

    def mark_failed(self, window_id: str, error_message: str) -> None:
        self._update(window_id, status=FAILED, error_message=error_message)

    def mark_cancelled(self, window_id: str, message: str = "Cancelled by user") -> None:
        self._update(window_id, status=CANCELLED, error_message=message)

    def mark_retrying(self, window_id: str, error_message: str) -> None:
        self._update(
            window_id,
            status=RETRYING,
            task_id=None,
            error_message=error_message,
        )

    def reset_failed_to_retrying(self) -> int:
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE exports
                SET status = ?, task_id = NULL, updated_at = ?,
                    error_message = COALESCE(error_message, 'manual retry')
                WHERE status = ?
                """,
                (RETRYING, now, FAILED),
            )
            return int(cur.rowcount)

    def _update(
        self,
        window_id: str,
        *,
        status: str | None = None,
        task_id: str | None | object = ...,
        error_message: str | None | object = ...,
        rows_exported: int | None | object = ...,
        bump_attempt: bool = False,
    ) -> None:
        fields: list[str] = ["updated_at = ?"]
        values: list[object] = [_utc_now()]
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if task_id is not ...:
            fields.append("task_id = ?")
            values.append(task_id)
        if error_message is not ...:
            fields.append("error_message = ?")
            values.append(error_message)
        if rows_exported is not ...:
            fields.append("rows_exported = ?")
            values.append(rows_exported)
        if bump_attempt:
            fields.append("attempt = attempt + 1")
        values.append(window_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE exports SET {', '.join(fields)} WHERE window_id = ?",
                values,
            )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ExportRecord:
        return ExportRecord(
            window_id=row["window_id"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            task_id=row["task_id"],
            status=row["status"],
            attempt=int(row["attempt"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error_message=row["error_message"],
            gcs_path=row["gcs_path"],
            window_index=row["window_index"],
            rows_exported=row["rows_exported"],
        )
