"""Resumable scheduler for Sentinel-2 feature exports."""

from __future__ import annotations

import time
from datetime import date
from logging import Logger

from . import export as export_mod
from .config import AppConfig, resolve_path
from .logging import log_event
from .sentinel2 import (
    TimeWindow,
    find_window_for_date,
    generate_windows,
    windows_for_year,
)
from .state import (
    CANCELLED,
    COMPLETED,
    FAILED,
    PENDING,
    RETRYING,
    RUNNING,
    StateDB,
)


class FeatureScheduler:
    def __init__(
        self,
        cfg: AppConfig,
        logger: Logger,
        db: StateDB | None = None,
        exporter=None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.db = db or StateDB(resolve_path(cfg, cfg.scheduler.state_db))
        # Source-specific exporters share a small interface. Keeping the
        # Sentinel-2 exporter as the default preserves the existing CLI while
        # allowing Sentinel-5P to reuse the proven resumable scheduler.
        self.exporter = exporter or export_mod
        self._only_window_id: str | None = None

    def seed_windows(self, windows: list[TimeWindow] | None = None) -> int:
        if windows is None:
            windows = generate_windows(
                self.cfg.temporal.start_year,
                self.cfg.temporal.end_year,
                self.cfg.temporal.window_days,
            )
        for w in windows:
            self.db.upsert_window(
                window_id=w.window_id,
                start_date=w.start_str,
                end_date=w.end_str,
                window_index=w.window_index,
                gcs_path=self.exporter.gcs_final_uri(self.cfg, w),
            )
        log_event(
            self.logger,
            "Windows Seeded",
            status=PENDING,
            total=len(windows),
            db=str(self.db.db_path),
        )
        return len(windows)

    def run(
        self,
        *,
        once: bool = False,
        only_date: date | None = None,
        only_year: int | None = None,
        only_month: tuple[int, int] | None = None,
        force: bool = False,
        submit_all: bool = False,
    ) -> None:
        self.exporter.initialize(self.cfg.project_id)

        if sum(value is not None for value in (only_date, only_year, only_month)) > 1:
            raise ValueError("Use only one of only_date / only_year / only_month")

        if only_date is not None:
            window = find_window_for_date(
                only_date,
                self.cfg.temporal.start_year,
                self.cfg.temporal.end_year,
                self.cfg.temporal.window_days,
            )
            if window is None:
                raise ValueError(
                    f"No temporal window contains {only_date.isoformat()} "
                    f"in {self.cfg.temporal.start_year}–{self.cfg.temporal.end_year}"
                )
            self._only_window_id = window.window_id
            self.seed_windows([window])
            if force:
                self.db.reset_window_to_pending(window.window_id)
            uri = self.exporter.gcs_final_uri(self.cfg, window)
            log_event(
                self.logger,
                "Single-Day Test Mode",
                window=window.window_id,
                status=PENDING,
                gcs=uri,
            )
            print(
                f"Single-day mode: {only_date.isoformat()} → window "
                f"{window.window_id} (index {window.window_index:03d})"
            )
            print(f"Export URI: {uri}")
        elif only_month is not None:
            year, month = only_month
            if not 1 <= month <= 12:
                raise ValueError(f"Invalid month: {month}")
            windows = [
                window
                for window in generate_windows(
                    self.cfg.temporal.start_year,
                    self.cfg.temporal.end_year,
                    self.cfg.temporal.window_days,
                )
                if window.start_date.year == year
                and window.start_date.month == month
            ]
            windows.reverse()
            if not windows:
                raise ValueError(
                    f"No windows for {year:04d}-{month:02d} in "
                    f"{self.cfg.temporal.start_year}-{self.cfg.temporal.end_year}"
                )
            self._only_window_id = None
            self.seed_windows(windows)
            if force:
                for window in windows:
                    self.db.reset_window_to_pending(window.window_id)
            log_event(
                self.logger,
                "Month Run Mode",
                status=PENDING,
                month=f"{year:04d}-{month:02d}",
                total=len(windows),
                db=str(self.db.db_path),
            )
            print(
                f"Month mode: {year:04d}-{month:02d} -> {len(windows)} windows "
                f"({windows[0].window_id} ... {windows[-1].window_id})"
            )
            print(f"State DB: {self.db.db_path}")
            print(
                f"Export root: gs://{self.cfg.export.bucket}/"
                f"{self.cfg.export.prefix}/year={year:04d}/month={month:02d}/"
            )
        elif only_year is not None:
            windows = windows_for_year(
                only_year,
                self.cfg.temporal.start_year,
                self.cfg.temporal.end_year,
                self.cfg.temporal.window_days,
            )
            if not windows:
                raise ValueError(
                    f"No windows for year {only_year} in "
                    f"{self.cfg.temporal.start_year}–{self.cfg.temporal.end_year}"
                )
            self._only_window_id = None
            self.seed_windows(windows)
            if force:
                for w in windows:
                    self.db.reset_window_to_pending(w.window_id)
            log_event(
                self.logger,
                "Year Run Mode",
                status=PENDING,
                year=only_year,
                total=len(windows),
                db=str(self.db.db_path),
            )
            print(
                f"Year mode: {only_year} → {len(windows)} windows "
                f"({windows[0].window_id} … {windows[-1].window_id})"
            )
            print(f"State DB: {self.db.db_path}")
            print(
                f"Export root: gs://{self.cfg.export.bucket}/"
                f"{self.cfg.export.prefix}/year={only_year}/"
            )
        else:
            self._only_window_id = None
            self.seed_windows()

        self._reconcile_on_startup()

        if submit_all:
            submitted = self._submit_all_pending()
            log_event(
                self.logger,
                "All Tasks Submitted",
                status=RUNNING,
                submitted=submitted,
                unfinished=self.db.unfinished_count(self._only_window_id),
            )
            print(
                f"Submitted {submitted} Earth Engine task(s). "
                "They keep running on Google after you shut the laptop."
            )
            print(
                "CSVs land in GCS when EE finishes. "
                "Later, run the same command without --submit-all "
                "to convert CSV→Parquet and pick up failures."
            )
            return

        log_event(
            self.logger,
            "Scheduler Started",
            max_running=self.cfg.scheduler.max_running_tasks,
            poll_interval=self.cfg.scheduler.poll_interval_seconds,
            unfinished=self.db.unfinished_count(self._only_window_id),
            window=self._only_window_id,
        )

        while True:
            unfinished = self.db.unfinished_count(self._only_window_id)
            if unfinished == 0:
                log_event(
                    self.logger,
                    "All Exports Completed",
                    status=COMPLETED,
                    window=self._only_window_id,
                )
                if self._only_window_id:
                    rec = self.db.get(self._only_window_id)
                    print(
                        f"Finished {self._only_window_id}: "
                        f"status={rec.status if rec else 'missing'} "
                        f"gcs={rec.gcs_path if rec else 'n/a'}"
                    )
                break

            self._refresh_active_tasks()
            submitted = self._fill_capacity()

            if once:
                log_event(
                    self.logger,
                    "Single Cycle Complete",
                    submitted=submitted,
                    unfinished=self.db.unfinished_count(self._only_window_id),
                )
                break

            if submitted == 0:
                log_event(
                    self.logger,
                    "Waiting For Capacity",
                    status=RUNNING,
                    unfinished=self.db.unfinished_count(self._only_window_id),
                    poll_interval=self.cfg.scheduler.poll_interval_seconds,
                )
                time.sleep(self.cfg.scheduler.poll_interval_seconds)
            else:
                time.sleep(min(15, self.cfg.scheduler.poll_interval_seconds))

    def retry_failed(self) -> int:
        n = self.db.reset_failed_to_retrying()
        log_event(self.logger, "Retry", status=RETRYING, reset=n)
        return n

    def _reconcile_on_startup(self) -> None:
        for record in self.db.list_by_status(RUNNING):
            if self._only_window_id and record.window_id != self._only_window_id:
                continue
            if not record.task_id:
                self._handle_failure(
                    record.window_id,
                    record.attempt,
                    None,
                    "Missing task id on resume",
                )
                continue
            status = self.exporter.get_task_status(record.task_id)
            if status is None:
                self._handle_failure(
                    record.window_id,
                    record.attempt,
                    record.task_id,
                    "Task not found in Earth Engine",
                )
                continue
            self._apply_ee_state(record, status)

    def _refresh_active_tasks(self) -> None:
        for record in self.db.list_by_status(RUNNING):
            if self._only_window_id and record.window_id != self._only_window_id:
                continue
            if not record.task_id:
                continue
            status = self.exporter.get_task_status(record.task_id)
            if status is None:
                continue
            self._apply_ee_state(record, status)

    def _apply_ee_state(self, record, status: dict) -> None:
        state = str(status.get("state", "")).upper()
        task_id = record.task_id
        window_id = record.window_id

        if state == self.exporter.EE_COMPLETED:
            try:
                rows = self._post_process(record)
            except Exception as exc:  # noqa: BLE001
                self._handle_failure(
                    window_id, record.attempt, task_id, str(exc)
                )
                return
            self.db.mark_completed(window_id, rows_exported=rows)
            log_event(
                self.logger,
                "Task Completed",
                window=window_id,
                task_id=task_id,
                status=COMPLETED,
                rows_exported=rows if rows is not None and rows >= 0 else None,
            )
            return

        if state == self.exporter.EE_CANCELLED:
            err = status.get("error_message") or status.get("errorMessage") or state
            self.db.mark_cancelled(window_id, str(err))
            log_event(
                self.logger,
                "Task Cancelled",
                window=window_id,
                task_id=task_id,
                status=CANCELLED,
                error=str(err),
            )
            return

        if state == self.exporter.EE_FAILED:
            err = status.get("error_message") or status.get("errorMessage") or state
            self._handle_failure(window_id, record.attempt, task_id, str(err))
            return

    def _post_process(self, record) -> int | None:
        if self.cfg.export.format != "parquet":
            return None
        window = TimeWindow(
            window_id=record.window_id,
            start_date=date.fromisoformat(record.start_date),
            end_date=date.fromisoformat(record.end_date),
            window_index=record.window_index or 0,
        )
        try:
            rows = self.exporter.convert_csv_to_parquet(self.cfg, window)
            return rows
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Parquet conversion failed: {exc}") from exc

    def _handle_failure(
        self,
        window_id: str,
        attempt: int,
        task_id: str | None,
        error_message: str,
    ) -> None:
        retry_limit = self.cfg.scheduler.retry_attempts
        transient = self.exporter.is_transient_error(error_message)
        if transient and attempt < retry_limit:
            self.db.mark_retrying(window_id, error_message)
            log_event(
                self.logger,
                "Retry",
                window=window_id,
                task_id=task_id,
                status=RETRYING,
                attempt=attempt,
                error=error_message,
            )
        else:
            self.db.mark_failed(window_id, error_message)
            log_event(
                self.logger,
                "Task Failed",
                window=window_id,
                task_id=task_id,
                status=FAILED,
                attempt=attempt,
                error=error_message,
            )

    def _submit_all_pending(self) -> int:
        """Queue every PENDING/RETRYING window on Earth Engine, then return."""
        submitted = 0
        while True:
            nxt = self.db.next_submittable(self._only_window_id)
            if nxt is None:
                break

            window = TimeWindow(
                window_id=nxt.window_id,
                start_date=date.fromisoformat(nxt.start_date),
                end_date=date.fromisoformat(nxt.end_date),
                window_index=nxt.window_index or 0,
            )
            try:
                log_event(
                    self.logger,
                    "Task Created",
                    window=window.window_id,
                    status=PENDING,
                    attempt=nxt.attempt + 1,
                )
                t0 = time.time()
                task_id, uri = self.exporter.start_export(self.cfg, window)
                self.db.mark_running(window.window_id, task_id)
                submitted += 1
                log_event(
                    self.logger,
                    "Task Started",
                    window=window.window_id,
                    task_id=task_id,
                    status=RUNNING,
                    elapsed_time=round(time.time() - t0, 3),
                    gcs=uri,
                )
                print(f"  queued {submitted}: {window.window_id} → {task_id}")
            except Exception as exc:  # noqa: BLE001
                self._handle_failure(window.window_id, nxt.attempt, None, str(exc))
                if self.exporter.is_transient_error(str(exc)):
                    print(f"  stopped after transient error: {exc}")
                    break
        return submitted

    def _fill_capacity(self) -> int:
        max_running = self.cfg.scheduler.max_running_tasks
        submitted = 0

        while True:
            running = self.db.list_by_status(RUNNING)
            if self._only_window_id:
                local_inflight = sum(
                    1 for r in running if r.window_id == self._only_window_id
                )
            else:
                local_inflight = len(running)
            ee_active = self.exporter.count_active_tasks()
            if max(local_inflight, ee_active) >= max_running:
                break

            nxt = self.db.next_submittable(self._only_window_id)
            if nxt is None:
                break

            window = TimeWindow(
                window_id=nxt.window_id,
                start_date=date.fromisoformat(nxt.start_date),
                end_date=date.fromisoformat(nxt.end_date),
                window_index=nxt.window_index or 0,
            )

            try:
                log_event(
                    self.logger,
                    "Task Created",
                    window=window.window_id,
                    status=PENDING,
                    attempt=nxt.attempt + 1,
                )
                t0 = time.time()
                task_id, uri = self.exporter.start_export(self.cfg, window)
                self.db.mark_running(window.window_id, task_id)
                submitted += 1
                log_event(
                    self.logger,
                    "Task Started",
                    window=window.window_id,
                    task_id=task_id,
                    status=RUNNING,
                    elapsed_time=round(time.time() - t0, 3),
                    gcs=uri,
                )
            except Exception as exc:  # noqa: BLE001
                self._handle_failure(window.window_id, nxt.attempt, None, str(exc))
                if self.exporter.is_transient_error(str(exc)):
                    break

        return submitted


def estimate_completion(
    counts: dict[str, int],
    poll_interval_seconds: int,
    max_running_tasks: int,
    *,
    avg_task_seconds: float = 2400.0,
) -> str:
    remaining = (
        counts.get(PENDING, 0)
        + counts.get(RETRYING, 0)
        + counts.get(RUNNING, 0)
    )
    if remaining <= 0:
        return "done"
    concurrency = max(1, max_running_tasks)
    per_task = max(avg_task_seconds, float(poll_interval_seconds))
    seconds = (remaining / concurrency) * per_task
    hours = seconds / 3600.0
    if hours < 1:
        return f"~{int(seconds // 60)} min"
    if hours < 48:
        return f"~{hours:.1f} hours"
    return f"~{hours / 24:.1f} days"
