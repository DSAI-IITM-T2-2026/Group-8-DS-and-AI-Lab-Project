from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
from contextlib import contextmanager

from .settings import Settings
from .store import RunStore

EVENT_PREFIX = "PIPELINE_EVENT "


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


@contextmanager
def pipeline_lock(pipeline_root):
    """Share the cron PID lock; fail closed when another live process owns it."""
    pidfile = pipeline_root / ".cache" / "daily_pipeline.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    if pidfile.exists():
        try:
            owner = int(pidfile.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            owner = -1
        if owner > 0 and _pid_is_running(owner):
            raise RuntimeError("pipeline_already_running")
        pidfile.unlink(missing_ok=True)
    descriptor = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        try:
            if pidfile.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pidfile.unlink(missing_ok=True)
        except OSError:
            pass


def safe_error_code(output: str) -> str:
    lowered = output.lower()
    if "feature contract mismatch" in lowered:
        return "feature_contract_mismatch"
    if re.search(r"no rows for label_date=\d{4}-\d{2}-\d{2} in history panel", lowered):
        return "prediction_day_row_missing"
    if "cds credentials not found" in lowered or ("cds_api_key" in lowered and "not found" in lowered):
        return "cds_authentication_failed"
    if "earth engine init failed" in lowered or "default credentials" in lowered:
        return "cloud_authentication_failed"
    if "cancelled" in lowered or "failed" in lowered and "earth engine" in lowered:
        return "earth_engine_task_failed"
    if "timed out" in lowered:
        return "external_data_timeout"
    if "quota" in lowered or "429" in lowered:
        return "cloud_quota_exceeded"
    return "pipeline_failed"


def safe_error_message(output: str) -> str:
    """Return an actionable failure message without exposing raw log output."""
    missing_day = re.search(
        r"No rows for label_date=(\d{4}-\d{2}-\d{2}) in history panel",
        output,
        flags=re.IGNORECASE,
    )
    if missing_day:
        return (
            "Feature preparation did not produce the required prediction-day row for "
            f"{missing_day.group(1)}. The Stage C date window must include the prediction "
            "day before parquet export."
        )
    messages = {
        "feature_contract_mismatch": "The deployed model feature contract is stale or invalid. Rebuild the application before retrying.",
        "cds_authentication_failed": "ERA5 access failed because CDS credentials are unavailable.",
        "cloud_authentication_failed": "Cloud access failed because application credentials are unavailable.",
        "earth_engine_task_failed": "An Earth Engine data-preparation task failed.",
        "external_data_timeout": "A required external data task did not finish before the timeout.",
        "cloud_quota_exceeded": "A cloud data provider rejected the request because its quota was exceeded.",
    }
    return messages.get(
        safe_error_code(output),
        "The preparation pipeline failed. Review the server log for the underlying error.",
    )


class PipelineWorker:
    def __init__(self, settings: Settings, store: RunStore):
        self.settings = settings
        self.store = store
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process_lock = threading.RLock()
        self._active_run_id: str | None = None
        self._active_process: subprocess.Popen[str] | None = None

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.alive:
            return
        self.settings.logs_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="pipeline-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    @staticmethod
    def _signal_process(process: subprocess.Popen[str], sig: signal.Signals) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return

    def _force_kill_after_grace(self, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._signal_process(process, signal.SIGKILL)

    def cancel(self, run_id: str) -> dict | None:
        """Persist cancellation, then stop the matching local process group."""
        run = self.store.cancel(run_id)
        if run is None or run.get("errorCode") != "cancelled_by_user":
            return run
        with self._process_lock:
            process = self._active_process if self._active_run_id == run_id else None
        if process is not None and process.poll() is None:
            self._signal_process(process, signal.SIGTERM)
            threading.Thread(
                target=self._force_kill_after_grace,
                args=(process,),
                name=f"pipeline-cancel-{run_id}",
                daemon=True,
            ).start()
        return self.store.get(run_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            run = self.store.claim_next()
            if run is None:
                self._stop.wait(1)
                continue
            self._execute(run)

    def _execute(self, run: dict) -> None:
        run_id = run["runId"]
        label_date = run["predictionDate"]
        log_path = self.settings.logs_dir / f"{run_id}.log"
        command = [
            self.settings.python_executable,
            "run_daily.py", "all", "--label-date", label_date,
        ]
        env = os.environ.copy()
        lines: list[str] = []
        process: subprocess.Popen[str] | None = None
        try:
            with pipeline_lock(self.settings.pipeline_root):
                current = self.store.get(run_id) or {}
                if current.get("status") == "interrupted":
                    return
                with log_path.open("a", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command, cwd=self.settings.pipeline_root, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, start_new_session=True,
                    )
                    with self._process_lock:
                        self._active_run_id = run_id
                        self._active_process = process
                    self.store.update(run_id, only_if_active=True, pid=process.pid)
                    current = self.store.get(run_id) or {}
                    if current.get("status") == "interrupted":
                        self._signal_process(process, signal.SIGTERM)
                    assert process.stdout is not None
                    for raw in process.stdout:
                        log.write(raw); log.flush()
                        line = raw.rstrip()
                        lines.append(line)
                        if line.startswith(EVENT_PREFIX):
                            try:
                                event = json.loads(line[len(EVENT_PREFIX):])
                            except json.JSONDecodeError:
                                continue
                            updates = {
                                "status": event.get("status", "running"),
                                "stage": event.get("stage", "validating"),
                                "message": event.get("message", "Pipeline is running."),
                                "progressCompleted": event.get("progressCompleted", 0),
                                "progressTotal": event.get("progressTotal", 0),
                            }
                            if "sourceInventory" in event:
                                updates["sourceInventory"] = event["sourceInventory"]
                            if "artifact" in event:
                                updates["artifact"] = event["artifact"]
                            self.store.update(run_id, only_if_active=True, **updates)
                    code = process.wait()
            current = self.store.get(run_id) or {}
            if current.get("status") == "interrupted":
                return
            if code == 0:
                if current.get("status") == "unavailable":
                    pass
                elif current.get("artifact"):
                    self.store.update(run_id, status="succeeded", stage="completed", message="Prediction data is ready.")
                else:
                    self.store.update(run_id, status="failed", stage="completed", message="The pipeline ended without a verified artifact.", errorCode="artifact_missing")
            else:
                output = "\n".join(lines[-80:])
                self.store.update(
                    run_id,
                    status="failed",
                    message=safe_error_message(output),
                    errorCode=safe_error_code(output),
                )
        except RuntimeError as exc:
            current = self.store.get(run_id) or {}
            if current.get("status") == "interrupted":
                return
            if str(exc) == "pipeline_already_running":
                self.store.update(run_id, status="failed", message="Another scheduled pipeline run is already using the shared cache. Retry after it finishes.", errorCode="pipeline_busy")
            else:
                self.store.update(run_id, status="failed", message="The worker could not start the preparation pipeline.", errorCode="worker_start_failed")
        except Exception:
            current = self.store.get(run_id) or {}
            if current.get("status") != "interrupted":
                self.store.update(run_id, status="failed", message="The worker could not start the preparation pipeline.", errorCode="worker_start_failed")
        finally:
            with self._process_lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None
                    self._active_process = None
