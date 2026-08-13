from __future__ import annotations

import json
import os
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


class PipelineWorker:
    def __init__(self, settings: Settings, store: RunStore):
        self.settings = settings
        self.store = store
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
        try:
            with pipeline_lock(self.settings.pipeline_root):
                with log_path.open("a", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command, cwd=self.settings.pipeline_root, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    self.store.update(run_id, pid=process.pid)
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
                            self.store.update(run_id, **updates)
                    code = process.wait()
            if code == 0:
                current = self.store.get(run_id) or {}
                if current.get("artifact"):
                    self.store.update(run_id, status="succeeded", stage="completed", message="Prediction data is ready.")
                else:
                    self.store.update(run_id, status="failed", stage="completed", message="The pipeline ended without a verified artifact.", errorCode="artifact_missing")
            else:
                output = "\n".join(lines[-80:])
                self.store.update(run_id, status="failed", message="The preparation pipeline failed. Review the server log and retry after correcting the dependency.", errorCode=safe_error_code(output))
        except RuntimeError as exc:
            if str(exc) == "pipeline_already_running":
                self.store.update(run_id, status="failed", message="Another scheduled pipeline run is already using the shared cache. Retry after it finishes.", errorCode="pipeline_busy")
            else:
                self.store.update(run_id, status="failed", message="The worker could not start the preparation pipeline.", errorCode="worker_start_failed")
        except Exception:
            self.store.update(run_id, status="failed", message="The worker could not start the preparation pipeline.", errorCode="worker_start_failed")
