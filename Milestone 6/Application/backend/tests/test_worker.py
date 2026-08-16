import json
from types import SimpleNamespace

from app.store import RunStore
from app.worker import EVENT_PREFIX, PipelineWorker, safe_error_code


def test_worker_errors_are_classified_without_exposing_output():
    assert safe_error_code("Earth Engine init failed: credentials missing") == "cloud_authentication_failed"
    assert safe_error_code("CDS credentials not found. Set CDS_API_KEY") == "cds_authentication_failed"
    assert safe_error_code("Timed out waiting for Sentinel-5P") == "external_data_timeout"
    assert safe_error_code("quota exceeded 429") == "cloud_quota_exceeded"
    assert safe_error_code("unexpected stack trace") == "pipeline_failed"


def test_worker_preserves_unavailable_terminal_event(tmp_path, monkeypatch):
    pipeline_root = tmp_path / "pipeline"
    pipeline_root.mkdir()
    settings = SimpleNamespace(
        python_executable="python3",
        pipeline_root=pipeline_root,
        logs_dir=tmp_path / "logs",
    )
    settings.logs_dir.mkdir()

    class Store:
        def __init__(self):
            self.run = {
                "runId": "run-1", "status": "running", "stage": "inventory",
                "artifact": None, "message": "running",
            }

        def update(self, _run_id, **values):
            self.run.update(values)

        def get(self, _run_id):
            return dict(self.run)

    event = {
        "status": "unavailable",
        "stage": "inventory",
        "message": "Tomorrow's data is not available yet.",
        "sourceInventory": {"firms": {"required": 1, "available": 0, "missing": 1}},
    }

    class Process:
        pid = 12345
        stdout = iter([EVENT_PREFIX + json.dumps(event) + "\n"])

        @staticmethod
        def wait():
            return 0

    monkeypatch.setattr("app.worker.subprocess.Popen", lambda *args, **kwargs: Process())
    store = Store()
    PipelineWorker(settings, store)._execute({"runId": "run-1", "predictionDate": "2026-08-14"})

    assert store.run["status"] == "unavailable"
    assert store.run["message"] == "Tomorrow's data is not available yet."
    assert store.run.get("errorCode") is None


def test_cancelled_process_is_signalled_and_not_overwritten_as_failed(tmp_path, monkeypatch):
    pipeline_root = tmp_path / "pipeline"
    pipeline_root.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    store = RunStore(tmp_path / "runs.sqlite3")
    queued, _ = store.create_or_reuse("2026-08-14")
    claimed = store.claim_next()
    assert claimed is not None
    worker = PipelineWorker(
        SimpleNamespace(
            python_executable="python3",
            pipeline_root=pipeline_root,
            logs_dir=logs_dir,
        ),
        store,
    )
    signals = []

    class Output:
        def __iter__(self):
            worker.cancel(queued["runId"])
            return iter(())

    class Process:
        pid = 12345
        stdout = Output()

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(timeout=None):
            return -15

    monkeypatch.setattr("app.worker.subprocess.Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(worker, "_signal_process", lambda process, sig: signals.append((process.pid, sig)))
    monkeypatch.setattr(worker, "_force_kill_after_grace", lambda process: None)

    worker._execute(claimed)

    result = store.get(queued["runId"])
    assert result["status"] == "interrupted"
    assert result["errorCode"] == "cancelled_by_user"
    assert signals
