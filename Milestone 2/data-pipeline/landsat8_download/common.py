"""
Shared helpers for the wildfire satellite fetch pipeline (Landsat / Sentinel).

If your team already has a `common.py` with these names, use that one instead —
this is a self-contained reimplementation built from the pipeline architecture
doc, in case the original isn't available on this machine.
"""

import json
import logging
import time
from datetime import date, timedelta

import ee
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("wildfire_pipeline")


def load_config(path: str = "config.yaml") -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def init_ee(cfg: dict) -> None:
    """Initialize Earth Engine, authenticating interactively if needed."""
    project = cfg["gee"]["project"]
    try:
        ee.Initialize(project=project)
    except Exception:
        logger.info("Earth Engine not authenticated yet — launching browser auth flow.")
        ee.Authenticate()
        ee.Initialize(project=project)


def get_aoi(cfg: dict) -> ee.Geometry:
    area = cfg["area"]
    return ee.Geometry.Rectangle([area["west"], area["south"], area["east"], area["north"]])


def descending_date_steps(cfg: dict):
    """
    Yield (start_date_iso, end_date_iso, label) tuples, stepping backward from
    start_year (latest) to end_year (earliest) by step_days, each covering a
    trailing window_days lookback window.
    """
    t = cfg["temporal"]
    step_days = t["step_days"]
    window_days = t["window_days"]
    current = date(t["start_year"], 12, 31)
    floor_date = date(t["end_year"], 1, 1)

    while current >= floor_date:
        window_start = current - timedelta(days=window_days)
        label = current.strftime("%Y_%m_%d")
        yield (window_start.isoformat(), current.isoformat(), label)
        current -= timedelta(days=step_days)


def safe_get_info(ee_object, retries: int = 4, wait_seconds: int = 5):
    """getInfo() with retries — GEE calls occasionally time out under load."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return ee_object.getInfo()
        except Exception as e:
            last_err = e
            logger.warning(f"getInfo() failed (attempt {attempt}/{retries}): {e}")
            time.sleep(wait_seconds)
    raise RuntimeError(f"getInfo() failed after {retries} attempts: {last_err}")


def submit_export(image, description, subfolder, aoi, scale_m, cfg):
    """Submit an Earth Engine export-to-GCS task. Does NOT mark progress — caller does that."""
    export_cfg = cfg["export"]
    file_prefix = f"{export_cfg['gcs_prefix']}/raw/{subfolder}/{description}"
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=description,
        bucket=export_cfg["gcs_bucket"],
        fileNamePrefix=file_prefix,
        region=aoi,
        scale=scale_m,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    logger.info(f"Submitted export task '{description}' (task id: {task.id})")
    return task


class GCSProgressLog:
    """
    Progress tracker stored as a JSON file in GCS (not locally), so it's shared
    across machines/collaborators and survives local disk wipes.

    Status values: "done", "empty", "submitted" (task fired, not yet confirmed).
    IMPORTANT: only call mark(label, "done") after confirming the GCS file
    actually exists (or the EE task status is COMPLETED) — never immediately
    after task.start(), which is the bug that caused the original stuck queue.
    """

    def __init__(self, name: str, cfg: dict):
        self.name = name
        export_cfg = cfg["export"]
        self.bucket_name = export_cfg["gcs_bucket"]
        self.blob_path = f"{export_cfg['gcs_prefix']}/metadata/{name}_progress.json"
        self.client = storage.Client(project=cfg["gee"]["project"])
        self.bucket = self.client.bucket(self.bucket_name)
        self._data = self._load()

    def _load(self) -> dict:
        blob = self.bucket.blob(self.blob_path)
        if blob.exists(self.client):
            return json.loads(blob.download_as_text())
        return {}

    def _save(self) -> None:
        blob = self.bucket.blob(self.blob_path)
        blob.upload_from_string(json.dumps(self._data, indent=2), content_type="application/json")

    def is_done(self, label: str) -> bool:
        return self._data.get(label) in ("done", "empty")

    def status(self, label: str):
        return self._data.get(label)

    def mark(self, label: str, status: str) -> None:
        self._data[label] = status
        self._save()  # flush immediately: durability > write-speed for this log