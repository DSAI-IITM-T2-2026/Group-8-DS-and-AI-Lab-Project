"""
Config loading + validation.

Owns exactly one job: read config.yaml, sanity-check it, and hand back a
plain dict plus a helper to build the AOI geometry from it. Nothing else in
this package should read config.yaml directly — go through here.
"""

import logging

import yaml

logger = logging.getLogger("sentinel2")

VALID_LOGIC_MODES = ("confirmed", "alt_design")
PLACEHOLDER_PROJECTS = {"your-gee-project-id", "changeme", ""}
PLACEHOLDER_BUCKETS = {"your-bucket-name", "changeme", ""}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)

    mode = cfg.get("logic_mode", "confirmed")
    if mode == "alt_design":
        logger.warning(
            "=" * 70 + "\n"
            "RUNNING IN alt_design MODE.\n"
            "This logic (30%% cloud filter, SCL masking, computed indices) was\n"
            "NEVER confirmed to have run in production. Every export from this\n"
            "run will be tagged ALT_DESIGN_UNVERIFIED. Do not treat this output\n"
            "as equivalent to the existing production data.\n" + "=" * 70
        )
    else:
        logger.info(
            "Running in 'confirmed' mode — production thresholds with SCL+QA60 "
            "cloud masking (SCL required because QA60 is unpopulated on "
            "S2_SR_HARMONIZED after ~2022)."
        )

    return cfg


def _validate(cfg: dict) -> None:
    mode = cfg.get("logic_mode", "confirmed")
    if mode not in VALID_LOGIC_MODES:
        raise ValueError(f"logic_mode must be one of {VALID_LOGIC_MODES}, got: {mode}")

    project = (cfg.get("ee_project") or "").strip()
    if project in PLACEHOLDER_PROJECTS:
        raise ValueError(
            "config ee_project is still a placeholder. Set it to your Google "
            "Earth Engine cloud project ID in config.yaml."
        )

    bucket = (cfg.get("export", {}).get("gcs_bucket") or "").strip()
    if bucket in PLACEHOLDER_BUCKETS:
        raise ValueError(
            "config export.gcs_bucket is still a placeholder. Set it to an "
            "existing GCS bucket your EE identity can write to."
        )

    temporal = cfg.get("temporal") or {}
    try:
        start_year = int(temporal["start_year"])
        end_year = int(temporal["end_year"])
        step_days = int(temporal["step_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "temporal.start_year, temporal.end_year, and temporal.step_days "
            "must all be integers."
        ) from exc

    # Pipeline steps backward: start_year is the latest year, end_year the earliest.
    if start_year < end_year:
        raise ValueError(
            f"temporal.start_year ({start_year}) must be >= temporal.end_year "
            f"({end_year}). This pipeline walks backward in time; start_year is "
            "the newest year and end_year is the oldest. "
            "Example: start_year: 2025, end_year: 2016."
        )

    if step_days < 1:
        raise ValueError(f"temporal.step_days must be >= 1, got: {step_days}")

    if "aoi" not in cfg:
        raise ValueError("config missing required 'aoi' section.")


def get_aoi(cfg: dict):
    """Returns an ee.Geometry.Rectangle for the configured AOI.

    Import of `ee` is local to avoid forcing earthengine-api to be importable
    just to load/validate a config file (useful for quick config unit tests).
    """
    import ee

    a = cfg["aoi"]
    return ee.Geometry.Rectangle([a["west"], a["south"], a["east"], a["north"]])
