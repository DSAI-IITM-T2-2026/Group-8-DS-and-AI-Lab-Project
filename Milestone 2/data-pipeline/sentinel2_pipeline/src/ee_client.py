"""Earth Engine authentication + initialization."""

import logging

import ee

logger = logging.getLogger("sentinel2")


def ee_init(cfg: dict) -> None:
    project = cfg.get("ee_project")
    try:
        ee.Initialize(project=project)
    except Exception:
        logger.info("No cached EE credentials found — launching authentication flow.")
        ee.Authenticate()
        ee.Initialize(project=project)
    logger.info(f"Earth Engine initialized (project={project}).")
