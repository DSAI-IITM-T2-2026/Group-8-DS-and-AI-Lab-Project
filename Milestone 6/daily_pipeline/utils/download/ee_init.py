"""Earth Engine init that never blocks a cron / Cloud Run job on a browser login."""

from __future__ import annotations

import sys


def initialize_ee(project_id: str) -> None:
    """Initialize EE with ADC / service-account credentials.

    Interactive ``ee.Authenticate()`` is only used on a TTY. Scheduled jobs
    must already have Application Default Credentials.
    """
    import ee

    try:
        ee.Initialize(project=project_id)
        return
    except Exception as exc:
        if not sys.stdin.isatty():
            raise RuntimeError(
                f"Earth Engine init failed for project={project_id}. "
                "Cron/GCS jobs need Application Default Credentials "
                "(GOOGLE_APPLICATION_CREDENTIALS or gcloud auth application-default login). "
                f"Original error: {exc}"
            ) from exc
        ee.Authenticate()
        ee.Initialize(project=project_id)
