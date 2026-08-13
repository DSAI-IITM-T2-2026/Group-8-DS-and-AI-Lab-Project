"""Earth Engine init that never blocks a cron / Cloud Run job on a browser login."""

from __future__ import annotations

import os
import sys


def initialize_ee(project_id: str) -> None:
    """Initialize EE with ADC / service-account credentials.

    Interactive ``ee.Authenticate()`` is disabled by default, even on a TTY,
    because API workers may inherit a terminal and accidentally open a browser.
    It is available only for an intentional local setup session via
    ``WILDFIRE_ALLOW_INTERACTIVE_EE_AUTH=true``.
    """
    import ee

    try:
        ee.Initialize(project=project_id)
        return
    except Exception as exc:
        interactive_allowed = (
            os.environ.get("WILDFIRE_ALLOW_INTERACTIVE_EE_AUTH", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if not (interactive_allowed and sys.stdin.isatty()):
            raise RuntimeError(
                f"Earth Engine init failed for project={project_id}. "
                "API/cron jobs need Earth Engine-enabled Application Default Credentials "
                "(GOOGLE_APPLICATION_CREDENTIALS or gcloud auth application-default login). "
                "Interactive browser authentication is disabled for worker safety. "
                f"Original error: {exc}"
            ) from exc
        ee.Authenticate()
        ee.Initialize(project=project_id)
