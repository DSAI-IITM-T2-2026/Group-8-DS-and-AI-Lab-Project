"""Earth Engine init that never blocks a cron / Cloud Run job on a browser login."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_EE_SCOPES = (
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/devstorage.full_control",
)


def _load_ee_credentials():
    """Prefer the mounted service-account JSON; never rely on interactive EE tokens."""
    from google.auth import default as google_auth_default
    from google.oauth2 import service_account

    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if key_path and Path(key_path).is_file():
        return service_account.Credentials.from_service_account_file(
            key_path, scopes=list(_EE_SCOPES)
        )

    credentials, _project = google_auth_default(scopes=list(_EE_SCOPES))
    return credentials


def initialize_ee(project_id: str) -> None:
    """Initialize EE with ADC / service-account credentials.

    Explicitly passes credentials into ``ee.Initialize``. Newer earthengine-api
    builds otherwise call ``get_persistent_credentials()`` (user oauth from
    ``earthengine authenticate``), which fails on VMs/cron even when a service
    account JSON is mounted.

    Interactive ``ee.Authenticate()`` is disabled by default, even on a TTY,
    because API workers may inherit a terminal and accidentally open a browser.
    It is available only for an intentional local setup session via
    ``WILDFIRE_ALLOW_INTERACTIVE_EE_AUTH=true``.
    """
    import ee

    try:
        credentials = _load_ee_credentials()
        ee.Initialize(credentials=credentials, project=project_id)
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
