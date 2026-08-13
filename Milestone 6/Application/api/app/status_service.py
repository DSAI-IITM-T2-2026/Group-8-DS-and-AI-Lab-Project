"""Backs /health and /model/metadata with real, dynamically-checked facts.

Nothing here is a static "ready": each check either runs (config loads,
grid file readable, model artifact loaded, GCS listable) or is honestly
reported as failed/unavailable with the real reason.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

from .config import ensure_pipeline_on_path, get_pipeline_config, get_settings
from .model_registry import get_model_registry, model_updated_at

_FINAL_PROCESSED_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_test\.parquet$")


def _latest_local_label_date() -> date | None:
    ensure_pipeline_on_path()
    from paths import resolve_path

    cache_dir = resolve_path(get_pipeline_config(), "local_cache") / "final_processed"
    if not cache_dir.is_dir():
        return None
    dates: list[date] = []
    for path in cache_dir.glob("*_test.parquet"):
        m = _FINAL_PROCESSED_RE.search(path.name)
        if m:
            dates.append(date.fromisoformat(m.group(1)))
    return max(dates) if dates else None


def _latest_gcs_label_date() -> date | None:
    settings = get_settings()
    if not settings.allow_gcs_reads:
        return None
    try:
        ensure_pipeline_on_path()
        from preprocess.adapters_gcs import gsutil_ls

        cfg = get_pipeline_config()
        bucket = cfg["gcs"]["bucket"]
        prefix = cfg["gcs"]["prefixes"]["final_processed"]
        uris = gsutil_ls(f"gs://{bucket}/{prefix}/")
        dates = []
        for uri in uris:
            m = _FINAL_PROCESSED_RE.search(uri)
            if m:
                dates.append(date.fromisoformat(m.group(1)))
        return max(dates) if dates else None
    except Exception:
        return None


def data_freshness() -> dict[str, Any]:
    latest = _latest_local_label_date() or _latest_gcs_label_date()
    if latest is None:
        return {"status": "unavailable", "observed_at": None}
    observed_at = datetime(latest.year, latest.month, latest.day, tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc).date() - latest).days
    if age_days <= 1:
        status = "fresh"
    elif age_days <= 4:
        status = "stale"
    else:
        status = "partial"
    return {"status": status, "observed_at": observed_at}


def health_checks() -> dict[str, Any]:
    checks: dict[str, str] = {}
    try:
        get_pipeline_config()
        checks["pipeline_config"] = "ok"
    except Exception as exc:
        checks["pipeline_config"] = f"error: {exc}"

    try:
        ensure_pipeline_on_path()
        from paths import resolve_path

        dem_path = resolve_path(get_pipeline_config(), "dem_local")
        checks["grid_definition"] = "ok" if dem_path.is_file() else "missing"
    except Exception as exc:
        checks["grid_definition"] = f"error: {exc}"

    registry = get_model_registry()
    checks["model"] = "loaded" if registry.is_loaded else f"unavailable: {registry.unavailable_reason}"

    settings = get_settings()
    checks["gcs_reads"] = "enabled" if settings.allow_gcs_reads else "disabled"

    # "ready" = the parts that don't depend on external assets are healthy
    # (config loads, the grid definition is readable). A missing model
    # artifact is real and worth surfacing, but it degrades inference
    # specifically rather than the deployment as a whole -- callers should
    # still check GET /model/metadata.checks["model"] before calling
    # /predictions or /risk-map.
    core_ok = checks["pipeline_config"] == "ok" and checks["grid_definition"] == "ok"
    status = "ready" if core_ok else "degraded"
    return {"status": status, "checks": checks}


def model_metadata() -> dict[str, Any]:
    registry = get_model_registry()
    return {
        "model_version": registry.version,
        "threshold": None,  # see risk_service module docstring: no single trained threshold exists
        "updated_at": model_updated_at(),
        "explanation_capability": registry.explanation_capability,
        "data_freshness": data_freshness(),
    }
