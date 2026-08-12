"""Path helpers for daily pipeline."""

from __future__ import annotations

from pathlib import Path

# Keys resolved against daily_pipeline/ (outer folder)
_OUTER_KEYS = frozenset({"local_cache", "m4_shared_cache", "repo_root"})


def utils_root() -> Path:
    """Milestone 6/daily_pipeline/utils/"""
    return Path(__file__).resolve().parent


def pipeline_root() -> Path:
    """Milestone 6/daily_pipeline/ (public entry folder)."""
    return utils_root().parent


def resolve_path(cfg: dict, key: str) -> Path:
    rel = cfg["paths"][key]
    base = pipeline_root() if key in _OUTER_KEYS else utils_root()
    return (base / rel).resolve()


def gcs_uri(bucket: str, prefix: str, *parts: str) -> str:
    path = "/".join([prefix.strip("/"), *parts])
    return f"gs://{bucket}/{path}"
