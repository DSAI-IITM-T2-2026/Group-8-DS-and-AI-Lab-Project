"""California-time cutoff policy for provisional tomorrow forecasts."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo


S2_NAME = re.compile(r"s2feat_(\d{8})_(\d{8})\.(?:parquet|csv)$")
S5P_NAME = re.compile(r"s5pfeat_(\d{8})_(\d{8})\.parquet$")


def _parse_clock(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def forecast_cutoff_at(label_date: date, cfg: dict[str, Any]) -> datetime:
    """Return 06:30 California time on the day before ``label_date``."""
    timezone_name = str(cfg["task"].get("timezone", "America/Los_Angeles"))
    clock = _parse_clock(str(cfg["task"].get("forecast_cutoff_local_time", "06:30")))
    return datetime.combine(label_date - timedelta(days=1), clock, ZoneInfo(timezone_name))


def california_now(cfg: dict[str, Any]) -> datetime:
    return datetime.now(ZoneInfo(str(cfg["task"].get("timezone", "America/Los_Angeles"))))


def _blob_time(blob: Any) -> datetime | None:
    value = getattr(blob, "updated", None) or getattr(blob, "time_created", None)
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo("UTC"))
    return value


def _available_by(blob: Any, cutoff_at: datetime) -> bool:
    created = _blob_time(blob)
    return created is not None and created <= cutoff_at


def _snapshot(
    *,
    required: int,
    available: int,
    required_through: date | None,
    selected_through: date | None,
    mode: str,
    ready: bool,
    message: str,
    selected_start: date | None = None,
) -> dict[str, Any]:
    age = None
    if required_through is not None and selected_through is not None:
        age = max(0, (required_through - selected_through).days)
    return {
        "required": required,
        "available": available,
        "missing": max(0, required - available),
        "scheduled": 0,
        "pending": 0,
        "requiredThroughDate": required_through.isoformat() if required_through else None,
        "selectedThroughDate": selected_through.isoformat() if selected_through else None,
        "selectedWindowStartDate": selected_start.isoformat() if selected_start else None,
        "ageDays": age,
        "mode": mode,
        "ready": ready,
        "message": message,
    }


def _list_blobs(client: Any, bucket: str, prefix: str) -> list[Any]:
    return list(client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"))


def _exact_blobs(
    client: Any,
    bucket: str,
    names: list[str],
    cutoff_at: datetime,
) -> tuple[int, list[Any]]:
    ready: list[Any] = []
    target_bucket = client.bucket(bucket)
    for name in names:
        blob = target_bucket.blob(name)
        if not blob.exists():
            continue
        blob.reload()
        if _available_by(blob, cutoff_at):
            ready.append(blob)
    return len(ready), ready


def build_cutoff_inventory(
    label_date: date,
    cfg: dict[str, Any],
    *,
    storage_client: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Select causal inputs that were present by the forecast cutoff.

    ERA5 and FIRMS remain exact because silently treating missing weather or fire
    history as zero changes the trained feature semantics. EO sources may use
    their trained causal fallback windows.
    """
    if storage_client is None:
        from google.cloud import storage

        storage_client = storage.Client(project=cfg["gcs"].get("project"))

    bucket = cfg["gcs"]["bucket"]
    prefixes = cfg["gcs"]["prefixes"]
    cutoff_at = forecast_cutoff_at(label_date, cfg)
    eo_asof = label_date - timedelta(days=1)
    feature_end = label_date - timedelta(
        days=int(cfg["task"].get("era5_lag_days", 5))
        + int(cfg["task"].get("lead_days", 1))
    )
    lookback = int(cfg["task"].get("lookback_days", 30))
    history = int(cfg["task"].get("history_days", 7))

    feature_offset = int(cfg["task"].get("era5_lag_days", 5)) + int(
        cfg["task"].get("lead_days", 1)
    )
    era5_start = label_date - timedelta(days=lookback + history + feature_offset)
    era5_days = [
        era5_start + timedelta(days=i)
        for i in range((feature_end - era5_start).days + 1)
    ]
    era5_prefix = prefixes["era5"].rstrip("/")
    era5_blobs = {
        blob.name: blob
        for blob in _list_blobs(storage_client, bucket, era5_prefix)
        if _available_by(blob, cutoff_at)
    }
    era5_count = 0
    for day in era5_days:
        daily = f"{era5_prefix}/{day.year:04d}/era5_{day.year:04d}_{day.month:02d}_{day.day:02d}.nc"
        monthly = (
            f"{era5_prefix}/{day.year:04d}/era5_{day.year:04d}_{day.month:02d}.nc",
            f"{era5_prefix}/raw/{day.year:04d}/era5_{day.year:04d}_{day.month:02d}.nc",
        )
        if daily in era5_blobs or any(name in era5_blobs for name in monthly):
            era5_count += 1
    era5_ready = era5_count == len(era5_days)

    firms_start = label_date - timedelta(days=lookback)
    firms_end = label_date - timedelta(days=2)
    firms_days = [
        firms_start + timedelta(days=i)
        for i in range((firms_end - firms_start).days + 1)
    ]
    firms_names = [
        f"{prefixes['firms'].rstrip('/')}/{day.isoformat()}.tif" for day in firms_days
    ]
    firms_count, _ = _exact_blobs(storage_client, bucket, firms_names, cutoff_at)
    firms_ready = firms_count == len(firms_names)

    s2_candidates: list[tuple[date, date, Any]] = []
    for blob in _list_blobs(storage_client, bucket, prefixes["sentinel2"]):
        match = S2_NAME.search(blob.name)
        if not match or not _available_by(blob, cutoff_at):
            continue
        start = datetime.strptime(match.group(1), "%Y%m%d").date()
        end = datetime.strptime(match.group(2), "%Y%m%d").date()
        if end <= eo_asof:
            s2_candidates.append((start, end, blob))
    s2_candidates.sort(key=lambda item: item[1])
    s2_selected = s2_candidates[-1] if s2_candidates else None
    s2_ready = s2_selected is not None

    max_s5p_age = int(
        cfg.get("cutoff_policy", {}).get(
            "sentinel5p_max_age_days",
            cfg.get("s5p_features", {}).get("forward_fill_max_days", 7),
        )
    )
    s5p_available: list[tuple[date, Any]] = []
    for blob in _list_blobs(storage_client, bucket, prefixes["sentinel5p"]):
        match = S5P_NAME.search(blob.name)
        if not match or not _available_by(blob, cutoff_at):
            continue
        observed = datetime.strptime(match.group(2), "%Y%m%d").date()
        if observed <= eo_asof:
            s5p_available.append((observed, blob))
    s5p_candidates = [
        item for item in s5p_available
        if (eo_asof - item[0]).days <= max_s5p_age
    ]
    s5p_candidates.sort(key=lambda item: item[0])
    s5p_selected = s5p_candidates[-1] if s5p_candidates else None
    s5p_ready = s5p_selected is not None

    dem_name = f"{prefixes['dem'].rstrip('/')}/{cfg['paths']['dem_gcs_name']}"
    dem_count, _ = _exact_blobs(storage_client, bucket, [dem_name], cutoff_at)
    dem_ready = dem_count == 1

    inventory = {
        "era5": _snapshot(
            required=len(era5_days), available=era5_count,
            required_through=feature_end,
            selected_through=feature_end if era5_ready else None,
            mode="exact", ready=era5_ready,
            message=(f"ERA5 history is complete through {feature_end}." if era5_ready else f"ERA5 history is incomplete through {feature_end}."),
        ),
        "firms": _snapshot(
            required=len(firms_names), available=firms_count,
            required_through=firms_end,
            selected_through=firms_end if firms_ready else None,
            mode="exact", ready=firms_ready,
            message=(f"FIRMS neighbour history is complete through {firms_end}." if firms_ready else f"FIRMS neighbour history is incomplete through {firms_end}."),
        ),
        "sentinel2": _snapshot(
            required=1, available=1 if s2_ready else 0,
            required_through=eo_asof,
            selected_through=s2_selected[1] if s2_selected else None,
            selected_start=s2_selected[0] if s2_selected else None,
            mode="latest_causal", ready=s2_ready,
            message=(f"Using the latest completed Sentinel-2 window ending {s2_selected[1]}." if s2_selected else "No completed causal Sentinel-2 window was available by the cutoff."),
        ),
        "sentinel5p": _snapshot(
            required=1, available=1 if s5p_ready else 0,
            required_through=eo_asof,
            selected_through=s5p_selected[0] if s5p_selected else None,
            mode="latest_causal", ready=s5p_ready,
            message=(f"Using the latest Sentinel-5P observation from {s5p_selected[0]}." if s5p_selected else f"No Sentinel-5P observation within {max_s5p_age} days was available by the cutoff."),
        ),
        "dem": _snapshot(
            required=1, available=dem_count,
            required_through=None, selected_through=None,
            mode="static", ready=dem_ready,
            message=("Static Copernicus DEM is available." if dem_ready else "Static Copernicus DEM is unavailable."),
        ),
    }
    # Internal enforcement list: Stage C must not discover an EO object that
    # arrived after the immutable cutoff snapshot. API models ignore this
    # implementation field, while provenance retains the exact object roster.
    inventory["sentinel2"]["objectNames"] = [item[2].name for item in s2_candidates]
    inventory["sentinel5p"]["objectNames"] = [item[1].name for item in s5p_available]
    return inventory


def unavailable_message(inventory: dict[str, dict[str, Any]]) -> str:
    missing = [item["message"] for item in inventory.values() if not item.get("ready")]
    return missing[0] if missing else "Required causal source data is unavailable."
