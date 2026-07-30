from __future__ import annotations

import re
from calendar import isleap
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from ..io import atomic_json, gcs_list


def _probe(
    uri: str,
    expected_months: list[int],
    month_pattern: str,
    expected_object_count: int | None = None,
) -> dict:
    try:
        objects = gcs_list(uri, timeout=180)
        available_months = sorted(
            {int(match.group(1)) for item in objects if (match := re.search(month_pattern, item))}
        )
        missing_months = sorted(set(expected_months) - set(available_months))
        missing_objects = (
            max(expected_object_count - len(objects), 0)
            if expected_object_count is not None
            else None
        )
        return {
            "uri": uri,
            "status": (
                "ok" if objects and not missing_months and not (missing_objects or 0) else "missing"
            ),
            "object_count": len(objects),
            "expected_object_count": expected_object_count,
            "missing_object_count": missing_objects,
            "example": objects[0] if objects else None,
            "available_months": available_months,
            "missing_months": missing_months,
        }
    except (PermissionError, OSError) as exc:
        return {"uri": uri, "status": "error", "error": str(exc)}


def verify_gcs(cfg: dict, years: list[int], months: list[int]) -> dict:
    specifications: list[tuple[str, list[int], str, int | None] | dict] = []
    for year in years:
        specifications.append(
            (
                f"{cfg['gcs']['era5_prefix'].rstrip('/')}/{year}/era5_{year}_*.nc",
                months,
                r"_(\d{2})\.nc$",
                len(months),
            )
        )
        specifications.append(
            (
                f"{cfg['gcs']['firms_prefix'].rstrip('/')}/{year}-*.tif",
                months,
                rf"/{year}-(\d{{2}})-",
                (366 if isleap(year) else 365) if months == list(range(1, 13)) else None,
            )
        )
        for source in ("s2", "s5p"):
            prefix = cfg["gcs"][f"{source}_prefix_by_year"].get(str(year))
            if not prefix:
                specifications.append(
                    {
                        "uri": f"{source}:{year}",
                        "status": "error",
                        "error": "no configured prefix",
                    }
                )
                continue
            specifications.append(
                (
                    f"{prefix.rstrip('/')}/year={year}/month=*/window=*/features.csv",
                    months,
                    r"/month=(\d{2})/",
                    (
                        (366 if isleap(year) else 365)
                        if source == "s5p" and months == list(range(1, 13))
                        else None
                    ),
                )
            )
    fixed_errors = [spec for spec in specifications if isinstance(spec, dict)]
    runnable = [spec for spec in specifications if isinstance(spec, tuple)]
    with ThreadPoolExecutor(max_workers=int(cfg["execution"]["max_workers"])) as pool:
        probes = list(pool.map(lambda spec: _probe(*spec), runnable))
    probes.extend(fixed_errors)
    probes.sort(key=lambda probe: probe["uri"])
    status = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "config_hash": cfg["_config_hash"],
        "probes": probes,
        "summary": {
            state: sum(probe["status"] == state for probe in probes)
            for state in ("ok", "missing", "error")
        },
    }
    destination = cfg["paths"]["report_dir"] / "gcs_validation.json"
    atomic_json(status, destination)
    return status
