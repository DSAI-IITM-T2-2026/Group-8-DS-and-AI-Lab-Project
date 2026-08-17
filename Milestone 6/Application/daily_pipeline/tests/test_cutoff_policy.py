from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import sys

import pandas as pd

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from cutoff_policy import build_cutoff_inventory, forecast_cutoff_at


class Blob:
    def __init__(self, name: str, created: datetime | None):
        self.name = name
        self.time_created = created
        self.updated = created
        self.size = 1

    def exists(self):
        return self.name in self.bucket.objects

    def reload(self):
        source = self.bucket.objects[self.name]
        self.time_created = source.time_created
        self.updated = source.updated


class Bucket:
    def __init__(self, objects):
        self.objects = objects
        for blob in objects.values():
            blob.bucket = self

    def blob(self, name):
        blob = self.objects.get(name, Blob(name, None))
        blob.bucket = self
        return blob


class Client:
    def __init__(self, objects):
        self._bucket = Bucket(objects)

    def bucket(self, _name):
        return self._bucket

    def list_blobs(self, _bucket, prefix):
        return [blob for name, blob in self._bucket.objects.items() if name.startswith(prefix)]


def config():
    return {
        "task": {
            "timezone": "America/Los_Angeles",
            "forecast_cutoff_local_time": "06:30",
            "era5_lag_days": 5,
            "lead_days": 1,
            "lookback_days": 30,
            "history_days": 7,
        },
        "cutoff_policy": {"sentinel5p_max_age_days": 7},
        "gcs": {
            "bucket": "test",
            "project": "test",
            "prefixes": {
                "era5": "era5",
                "firms": "firms",
                "sentinel2": "s2",
                "sentinel5p": "s5p",
                "dem": "dem",
            },
        },
        "paths": {"dem_gcs_name": "dem.parquet"},
    }


def ready_objects(target: date, *, s5p_age: int = 2):
    cfg = config()
    cutoff = forecast_cutoff_at(target, cfg)
    created = cutoff.astimezone(timezone.utc) - timedelta(minutes=1)
    feature_end = target - timedelta(days=6)
    era5_start = target - timedelta(days=43)
    objects = {}
    day = era5_start
    while day <= feature_end:
        name = f"era5/{day.year:04d}/era5_{day.year:04d}_{day.month:02d}_{day.day:02d}.nc"
        objects[name] = Blob(name, created)
        day += timedelta(days=1)
    day = target - timedelta(days=30)
    while day <= target - timedelta(days=2):
        name = f"firms/{day.isoformat()}.tif"
        objects[name] = Blob(name, created)
        day += timedelta(days=1)
    eo = target - timedelta(days=1)
    for start, end in ((eo - timedelta(days=9), eo - timedelta(days=5)), (eo - timedelta(days=4), eo)):
        name = f"s2/s2feat_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        objects[name] = Blob(name, created)
    future_name = f"s2/s2feat_{eo:%Y%m%d}_{target:%Y%m%d}.parquet"
    objects[future_name] = Blob(future_name, created)
    observed = eo - timedelta(days=s5p_age)
    s5p_name = f"s5p/s5pfeat_{observed:%Y%m%d}_{observed:%Y%m%d}.parquet"
    objects[s5p_name] = Blob(s5p_name, created)
    objects["dem/dem.parquet"] = Blob("dem/dem.parquet", created)
    return cfg, objects


def test_cutoff_is_0630_day_before_across_daylight_saving():
    spring = forecast_cutoff_at(date(2026, 3, 9), config())
    fall = forecast_cutoff_at(date(2026, 11, 2), config())
    assert (spring.hour, spring.minute, spring.utcoffset()) == (6, 30, timedelta(hours=-7))
    assert (fall.hour, fall.minute, fall.utcoffset()) == (6, 30, timedelta(hours=-8))


def test_latest_causal_selection_and_date_invariants():
    target = date(2026, 8, 18)
    cfg, objects = ready_objects(target)
    inventory = build_cutoff_inventory(target, cfg, storage_client=Client(objects))
    assert all(item["ready"] for item in inventory.values())
    assert inventory["era5"]["selectedThroughDate"] == "2026-08-12"
    assert inventory["firms"]["selectedThroughDate"] == "2026-08-16"
    assert inventory["sentinel2"]["selectedThroughDate"] == "2026-08-17"
    assert inventory["sentinel5p"]["ageDays"] == 2


def test_objects_created_after_cutoff_are_excluded():
    target = date(2026, 8, 18)
    cfg, objects = ready_objects(target)
    objects["dem/dem.parquet"].time_created = forecast_cutoff_at(target, cfg) + timedelta(seconds=1)
    objects["dem/dem.parquet"].updated = objects["dem/dem.parquet"].time_created
    inventory = build_cutoff_inventory(target, cfg, storage_client=Client(objects))
    assert inventory["dem"]["ready"] is False


def test_open_month_era5_uses_one_day_fallback_not_monthly_presence():
    target = date(2026, 8, 18)
    cfg, objects = ready_objects(target)
    endpoint = "era5/2026/era5_2026_08_12.nc"
    del objects[endpoint]
    created = forecast_cutoff_at(target, cfg).astimezone(timezone.utc) - timedelta(minutes=1)
    monthly = Blob("era5/2026/era5_2026_08.nc", created)
    objects[monthly.name] = monthly

    inventory = build_cutoff_inventory(target, cfg, storage_client=Client(objects))

    assert inventory["era5"]["ready"] is True
    assert inventory["era5"]["selectedThroughDate"] == "2026-08-11"
    assert inventory["era5"]["requiredThroughDate"] == "2026-08-12"
    assert inventory["era5"]["ageDays"] == 1
    assert inventory["era5"]["mode"] == "latest_causal"
    assert inventory["era5"]["exactAvailable"] is False
    assert "one day older" in inventory["era5"]["message"]


def test_era5_rejects_two_day_old_endpoint():
    target = date(2026, 8, 18)
    cfg, objects = ready_objects(target)
    del objects["era5/2026/era5_2026_08_12.nc"]
    del objects["era5/2026/era5_2026_08_11.nc"]

    inventory = build_cutoff_inventory(target, cfg, storage_client=Client(objects))

    assert inventory["era5"]["ready"] is False
    assert inventory["era5"]["selectedThroughDate"] == "2026-08-10"
    assert inventory["era5"]["ageDays"] == 2


def test_exact_era5_after_cutoff_is_selected_for_late_refresh():
    target = date(2026, 8, 18)
    cfg, objects = ready_objects(target)
    endpoint = objects["era5/2026/era5_2026_08_12.nc"]
    endpoint.time_created = forecast_cutoff_at(target, cfg) + timedelta(hours=2)
    endpoint.updated = endpoint.time_created

    inventory = build_cutoff_inventory(target, cfg, storage_client=Client(objects))

    assert inventory["era5"]["ready"] is True
    assert inventory["era5"]["selectedThroughDate"] == "2026-08-12"
    assert inventory["era5"]["exactAvailable"] is True
    assert inventory["era5"]["exactArrivedAfterCutoff"] is True


def test_open_month_uses_actual_derived_cache_coverage(tmp_path):
    target = date(2026, 8, 18)
    cfg, objects = ready_objects(target)
    del objects["era5/2026/era5_2026_08_12.nc"]
    del objects["era5/2026/era5_2026_08_11.nc"]
    cfg["paths"]["local_cache"] = str(tmp_path)
    cache = tmp_path / "m4_shared_cache" / "era5_daily" / "year=2026"
    cache.mkdir(parents=True)
    derived = cache / "month=08.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2026-08-11")]}).to_parquet(derived)
    before_cutoff = forecast_cutoff_at(target, cfg).timestamp() - 60
    os.utime(derived, (before_cutoff, before_cutoff))

    inventory = build_cutoff_inventory(target, cfg, storage_client=Client(objects))

    assert inventory["era5"]["ready"] is True
    assert inventory["era5"]["selectedThroughDate"] == "2026-08-11"
    assert inventory["era5"]["exactAvailable"] is False


def test_sentinel5p_rejects_observation_older_than_seven_days():
    target = date(2026, 8, 18)
    cfg, objects = ready_objects(target, s5p_age=8)
    inventory = build_cutoff_inventory(target, cfg, storage_client=Client(objects))
    assert inventory["sentinel5p"]["ready"] is False
    assert "within 7 days" in inventory["sentinel5p"]["message"]
