from __future__ import annotations

import copy

import pytest

from numerical_nextday.cli import parse_int_set
from numerical_nextday.config import bucket_for_month, load_config, validate_config


def test_parse_int_set_and_month_router() -> None:
    cfg = load_config()
    assert parse_int_set("2019-2021,2025") == [2019, 2020, 2021, 2025]
    assert bucket_for_month(cfg, 1) == "jan"
    assert bucket_for_month(cfg, 4) == "fire_season"
    assert bucket_for_month(cfg, 12) == "dec"


def test_config_rejects_overlapping_splits() -> None:
    cfg = load_config()
    broken = copy.deepcopy(cfg)
    broken["temporal"]["test_years"] = [2024]
    with pytest.raises(ValueError, match="overlap"):
        validate_config(broken)


def test_config_rejects_invalid_single_machine_parallelism() -> None:
    cfg = load_config()
    broken = copy.deepcopy(cfg)
    broken["execution"]["eo_parallel_streams"] = 0
    with pytest.raises(ValueError, match="eo_parallel_streams"):
        validate_config(broken)
