from __future__ import annotations

from pathlib import Path
import sys


UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from download.sentinel2 import build_runtime_config  # noqa: E402


def test_runtime_config_uses_vendored_loader_signature_and_applies_overrides():
    config = build_runtime_config(
        project_id="test-earth-engine-project",
        bucket="test-bucket",
        prefix="test-sentinel2",
        grid_asset_id="projects/test/assets/california-grid",
        year=2026,
    )

    assert config.project_id == "test-earth-engine-project"
    assert config.export.bucket == "test-bucket"
    assert config.export.prefix == "test-sentinel2"
    assert config.grid.asset_id == "projects/test/assets/california-grid"
    assert config.temporal.start_year == 2018
    assert config.temporal.end_year == 2026
