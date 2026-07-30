from __future__ import annotations

import copy

import pandas as pd

from numerical_nextday.config import config_hash, load_config
from numerical_nextday.data.assemble import attach_eo_year, build_stage_a_year
from numerical_nextday.data.synthetic import build_synthetic_inputs
from numerical_nextday.dataset import load_splits, write_splits


def test_synthetic_point_in_time_pipeline(tmp_path) -> None:
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["workspace"] = tmp_path
    cfg["paths"]["dem_cells"] = tmp_path / "data" / "dem.parquet"
    cfg["paths"]["cache_dir"] = tmp_path / "cache"
    cfg["paths"]["dataset_dir"] = tmp_path / "datasets"
    cfg["paths"]["artifact_dir"] = tmp_path / "artifacts"
    cfg["paths"]["manifest_dir"] = tmp_path / "manifests"
    cfg["paths"]["report_dir"] = tmp_path / "reports"
    cfg["_config_hash"] = config_hash(cfg)
    build_synthetic_inputs(cfg, cells=6, seed=9)
    for year in range(2019, 2026):
        build_stage_a_year(cfg, year, 5, force=True)
        attach_eo_year(cfg, "s2", year, 5, force=True)
        attach_eo_year(cfg, "s5p", year, 5, force=True)
    write_splits(cfg, "C", 5, force=True)
    splits = load_splits(cfg, "C", 5)
    assert set(splits) == {"train", "tune", "calibration", "test"}
    assert set(splits["train"]["label_date"].dt.year) == {2019, 2020, 2021, 2022}
    missing_2021 = splits["train"].loc[
        splits["train"]["label_date"].dt.year == 2021, "s5p_data_available"
    ]
    assert missing_2021.eq(0).all()
    missing_2021_values = splits["train"].loc[
        splits["train"]["label_date"].dt.year == 2021, "s5p_aai_mean"
    ]
    assert missing_2021_values.eq(0).all()
    assert (
        splits["test"]["era5_source_date"]
        == splits["test"]["feature_end_date"] - pd.Timedelta(days=5)
    ).all()
