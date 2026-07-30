from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SPLITS = ("train", "val", "test")
STAGES = ("stage_a", "stage_b", "stage_c")
KEY_AUDIT_COLUMNS = [
    "feature_end_date",
    "label_date",
    "eo_asof_date",
    "cell_id",
    "latitude",
    "longitude",
    "firms_n_pixels",
    "firms_max_confidence",
    "y_fire",
    "era5_lag_days",
]
FORBIDDEN_FEATURE_COLUMNS = {
    "feature_end_date",
    "label_date",
    "eo_asof_date",
    "cell_id",
    "latitude",
    "longitude",
    "region",
    "firms_n_pixels",
    "firms_max_confidence",
    "y_fire",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_signature(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(
        dtype=np.uint64
    )
    return {
        "count": int(values.size),
        "sum_mod_2_64": int(values.sum(dtype=np.uint64)),
        "xor": int(np.bitwise_xor.reduce(values)) if values.size else 0,
    }


def merge_signatures(signatures: list[dict[str, int]]) -> dict[str, int]:
    total = 0
    xor = 0
    count = 0
    for signature in signatures:
        count += signature["count"]
        total = (total + signature["sum_mod_2_64"]) % (1 << 64)
        xor ^= signature["xor"]
    return {"count": count, "sum_mod_2_64": total, "xor": xor}


def unique_ints(series: pd.Series) -> list[int]:
    return sorted(int(value) for value in series.dropna().unique())


def audit_keys(path: Path) -> tuple[dict, dict[str, int]]:
    frame = pd.read_parquet(path, columns=KEY_AUDIT_COLUMNS)
    for column in ("feature_end_date", "label_date", "eo_asof_date"):
        frame[column] = pd.to_datetime(frame[column])
    label_days = frame["label_date"].dt.normalize()
    feature_days = frame["feature_end_date"].dt.normalize()
    eo_days = frame["eo_asof_date"].dt.normalize()
    daily_rows = frame.groupby(label_days, sort=True).size()
    daily_cells = frame.groupby(label_days, sort=True)["cell_id"].nunique()
    per_cell_coords = frame.groupby("cell_id")[["latitude", "longitude"]].nunique()
    positive = frame["y_fire"] == 1
    expected_positive = frame["firms_n_pixels"].fillna(0) > 0
    confidence_violation = positive & (
        frame["firms_max_confidence"].isna()
        | (frame["firms_max_confidence"] < 30)
    )
    result = {
        "rows": int(len(frame)),
        "columns_loaded": KEY_AUDIT_COLUMNS,
        "label_min": label_days.min().date().isoformat(),
        "label_max": label_days.max().date().isoformat(),
        "label_years": unique_ints(label_days.dt.year),
        "feature_min": feature_days.min().date().isoformat(),
        "feature_max": feature_days.max().date().isoformat(),
        "eo_asof_min": eo_days.min().date().isoformat(),
        "eo_asof_max": eo_days.max().date().isoformat(),
        "unique_cells": int(frame["cell_id"].nunique()),
        "unique_label_days": int(label_days.nunique()),
        "daily_rows_min": int(daily_rows.min()),
        "daily_rows_max": int(daily_rows.max()),
        "daily_unique_cells_min": int(daily_cells.min()),
        "daily_unique_cells_max": int(daily_cells.max()),
        "duplicate_cell_label_keys": int(
            frame.duplicated(["cell_id", "label_date"]).sum()
        ),
        "duplicate_cell_feature_keys": int(
            frame.duplicated(["cell_id", "feature_end_date"]).sum()
        ),
        "cells_with_inconsistent_coordinates": int(
            ((per_cell_coords["latitude"] > 1) | (per_cell_coords["longitude"] > 1)).sum()
        ),
        "label_minus_eo_asof_days": unique_ints((label_days - eo_days).dt.days),
        "eo_asof_minus_feature_days": unique_ints((eo_days - feature_days).dt.days),
        "label_minus_feature_days": unique_ints((label_days - feature_days).dt.days),
        "era5_lag_days_values": unique_ints(frame["era5_lag_days"]),
        "y_values": unique_ints(frame["y_fire"]),
        "positives": int(positive.sum()),
        "positive_rate": float(positive.mean()),
        "firms_label_mismatch_rows": int((positive != expected_positive).sum()),
        "positive_confidence_below_30_or_null_rows": int(confidence_violation.sum()),
        "negative_firms_pixel_count_rows": int((frame["firms_n_pixels"] < 0).sum()),
        "nulls_in_key_fields": {
            column: int(frame[column].isna().sum())
            for column in (
                "feature_end_date",
                "label_date",
                "eo_asof_date",
                "cell_id",
                "latitude",
                "longitude",
                "y_fire",
                "era5_lag_days",
            )
        },
    }
    signature = hash_signature(
        frame,
        [
            "cell_id",
            "feature_end_date",
            "label_date",
            "eo_asof_date",
            "y_fire",
        ],
    )
    return result, signature


def audit_features(path: Path, feature_columns: list[str]) -> dict:
    nulls = Counter({column: 0 for column in feature_columns})
    infs = Counter({column: 0 for column in feature_columns})
    mins = {column: np.inf for column in feature_columns}
    maxs = {column: -np.inf for column in feature_columns}
    non_null = Counter({column: 0 for column in feature_columns})
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=feature_columns, batch_size=65_536):
        frame = batch.to_pandas()
        for column in feature_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
            finite = np.isfinite(values)
            nulls[column] += int(np.isnan(values).sum())
            infs[column] += int(np.isinf(values).sum())
            non_null[column] += int(finite.sum())
            if finite.any():
                mins[column] = min(mins[column], float(values[finite].min()))
                maxs[column] = max(maxs[column], float(values[finite].max()))
    rows = parquet.metadata.num_rows
    constant = [
        column
        for column in feature_columns
        if non_null[column] and mins[column] == maxs[column]
    ]
    entirely_missing = [column for column in feature_columns if not non_null[column]]
    return {
        "rows": rows,
        "null_count": dict(nulls),
        "null_rate": {column: nulls[column] / rows for column in feature_columns},
        "infinite_count": dict(infs),
        "min": {
            column: None if not non_null[column] else mins[column]
            for column in feature_columns
        },
        "max": {
            column: None if not non_null[column] else maxs[column]
            for column in feature_columns
        },
        "constant_features": constant,
        "entirely_missing_features": entirely_missing,
    }


def audit_stage_c_availability(path: Path) -> dict:
    s2_measurements = [
        "s2n_B2_mean",
        "s2n_B3_mean",
        "s2n_B4_mean",
        "s2n_B8_mean",
        "s2n_B11_mean",
        "s2n_B12_mean",
        "s2n_B2_std",
        "s2n_B3_std",
        "s2n_B4_std",
        "s2n_B8_std",
        "s2n_B11_std",
        "s2n_B12_std",
        "s2n_NDVI_mean",
        "s2n_NDMI_mean",
        "s2n_NBR_mean",
        "s2n_NDWI_mean",
        "s2n_EVI_mean",
        "s2n_cloud_percentage",
        "s2n_valid_fraction",
    ]
    s5_measurements = [
        "s5n_s5p_aai_mean",
        "s5n_s5p_aai_max",
        "s5n_s5p_aai_std",
        "s5n_s5p_aai_valid_fraction",
        "s5n_s5p_co_mean",
        "s5n_s5p_co_max",
        "s5n_s5p_co_std",
        "s5n_s5p_co_valid_fraction",
        "s5n_s5p_data_available",
    ]
    columns = [
        "label_date",
        "y_fire",
        "s2n_available",
        "s2n_lag_days",
        "s5n_available",
        "s5n_lag_days",
        "s5p_2021_status",
        *s2_measurements,
        *s5_measurements,
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["label_date"] = pd.to_datetime(frame["label_date"])
    frame["year"] = frame["label_date"].dt.year
    by_year = {}
    for year, part in frame.groupby("year", sort=True):
        s2_available = part["s2n_available"] == 1
        s5_available = part["s5n_available"] == 1
        by_year[str(int(year))] = {
            "rows": int(len(part)),
            "s2_available_rows": int(s2_available.sum()),
            "s2_available_rate": float(s2_available.mean()),
            "s2_lag_min_available": (
                float(part.loc[s2_available, "s2n_lag_days"].min())
                if s2_available.any()
                else None
            ),
            "s2_lag_max_available": (
                float(part.loc[s2_available, "s2n_lag_days"].max())
                if s2_available.any()
                else None
            ),
            "s5_available_rows": int(s5_available.sum()),
            "s5_available_rate": float(s5_available.mean()),
            "s5_data_available_sum": float(part["s5n_s5p_data_available"].sum()),
            "s5_lag_min_available": (
                float(part.loc[s5_available, "s5n_lag_days"].min())
                if s5_available.any()
                else None
            ),
            "s5_lag_max_available": (
                float(part.loc[s5_available, "s5n_lag_days"].max())
                if s5_available.any()
                else None
            ),
            "s5p_2021_status_values": sorted(
                str(value) for value in part["s5p_2021_status"].dropna().unique()
            ),
        }
    year_2021 = frame.loc[frame["year"] == 2021]
    s5_values_2021 = year_2021[s5_measurements[:-1]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_2021 = np.isfinite(s5_values_2021.to_numpy(dtype="float64"))
    s2_unavailable = frame["s2n_available"] == 0
    s5_unavailable = frame["s5n_available"] == 0
    s2_unavailable_values = frame.loc[s2_unavailable, s2_measurements].to_numpy(
        dtype="float64"
    )
    s5_unavailable_values = frame.loc[s5_unavailable, s5_measurements].to_numpy(
        dtype="float64"
    )

    def freshness(mask: pd.Series) -> dict:
        part = frame.loc[mask, ["label_date", "year"]]
        return {
            "rows": int(len(part)),
            "days": int(part["label_date"].nunique()),
            "cells_per_affected_day": (
                int(len(part) / part["label_date"].nunique()) if len(part) else 0
            ),
            "date_min": (
                part["label_date"].min().date().isoformat() if len(part) else None
            ),
            "date_max": (
                part["label_date"].max().date().isoformat() if len(part) else None
            ),
            "rows_by_year": {
                str(int(year)): int(count)
                for year, count in part.groupby("year").size().items()
            },
        }

    label_by_year = {}
    for year, part in frame.groupby("year", sort=True):
        positives = int(part["y_fire"].sum())
        label_by_year[str(int(year))] = {
            "rows": int(len(part)),
            "positives": positives,
            "positive_rate": positives / len(part),
        }
    return {
        "by_year": by_year,
        "label_by_year": label_by_year,
        "s2_available_values": unique_ints(frame["s2n_available"]),
        "s5_available_values": unique_ints(frame["s5n_available"]),
        "s5_data_available_min": float(frame["s5n_s5p_data_available"].min()),
        "s5_data_available_max": float(frame["s5n_s5p_data_available"].max()),
        "s2_available_with_null_lag": int(
            ((frame["s2n_available"] == 1) & frame["s2n_lag_days"].isna()).sum()
        ),
        "s2_unavailable_with_nonnull_lag": int(
            ((frame["s2n_available"] == 0) & frame["s2n_lag_days"].notna()).sum()
        ),
        "s5_available_with_null_lag": int(
            ((frame["s5n_available"] == 1) & frame["s5n_lag_days"].isna()).sum()
        ),
        "s5_unavailable_with_nonnull_lag": int(
            ((frame["s5n_available"] == 0) & frame["s5n_lag_days"].notna()).sum()
        ),
        "freshness_contract_exceptions": {
            "s2_available_with_lag_over_15_days": freshness(
                (frame["s2n_available"] == 1) & (frame["s2n_lag_days"] > 15)
            ),
            "s5_available_with_lag_over_2_days": freshness(
                (frame["s5n_available"] == 1) & (frame["s5n_lag_days"] > 2)
            ),
        },
        "unavailable_value_encoding": {
            "s2_unavailable_rows": int(s2_unavailable.sum()),
            "s2_unavailable_rows_with_any_nonzero_measurement": int(
                (
                    np.isfinite(s2_unavailable_values)
                    & (s2_unavailable_values != 0)
                )
                .any(axis=1)
                .sum()
            ),
            "s5_unavailable_rows": int(s5_unavailable.sum()),
            "s5_unavailable_rows_with_any_nonzero_measurement": int(
                (
                    np.isfinite(s5_unavailable_values)
                    & (s5_unavailable_values != 0)
                )
                .any(axis=1)
                .sum()
            ),
        },
        "year_2021": {
            "rows": int(len(year_2021)),
            "s5_available_sum": int(year_2021["s5n_available"].sum()),
            "s5_data_available_sum": float(
                year_2021["s5n_s5p_data_available"].sum()
            ),
            "status_values": sorted(
                str(value)
                for value in year_2021["s5p_2021_status"].dropna().unique()
            ),
            "measurement_finite_values": int(finite_2021.sum()),
            "measurement_null_values": int((~finite_2021).sum()),
            "measurement_nonzero_finite_values": int(
                (
                    finite_2021
                    & (s5_values_2021.to_numpy(dtype="float64") != 0)
                ).sum()
            ),
        },
    }


def audit_physical_ranges(path: Path) -> dict[str, int]:
    columns = [
        "t2m_min",
        "t2m_mean",
        "t2m_max",
        "d2m_mean",
        "rh_mean",
        "tp_sum_mm",
        "wind_speed_mean",
        "swvl1_mean",
        "swvl2_mean",
        "soil_moisture_index",
        "s2n_valid_fraction",
        "s2n_cloud_percentage",
        "s5n_s5p_aai_valid_fraction",
        "s5n_s5p_co_valid_fraction",
    ]
    frame = pd.read_parquet(path, columns=columns)
    return {
        "t2m_min_above_mean_rows": int((frame["t2m_min"] > frame["t2m_mean"]).sum()),
        "t2m_mean_above_max_rows": int((frame["t2m_mean"] > frame["t2m_max"]).sum()),
        "dewpoint_above_temperature_rows": int(
            (frame["d2m_mean"] > frame["t2m_mean"]).sum()
        ),
        "relative_humidity_outside_0_100_rows": int(
            ((frame["rh_mean"] < 0) | (frame["rh_mean"] > 100)).sum()
        ),
        "negative_precipitation_rows": int((frame["tp_sum_mm"] < 0).sum()),
        "negative_wind_speed_rows": int((frame["wind_speed_mean"] < 0).sum()),
        "negative_swvl1_rows": int((frame["swvl1_mean"] < 0).sum()),
        "negative_swvl2_rows": int((frame["swvl2_mean"] < 0).sum()),
        "soil_moisture_index_outside_0_1_rows": int(
            (
                (frame["soil_moisture_index"] < 0)
                | (frame["soil_moisture_index"] > 1)
            ).sum()
        ),
        "s2_valid_fraction_outside_0_1_rows": int(
            (
                (frame["s2n_valid_fraction"] < 0)
                | (frame["s2n_valid_fraction"] > 1)
            ).sum()
        ),
        "s2_cloud_percentage_outside_0_100_rows": int(
            (
                (frame["s2n_cloud_percentage"] < 0)
                | (frame["s2n_cloud_percentage"] > 100)
            ).sum()
        ),
        "s5_aai_valid_fraction_outside_0_1_rows": int(
            (
                (frame["s5n_s5p_aai_valid_fraction"] < 0)
                | (frame["s5n_s5p_aai_valid_fraction"] > 1)
            ).sum()
        ),
        "s5_co_valid_fraction_outside_0_1_rows": int(
            (
                (frame["s5n_s5p_co_valid_fraction"] < 0)
                | (frame["s5n_s5p_co_valid_fraction"] > 1)
            ).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    root = args.archive.resolve()

    report: dict = {
        "archive": str(root),
        "stages": {},
        "checks": {},
    }
    stage_signatures: dict[str, dict[str, dict[str, int]]] = {}
    for stage in STAGES:
        stage_dir = root / stage
        feature_columns = json.loads(
            (stage_dir / "metadata" / "feature_columns.json").read_text()
        )
        metadata = json.loads(
            (stage_dir / "metadata" / "dataset_metadata.json").read_text()
        )
        schema_strings = {}
        key_audits = {}
        signatures = {}
        for split in (*SPLITS, "all"):
            path = stage_dir / f"{split}.parquet"
            parquet = pq.ParquetFile(path)
            schema_strings[split] = str(parquet.schema_arrow)
            key_audits[split], signatures[split] = audit_keys(path)
        split_union_signature = merge_signatures(
            [signatures[split] for split in SPLITS]
        )
        report["stages"][stage] = {
            "metadata": metadata,
            "feature_count": len(feature_columns),
            "feature_columns_missing_from_table": sorted(
                set(feature_columns)
                - set(pq.ParquetFile(stage_dir / "train.parquet").schema_arrow.names)
            ),
            "forbidden_columns_in_feature_list": sorted(
                set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
            ),
            "schemas_identical_across_files": len(set(schema_strings.values())) == 1,
            "key_audit": key_audits,
            "all_matches_split_union_signature": (
                split_union_signature == signatures["all"]
            ),
            "metadata_rows_match": (
                metadata["n_rows"]
                == sum(key_audits[split]["rows"] for split in SPLITS)
                == key_audits["all"]["rows"]
            ),
            "metadata_positives_match": (
                metadata["n_pos"]
                == sum(key_audits[split]["positives"] for split in SPLITS)
                == key_audits["all"]["positives"]
            ),
        }
        stage_signatures[stage] = signatures

    stage_c_features = json.loads(
        (root / "stage_c" / "metadata" / "feature_columns.json").read_text()
    )
    report["stage_c_feature_audit"] = {
        split: audit_features(
            root / "stage_c" / f"{split}.parquet", stage_c_features
        )
        for split in SPLITS
    }
    report["stage_c_availability"] = audit_stage_c_availability(
        root / "stage_c" / "all.parquet"
    )
    report["stage_c_physical_ranges"] = audit_physical_ranges(
        root / "stage_c" / "all.parquet"
    )
    report["checks"]["cross_stage_key_label_identity"] = {
        split: len(
            {
                (
                    stage_signatures[stage][split]["count"],
                    stage_signatures[stage][split]["sum_mod_2_64"],
                    stage_signatures[stage][split]["xor"],
                )
                for stage in STAGES
            }
        )
        == 1
        for split in (*SPLITS, "all")
    }
    report["checks"]["stage_c_split_sha256"] = {
        split: sha256(root / "stage_c" / f"{split}.parquet")
        for split in SPLITS
    }
    report["checks"]["stage_c_metadata_sha256"] = {
        "feature_columns.json": sha256(
            root / "stage_c" / "metadata" / "feature_columns.json"
        ),
        "dataset_metadata.json": sha256(
            root / "stage_c" / "metadata" / "dataset_metadata.json"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
