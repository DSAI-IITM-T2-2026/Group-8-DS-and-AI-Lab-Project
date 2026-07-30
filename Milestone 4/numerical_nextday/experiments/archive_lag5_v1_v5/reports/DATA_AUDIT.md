# Wildfire Training-Data Audit

**Audited:** 30 July 2026

**Primary table:** Stage C

**Verdict:** **Conditionally ready**

The archive is structurally suitable for direct LightGBM, MLP, or custom
tabular-model training without repeating GCS downloads. Before presenting a
strict production result, resolve or explicitly accept the freshness and
schema-compatibility warnings below.

## Verified contract

Use `eo_asof_date` as forecast day **D**:

- `label_date = eo_asof_date + 1 day` for every row.
- `feature_end_date = eo_asof_date - 5 days` for every row.
- `era5_lag_days = 5` for every row.

Therefore: **`y_fire` is the FIRMS outcome on D+1; ERA5 predictors end at
D−5.**

The name `feature_end_date` is potentially misleading in this archive: it
stores the ERA5 source day D−5, not forecast day D.

## Split and coverage audit

| Split | Label years | Rows | Positive rows | Positive rate |
|---|---:|---:|---:|---:|
| Train | 2019–2022 | 981,792 | 13,804 | 1.406% |
| Validation | 2023–2024 | 491,232 | 5,536 | 1.127% |
| Test | 2025 | 245,280 | 2,275 | 0.928% |
| **Total** | **2019–2025** | **1,718,304** | **21,615** | **1.258%** |

All 2,557 calendar days are present. Every day has exactly 672 unique cells.
There are no duplicate `(cell_id, label_date)` or `(cell_id,
feature_end_date)` keys, no coordinate conflicts, and no overlap between the
chronological splits.

The Stage A, B, and C tables contain the same keys and labels. Their `all`
tables match the union of train, validation, and test. Metadata row and
positive counts agree with the parquet contents.

## Labels and leakage

- `y_fire` is binary with values 0 and 1 only.
- Every positive label has at least one FIRMS pixel and maximum confidence at
  least 30.
- There are no mismatches between `y_fire` and `firms_n_pixels > 0`.
- The published feature allowlists contain no identifiers, dates, labels, or
  FIRMS outcome fields.

**Important:** `firms_n_pixels` and `firms_max_confidence` are physically
present in each parquet and directly encode the outcome. A custom trainer must
use the exact allowlist in `stage_c/metadata/feature_columns.json`; selecting
all numeric columns except `y_fire` would cause severe target leakage.

## 2021 S5P handling

The missing 2021 S5P year is handled consistently:

- 245,280 rows are retained.
- `s5n_available = 0` on every 2021 row.
- `s5n_s5p_data_available = 0`.
- All eight S5P measurement values are finite zero placeholders.
- `s5p_2021_status = "placeholder"`.

This is usable by LightGBM and MLP because the availability indicator remains
in the feature allowlist. It also creates a year-specific missing-data regime,
so report Stage B (without S5P) alongside Stage C to determine whether results
depend on the placeholder pattern.

## Feature-quality audit

- Stage C exposes 63 allowed features; every declared feature exists and is
  numeric.
- No Stage C feature contains NaN or infinity.
- Basic temperature, dewpoint, humidity, precipitation, wind, cloud, and
  coverage bounds pass.
- `s5n_s5p_aai_std` and `s5n_s5p_co_std` are constant zero in all splits and
  add no predictive information.
- Small negative soil values occur: 46,002 `swvl1_mean` rows, 45,232
  `swvl2_mean` rows, and 21,039 `soil_moisture_index` rows. The minima are
  approximately −0.0072, −0.0033, and −0.0029. Clip to zero or document the
  ERA5 numerical artifact.

## Freshness warnings

These rows conflict with the limits documented elsewhere in the project:

1. **S2:** 7,392 test rows, representing every cell on 11 days from
   2025-12-21 through 2025-12-31, are marked available with lag 16–26 days.
   The documented maximum is 15 days.
2. **S5P:** 6,720 training rows, representing every cell on 10 days during
   2020, are marked available with lag 3–7 days. The documented maximum is 2
   days.
3. **S2 missing-value encoding:** 3,360 training rows covering 2019-01-01
   through 2019-01-05 have `s2n_available=0` but retain nonzero S2 values.

These are stale past observations, not future-data leakage. They should
nevertheless be corrected or explicitly accepted before a strict
real-time-compatible benchmark is reported.

## Compatibility warning

The tables are ready for a standalone trainer that reads the supplied feature
allowlist. They are **not plug-compatible with the repository's current
`numerical_nextday` runner**:

- The runner treats `feature_end_date` as D, while the archive uses it for
  D−5 and uses `eo_asof_date` for D.
- The runner expects an `era5_source_date` column, which the archive does not
  contain.
- The runner expects `train`, `tune`, `calibration`, and `test`; the archive
  supplies `train`, combined `val`, and `test`.

For the existing training design, divide validation by label year: use 2023
for tuning/early stopping and 2024 for calibration. Do not use 2025 for model
selection, preprocessing, threshold choice, or calibration.

## Readiness decision

- **Go:** exploratory or standalone LightGBM/MLP training using the exact
  feature allowlist and chronological split.
- **Hold:** a strict production claim until the freshness exceptions and
  column-contract mismatch are repaired or formally accepted.

Checksums and the machine-readable training contract are in `meta.json`. The
reproducible audit program is
`Milestone 4/numerical_nextday/scripts/audit_archive.py`.
