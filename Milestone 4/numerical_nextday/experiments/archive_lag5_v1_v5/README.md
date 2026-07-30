# Archive lag-5 experiments (V1–V5)

This directory contains an independent experiment track for the numerical
next-day wildfire task. It is intentionally isolated from the implementation
in `Milestone 4/numerical_nextday` so both approaches can be reviewed and run
without overwriting one another.

## Forecasting contract

- Target: `y_fire`, qualifying FIRMS activity on **D+1**
- ERA5 cutoff: **D−5**
- Fire-history cutoff in V2–V5: **D−1**, conditional on D−1 FIRMS being
  available when the forecast is issued on D
- Split: train **2019–2022**, validation **2023–2024**, test **2025**
- 2021 Sentinel-5P: completely missing and represented by unavailable
  placeholders

No input Parquet files, downloaded GCS data, trained model bundles, prediction
tables, virtual environments, or caches are stored in Git.

## Experiment progression

| Version | Main change | 2025 Recall@25 |
|---|---|---:|
| V1 | Stage A/B/C LightGBM baseline; MLP comparison | 25.27% |
| V2 | Causal weather and fire-history features | 36.79% |
| V3 | Direction- and distance-aware spread features | 37.85% |
| V4 | Recall@25 tuning and classifier/ranker blend | **38.33%** |
| V5 | Top-75 hard-negative reranking | 38.37% |

V4 is the retained model. V5 captures only one additional 2025 positive at
K=25 and loses recall at K=50, so the extra complexity is not justified.
See [the consolidated report](reports/EXPERIMENT_REPORT.md) for architectures,
parameters, validation results, limitations, and the leakage assessment.

## Environment

Python 3.10–3.13 is supported.

```bash
cd "Milestone 4/numerical_nextday/experiments/archive_lag5_v1_v5"
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

The `.venv` directory is ignored and must remain local.

## Expected local data

Place the supplied archive outside version control:

```text
local_data/
  archive/
    meta.json
    stage_a/
    stage_b/
    stage_c/
      all.parquet
      train.parquet
      val.parquet
      test.parquet
      metadata/
        feature_columns.json
        dataset_metadata.json
```

The exact dataset contract, row counts, checksums, and known data-quality
exceptions are recorded in
[`reports/dataset_meta.json`](reports/dataset_meta.json) and
[`reports/DATA_AUDIT.md`](reports/DATA_AUDIT.md).

## Verify the archive

```bash
python scripts/audit_archive.py local_data/archive
```

The audited archive contains 1,718,304 rows across 672 grid cells and 2,557
days. Its label prevalence is 1.258%.

## Reproduce the experiments

All commands below use local files and do not repeat GCS downloads.

### V1 — baseline

```bash
python scripts/train_archive.py \
  --archive local_data/archive \
  --output local_artifacts/archive_training/lag5_full_year
```

### V2 — leakage-safe causal histories

```bash
python scripts/build_archive_v2.py \
  --archive local_data/archive \
  --output local_data/archive_v2

python scripts/train_archive_v2.py \
  --data local_data/archive_v2 \
  --output local_artifacts/archive_training/lag5_v2 \
  --baseline-metrics local_artifacts/archive_training/lag5_full_year/metrics.json
```

### V3 — directional spread

```bash
python scripts/build_archive_v3.py \
  --v2-data local_data/archive_v2 \
  --output local_data/archive_v3

python scripts/train_archive_v3.py \
  --data local_data/archive_v3 \
  --output local_artifacts/archive_training/lag5_v3 \
  --v1-metrics local_artifacts/archive_training/lag5_full_year/metrics.json \
  --v2-metrics local_artifacts/archive_training/lag5_v2/metrics.json
```

### V4 — Recall@25 tuning

```bash
python scripts/train_archive_v4.py \
  --data local_data/archive_v3 \
  --output local_artifacts/archive_training/lag5_v4_recall25 \
  --v3-metrics local_artifacts/archive_training/lag5_v3/metrics.json
```

### V5 — two-stage reranking

```bash
python scripts/train_archive_v5.py \
  --data local_data/archive_v3 \
  --output local_artifacts/archive_training/lag5_v5_two_stage \
  --v4-output local_artifacts/archive_training/lag5_v4_recall25
```

### Rebuild the consolidated report

```bash
python scripts/build_experiment_report.py \
  --archive-meta local_data/archive/meta.json \
  --artifact-root local_artifacts/archive_training
```

## Tests

The tests use temporary or synthetic data and do not need the archive:

```bash
PYTHONPATH=src python -m pytest -q
```

The checked-in version passes 30 tests.

## Leakage interpretation

The feature builders exclude direct outcomes and enforce the D−5 weather
cutoff. Fire histories end at D−1, never D or D+1. The missing 2021 S5P block
is a distribution-shift/year-indicator risk, not target leakage.

Only V1's initial 2025 evaluation was fully untouched across the experiment
series. V2–V5 were designed after earlier 2025 results had been inspected.
Their 2025 scores are therefore descriptive and must not be presented as fresh
holdout evidence. A future 2026 season or external geography should be frozen
as the next independent test.
