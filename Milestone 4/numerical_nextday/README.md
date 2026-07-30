# Milestone 4 — Numerical next-day wildfire prediction

Plan: [`docs/MILESTONE4_PLAN.md`](../../docs/MILESTONE4_PLAN.md)

**Task:** predict fire on **D+1** using ERA5 through **D−5** (`era5_lag_days=5`) + DEM + causal S2/S5P.  
**Split:** train 2019–2022 / val 2023–2024 / test 2025.  
**Models:** `fire_season` (Apr–Nov) + `jan` / `feb` / `mar` / `dec`.

## Additional experiment track

The independent archive-based V1–V5 experiments are available under
[`experiments/archive_lag5_v1_v5`](experiments/archive_lag5_v1_v5/README.md).
They are isolated from this implementation and include source code, tests, and
lightweight result reports only—no downloaded input data or trained models.

---

## Setup

```bash
cd "Milestone 4/numerical_nextday"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
export GS_NO_SIGN_REQUEST=YES
```

CLI: `python scripts/run_pipeline.py run --stage <name> ...`  
(or `python -m numerical_nextday run --stage <name> ...`)

---

## Smoke (one month, lag-5 check)

```bash
python scripts/run_pipeline.py run --stage era5_firms --years 2024 --months 8 --worker local --era5-lag-days 5
python scripts/run_pipeline.py run --stage stage_a_year --years 2024 --months 8 --worker local --era5-lag-days 5
# Expect label_date - feature_end_date == 6 days on outputs/m4_shared_cache/stage_a/year=2024.parquet
```

---

## Complete data build (solo, overnight)

One command (default years **2019–2025**, months **1–12**, lag **5**):

```bash
python scripts/run_pipeline.py run --stage build_data --years 2019-2025 --months 1-12 --worker local --era5-lag-days 5
```

Equivalent steps:

```bash
python scripts/run_pipeline.py run --stage era5_firms --years 2019-2025 --months 1-12 --worker local
python scripts/run_pipeline.py run --stage stage_a_year --years 2019-2025 --worker local --era5-lag-days 5
python scripts/run_pipeline.py run --stage merge_stage_a --years 2019-2025
python scripts/run_pipeline.py run --stage s2_cell_cache --years 2019-2025 --months 1-12 --worker local
python scripts/run_pipeline.py run --stage s2_attach --years 2019-2025
python scripts/run_pipeline.py run --stage s5p_cell_cache --years 2019-2025 --months 1-12 --worker local
python scripts/run_pipeline.py run --stage s5p_attach --years 2019-2025
```

Notes:

- **Resume-safe:** finished months/years are skipped via `claim_lock.jsonl` (re-run after interrupt).
- **S5P 2021:** skipped while `s5p_2021_mode: placeholder` in `config.yaml` (zeros + `s5n_available=0`). When ready, set `ready` and re-run `s5p_cell_cache` / `s5p_attach` for `--years 2021`.
- Multi-laptop shards are **optional**; solo uses the same stages with one `--worker`.

---

## Complete train / eval

After Stage A/B/C parquets exist:

```bash
python scripts/run_pipeline.py run --stage train_all
```

Steps inside `train_all`:

1. `train_fire_season` — Stage A/B/C defaults + ≤12 LGBM HP + MLP on fire_season  
2. `train_month_models` — jan/feb/mar/dec (reuse best fire_season HPs; winter_fallback if too few positives)  
3. `eval_figures` — per-bucket metrics, calibration, sample risk map, top-k alerts  

Outputs: `artifacts/` (`experiments_log.csv`, `eval_metrics.json`, `models/`, `figures/`).

---

## One-shot (data + train)

```bash
# shell helper
bash scripts/run_full_pipeline.sh

# or
python scripts/run_pipeline.py run --stage run_all --years 2019-2025 --months 1-12 --worker local
```

---

## Lag-0 oracle ablation (optional)

```bash
python scripts/run_pipeline.py run --stage build_lag0_data --years 2019-2025 --months 1-12 --worker local
python scripts/run_pipeline.py run --stage train_lag0_ablation
```

Writes under `outputs/m4_shared_cache/lag0/` and logs `lgbm_era5_lag0` in `experiments_log.csv`.

---

## Dry-run (no GCS — not for submission metrics)

```bash
python scripts/run_pipeline.py run --stage synthetic_smoke
python scripts/run_pipeline.py run --stage train_all
# optional quick skip: M4_SKIP_MLP=1 python scripts/run_pipeline.py run --stage train_all
```

---

## Stages

| Stage | Purpose |
|-------|---------|
| `verify_gcs` | List GCS prefixes |
| `era5_firms` | Monthly ERA5 + FIRMS caches |
| `stage_a_year` | Lag-aware Stage A year parquet |
| `merge_stage_a` | Concat + train/val/test + metadata |
| `s2_cell_cache` / `s5p_cell_cache` | EO → ERA5 cell means |
| `s2_attach` / `s5p_attach` | Stage B / C (causal `window_end ≤ D`) |
| **`build_data`** | Solo full data build (lag-5) |
| **`build_lag0_data`** | Lag-0 Stage A→C under `lag0/` |
| `synthetic_smoke` | Tiny synthetic tables |
| `train_fire_season` / `train_month_models` / `eval_figures` | Training pieces |
| `train_lag0_ablation` | Oracle LGBM on lag0 Stage C |
| **`train_all`** | Full train + eval |
| **`run_all`** | `build_data` → `train_all` |

---

## Maps & metric charts

**Teammate-style CA outline maps** (Confidence % + blue FIRMS rings):

```bash
# all ~365 test days → artifacts/maps/daily/risk_YYYY-MM-DD.png  (gitignored)
python scripts/generate_test_risk_maps.py

# one day / smoke
python scripts/generate_test_risk_maps.py --date 2025-10-21
python scripts/generate_test_risk_maps.py --limit 5
```

**ROC / PR comparison charts** → `artifacts/figures/metrics_*.png`:

```bash
python scripts/generate_metrics_charts.py
```

`eval_figures` / `train_all` also regenerates sample CA maps + metric charts into `artifacts/figures/`.

---

## Outputs layout

```text
numerical_nextday/
  outputs/m4_shared_cache/ …
  artifacts/
    experiments_log.csv
    eval_metrics.json
    figures/          # sample maps + metrics_*.png + calibration
    maps/daily/       # ~365 day maps (local; gitignored)
    models/
  data/california.geojson
```

Do **not** git-commit large caches or `artifacts/maps/daily/`; keep them local or sync via Drive/USB.
