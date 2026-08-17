# Cutoff-policy replay

The replay runner scores a fully reconstructed 2023–2025 **one-day ERA5
fallback** panel with the unchanged champion artifact and reports PR-AUC and
Recall@25 against the published held-out references (0.1451 and 0.3638). The
input panel must have `feature_end_date = D−7`; shifting only columns in the
already-pruned 86-feature table is not valid because weather interactions and
rolling features must be recomputed by the production feature pipeline.

The runner refuses partial date ranges, missing daily provenance,
non-California cutoffs, unready sources, ERA5 snapshots whose age is not
exactly one day, or panels that still use the exact `D−6` endpoint.

```bash
python daily_pipeline/scripts/replay_cutoff_policy.py \
  --model /secure/path/champion_model.joblib \
  --panel /secure/path/cutoff_replay_2023_2025.parquet \
  --provenance-jsonl /secure/path/cutoff_replay_2023_2025.provenance.jsonl \
  --output documentation/cutoff_policy_replay_results.json
```

The large trained artifact and reconstructed historical fallback panel are
intentionally not checked into this repository, so no metric is fabricated in
source control. Review the generated JSON before deciding whether retraining
is warranted; this implementation never retrains automatically.
