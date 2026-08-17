# Cutoff-policy replay

The replay runner scores a fully reconstructed 2023–2025 panel with the
unchanged champion artifact and reports PR-AUC and Recall@25 against the
published held-out references (0.1451 and 0.3638). It refuses partial date
ranges, missing daily provenance, non-California cutoffs, or an unready source
snapshot.

```bash
python daily_pipeline/scripts/replay_cutoff_policy.py \
  --model /secure/path/champion_model.joblib \
  --panel /secure/path/cutoff_replay_2023_2025.parquet \
  --provenance-jsonl /secure/path/cutoff_replay_2023_2025.provenance.jsonl \
  --output documentation/cutoff_policy_replay_results.json
```

The large trained artifact and historical cutoff panel are intentionally not
checked into this repository, so no metric is fabricated in source control.
Review the generated JSON before deciding whether retraining is warranted.
