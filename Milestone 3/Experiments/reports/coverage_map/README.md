# California coverage explorer

Theme-aware interactive map (D3 + month selector). Flip months to see S2/S5P CSV completeness on the shared CA grid.

## View

Needs a local server (ES modules + JSON fetch):

```bash
cd reports/coverage_map
python -m http.server 8765
# open http://localhost:8765/
```

## Refresh data

```bash
cd ../..   # Milestone 3/Experiments/
python scripts/verify_gcs_data.py --year 2025
```

## Behavior

- **Month** dropdown drives fill opacity of the CSV grid extent (S2 completeness).
- Absent months → destructive fill; present → `--viz-series-1` with opacity by window count.
- Chip strips list S2 (~5-day) and S5P (daily) window IDs for the selection.
- Colors use CSS theme tokens (`--foreground`, `--viz-series-1`, …) with light/dark fallbacks.
