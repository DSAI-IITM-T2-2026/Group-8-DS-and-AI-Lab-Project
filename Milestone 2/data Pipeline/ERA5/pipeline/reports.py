"""HTML quality report generation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.validator import ValidationResult

logger = logging.getLogger(__name__)


def _validation_table(results: list[ValidationResult]) -> str:
    if not results:
        return "<p>No validation results available.</p>"

    rows = []
    for r in results:
        status = "✓" if r.valid else "✗"
        rows.append(
            f"<tr><td>{Path(r.file).name}</td>"
            f"<td>{status}</td>"
            f"<td>{r.time_steps}</td>"
            f"<td>{r.missing_fraction:.4%}</td>"
            f"<td>{'; '.join(r.errors) or '-'}</td>"
            f"<td>{'; '.join(r.warnings) or '-'}</td></tr>"
        )

    return (
        "<table><thead><tr>"
        "<th>File</th><th>Valid</th><th>Time Steps</th>"
        "<th>Missing</th><th>Errors</th><th>Warnings</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _plot_section(plot_paths: list[Path]) -> str:
    if not plot_paths:
        return "<p>No plots generated yet.</p>"

    items = []
    for path in plot_paths:
        rel = path.name
        items.append(f'<div class="plot"><h3>{path.stem}</h3><img src="plots/{rel}" alt="{rel}"></div>')
    return "\n".join(items)


def generate_quality_report(
    config: PipelineConfig,
    validation_results: list[ValidationResult],
    metadata: dict,
    plot_paths: list[Path],
) -> Path:
    outfile = config.quality_report()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    val = metadata.get("validation", {})
    download = metadata.get("download_stats", {})
    spatial = metadata.get("spatial_coverage", {})
    temporal = metadata.get("temporal_coverage", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ERA5 Wildfire Pipeline — Quality Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
    h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }}
    h2 {{ color: #1e40af; margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #eff6ff; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; }}
    .card strong {{ display: block; font-size: 1.5rem; color: #2563eb; }}
    .plot img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; }}
    .ok {{ color: #16a34a; }} .fail {{ color: #dc2626; }}
  </style>
</head>
<body>
  <h1>ERA5 Wildfire Data Pipeline — Quality Report</h1>
  <p>Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>

  <h2>Summary</h2>
  <div class="summary">
    <div class="card"><span>Raw Files</span><strong>{metadata.get("file_counts", {}).get("raw_monthly", 0)}</strong></div>
    <div class="card"><span>Merged Years</span><strong>{metadata.get("file_counts", {}).get("merged_yearly", 0)}</strong></div>
    <div class="card"><span>Processed Years</span><strong>{metadata.get("file_counts", {}).get("processed_yearly", 0)}</strong></div>
    <div class="card"><span>Valid Files</span><strong class="ok">{val.get("valid", 0)}</strong></div>
    <div class="card"><span>Invalid Files</span><strong class="fail">{val.get("invalid", 0)}</strong></div>
    <div class="card"><span>Downloads OK</span><strong>{download.get("success", 0)}</strong></div>
  </div>

  <h2>Spatial Coverage</h2>
  <ul>
    <li>Latitude: {spatial.get("lat_min", "N/A")} to {spatial.get("lat_max", "N/A")}</li>
    <li>Longitude: {spatial.get("lon_min", "N/A")} to {spatial.get("lon_max", "N/A")}</li>
    <li>Grid: {spatial.get("lat_points", "N/A")} × {spatial.get("lon_points", "N/A")}</li>
  </ul>

  <h2>Temporal Coverage</h2>
  <ul>
    <li>Start: {temporal.get("start", "N/A")}</li>
    <li>End: {temporal.get("end", "N/A")}</li>
    <li>Total hourly steps (processed): {temporal.get("total_hours", "N/A")}</li>
  </ul>

  <h2>Validation Results</h2>
  {_validation_table(validation_results)}

  <h2>Visualizations</h2>
  {_plot_section(plot_paths)}

</body>
</html>"""

    with open(outfile, "w") as f:
        f.write(html)

    logger.info("Wrote quality report: %s", outfile)
    return outfile
