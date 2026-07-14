#!/usr/bin/env python3
"""Extract numerical terrain values for California and write EDA tables/plots."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eda.extract")

EDA_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with (EDA_ROOT / "config.yaml").open() as f:
        return yaml.safe_load(f)


def read_valid(path: Path, nodata: float) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    mask = (data != nodata) & np.isfinite(data)
    return data[mask]


def layer_stats(values: np.ndarray, name: str) -> dict:
    return {
        "layer": name,
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "skew": float(pd.Series(values).skew()),
    }


def main() -> None:
    cfg = load_config()
    nodata = cfg["clip"]["nodata"]
    clipped_dir = EDA_ROOT / cfg["paths"]["output_dir"] / "clipped_ca"
    num_dir = EDA_ROOT / cfg["paths"]["output_dir"] / "numerical"
    fig_dir = EDA_ROOT / cfg["paths"]["output_dir"] / "figures"
    num_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    layers = cfg["extraction"]["layers"]
    rng = np.random.default_rng(cfg["extraction"]["random_seed"])
    max_n = cfg["extraction"]["max_sample_pixels"]

    stats_rows = []
    sample_frames = []

    for layer in layers:
        path = clipped_dir / f"{layer}_ca.tif"
        if not path.exists():
            logger.warning("Missing clipped layer %s — run 01_clip_to_california.py first", path)
            continue

        values = read_valid(path, nodata)
        logger.info("%s: %s valid pixels", layer, f"{values.size:,}")
        stats_rows.append(layer_stats(values, layer))

        if values.size > max_n:
            idx = rng.choice(values.size, size=max_n, replace=False)
            sampled = values[idx]
        else:
            sampled = values
        sample_frames.append(pd.DataFrame({layer: sampled}))

    stats_df = pd.DataFrame(stats_rows)
    stats_path = num_dir / "california_terrain_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info("Wrote %s", stats_path)

    # Align samples to same length by independent sampling already done —
    # build a side-by-side sample table by truncating to min length.
    if sample_frames:
        min_len = min(len(df) for df in sample_frames)
        sample_df = pd.concat(
            [df.iloc[:min_len].reset_index(drop=True) for df in sample_frames],
            axis=1,
        )
        sample_path = num_dir / "california_terrain_sample.parquet"
        sample_df.to_parquet(sample_path, index=False)
        sample_df.to_csv(num_dir / "california_terrain_sample.csv", index=False)
        logger.info("Wrote sample table %s (%d rows)", sample_path, len(sample_df))

        corr = sample_df.corr(numeric_only=True)
        corr.to_csv(num_dir / "terrain_correlation.csv")

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title("California DEM terrain feature correlations")
        plt.tight_layout()
        plt.savefig(fig_dir / "terrain_correlation.png", dpi=140)
        plt.close()

        n = len(sample_df.columns)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, sample_df.columns):
            ax.hist(sample_df[col].dropna(), bins=60, color="#2a6f6f", alpha=0.85)
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel("count")
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle("California DEM — value distributions (state polygon)", y=1.01)
        fig.tight_layout()
        fig.savefig(fig_dir / "terrain_histograms.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    with (num_dir / "extraction_meta.json").open("w") as f:
        json.dump(
            {
                "boundary": "California state polygon (Census TIGER)",
                "not_bbox": True,
                "downsample_factor": cfg["clip"]["downsample_factor"],
                "layers": layers,
                "sample_rows": int(sample_df.shape[0]) if sample_frames else 0,
            },
            f,
            indent=2,
        )

    logger.info("Numerical extraction complete")


if __name__ == "__main__":
    main()
