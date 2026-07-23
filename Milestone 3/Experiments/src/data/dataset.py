"""Dataset build orchestration: normalize, save, load."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from src import config


def normalize_channels(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Z-score per channel from train stats only. X shape: (N, T, H, W, C)."""
    # mean/std over N, T, H, W
    channel_mean = np.nanmean(X_train, axis=(0, 1, 2, 3), keepdims=True)
    channel_std = np.nanstd(X_train, axis=(0, 1, 2, 3), keepdims=True) + 1e-8

    def apply(X):
        return ((X - channel_mean) / channel_std).astype(np.float32)

    stats = {
        "mean": channel_mean.squeeze().tolist(),
        "std": channel_std.squeeze().tolist(),
        "channel_names": config.FEATURE_CHANNEL_NAMES,
    }
    return apply(X_train), apply(X_val), apply(X_test), stats


def save_splits(splits: dict, norm_stats: dict, aoi_bounds: dict, extra_meta: Optional[dict] = None):
    out = config.OUTPUT_DIR
    for split_name in ("train", "val", "test"):
        split_dir = out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / f"X_{split_name}.npy", splits[split_name]["X"])
        np.save(split_dir / f"y_{split_name}.npy", splits[split_name]["y"])
        with open(split_dir / f"meta_{split_name}.json", "w") as f:
            json.dump(splits[split_name]["meta"], f, indent=2)

    meta = {
        "channel_names": config.FEATURE_CHANNEL_NAMES,
        "n_channels": len(config.FEATURE_CHANNEL_NAMES),
        "history_days": config.HISTORY_DAYS,
        "patch_size": config.PATCH_SIZE,
        "aoi_bounds": aoi_bounds,
        "split_dates": {k: splits[k]["dates"] for k in ("train", "val", "test")},
        "shapes": {
            k: {"X": list(splits[k]["X"].shape), "y": list(splits[k]["y"].shape)}
            for k in ("train", "val", "test")
        },
        "sampling": {
            "strategy": "fire_centered",
            "cluster_dilate_px": config.FIRE_CLUSTER_DILATE_PX,
            "max_clusters": config.MAX_FIRE_CLUSTERS_PER_DAY,
            "background_min_distance_px": config.BACKGROUND_MIN_DISTANCE_PX,
        },
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(config.METADATA_DIR / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    with open(config.METADATA_DIR / "dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved splits to {out}")


def load_splits(normalized: bool = True):
    """Load saved npy splits. Expects already-normalized arrays if built via build script."""
    out = {}
    for split_name in ("train", "val", "test"):
        split_dir = config.OUTPUT_DIR / split_name
        out[split_name] = {
            "X": np.load(split_dir / f"X_{split_name}.npy"),
            "y": np.load(split_dir / f"y_{split_name}.npy"),
        }
    with open(config.METADATA_DIR / "normalization_stats.json") as f:
        stats = json.load(f)
    with open(config.METADATA_DIR / "dataset_metadata.json") as f:
        meta = json.load(f)
    return out, stats, meta
