"""Fire-centered patch sampling for the 2024 candidate dataset."""
from __future__ import annotations

import gc
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage

from src import config
from src.data.fusion import feature_lookup_fn_patch, label_lookup_fn_patch


def cluster_fire_pixels(
    label: np.ndarray,
    dilate_px: int = config.FIRE_CLUSTER_DILATE_PX,
    max_clusters: int = config.MAX_FIRE_CLUSTERS_PER_DAY,
) -> list[tuple[int, int]]:
    """
    Dilate fire mask to merge nearby detections, then return centroids of up
    to max_clusters largest connected components as (row, col).
    """
    fire = label > 0
    if not fire.any():
        return []
    if dilate_px > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        fire = ndimage.binary_dilation(fire, structure=struct, iterations=dilate_px)
    labeled, n = ndimage.label(fire)
    if n == 0:
        return []
    sizes = ndimage.sum(fire, labeled, index=range(1, n + 1))
    order = np.argsort(sizes)[::-1][:max_clusters]
    centers = []
    for idx in order:
        ys, xs = np.where(labeled == (idx + 1))
        centers.append((int(ys.mean()), int(xs.mean())))
    return centers


def _clamp_patch_origin(center_r: int, center_c: int, H: int, W: int, patch_size: int):
    top = int(np.clip(center_r - patch_size // 2, 0, H - patch_size))
    left = int(np.clip(center_c - patch_size // 2, 0, W - patch_size))
    return top, left


def _patch_has_fire(label: np.ndarray, top: int, left: int, patch_size: int) -> bool:
    return bool(label[top : top + patch_size, left : left + patch_size].any())


def sample_background_origin(
    label: np.ndarray,
    fire_centers: list[tuple[int, int]],
    rng: np.random.Generator,
    patch_size: int,
    min_dist: int = config.BACKGROUND_MIN_DISTANCE_PX,
    max_tries: int = 200,
) -> tuple[int, int]:
    H, W = label.shape
    for _ in range(max_tries):
        top = int(rng.integers(0, max(1, H - patch_size)))
        left = int(rng.integers(0, max(1, W - patch_size)))
        if _patch_has_fire(label, top, left, patch_size):
            continue
        cy, cx = top + patch_size // 2, left + patch_size // 2
        if fire_centers:
            dists = [np.hypot(cy - fy, cx - fx) for fy, fx in fire_centers]
            if min(dists) < min_dist:
                continue
        return top, left
    # fallback: any no-fire patch
    for _ in range(max_tries):
        top = int(rng.integers(0, max(1, H - patch_size)))
        left = int(rng.integers(0, max(1, W - patch_size)))
        if not _patch_has_fire(label, top, left, patch_size):
            return top, left
    return 0, 0


def patches_for_day(
    label: np.ndarray,
    rng: np.random.Generator,
    patch_size: int = config.PATCH_SIZE,
) -> list[tuple[int, int, str]]:
    """
    Returns list of (top, left, kind) where kind is 'fire' or 'background'.
    Fire day: ≤3 fire-centered + 1 background. No-fire day: 1 background.
    """
    H, W = label.shape
    centers = cluster_fire_pixels(label)
    out: list[tuple[int, int, str]] = []
    if centers:
        for cy, cx in centers:
            top, left = _clamp_patch_origin(cy, cx, H, W, patch_size)
            out.append((top, left, "fire"))
        btop, bleft = sample_background_origin(label, centers, rng, patch_size)
        out.append((btop, bleft, "background"))
    else:
        btop, bleft = sample_background_origin(label, [], rng, patch_size)
        out.append((btop, bleft, "background"))
    return out


def build_fire_centered_dataset(
    dates: list[str],
    *,
    s2_cache,
    era5_cache,
    s5p_cache,
    firms_label_cache,
    dem_features_on_firms_grid: dict,
    history_days: int = config.HISTORY_DAYS,
    patch_size: int = config.PATCH_SIZE,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build (X, y, meta) for usable target dates.
    X: (N, 7, H, W, 30), y: (N, H, W, 1)
    Target day is dates[i]; history is the previous `history_days` calendar days.
    """
    rng = np.random.default_rng(seed)
    H, W = dem_features_on_firms_grid["elevation_m"].shape
    date_set = set(dates)
    dates_sorted = sorted(dates)

    X_samples, y_samples, meta = [], [], []

    # Usable targets need history_days prior days present
    for i, target_date in enumerate(dates_sorted):
        t = pd.Timestamp(target_date)
        window_dates = [
            (t - pd.Timedelta(days=d)).strftime("%Y-%m-%d")
            for d in range(history_days, 0, -1)
        ]
        if any(d not in date_set for d in window_dates):
            # allow synthetic continuity via FIRMS availability — skip if any missing
            # For calendar 2024 FIRMS is daily; still guard.
            missing = [d for d in window_dates if d not in date_set]
            if missing:
                continue

        try:
            label = firms_label_cache.get(target_date)
        except Exception as exc:
            print(f"  skip day {target_date}: FIRMS label failed: {exc}", flush=True)
            continue
        patch_list = patches_for_day(label, rng, patch_size)

        for top, left, kind in patch_list:
            try:
                X_seq = np.stack(
                    [
                        feature_lookup_fn_patch(
                            d,
                            top,
                            left,
                            patch_size,
                            s2_cache=s2_cache,
                            era5_cache=era5_cache,
                            s5p_cache=s5p_cache,
                            firms_label_cache=firms_label_cache,
                            dem_features_on_firms_grid=dem_features_on_firms_grid,
                        )
                        for d in window_dates
                    ],
                    axis=0,
                )
                y_patch = label_lookup_fn_patch(
                    target_date,
                    top,
                    left,
                    patch_size,
                    firms_label_cache=firms_label_cache,
                )
            except Exception as exc:
                print(f"  skip {target_date} patch ({top},{left}): {exc}")
                continue

            X_samples.append(X_seq)
            y_samples.append(y_patch)
            meta.append(
                {
                    "target_date": target_date,
                    "window_dates": window_dates,
                    "top": top,
                    "left": left,
                    "kind": kind,
                    "has_fire": bool(y_patch.any()),
                }
            )

        if i % 10 == 0:
            gc.collect()
            print(f"  processed {i+1}/{len(dates_sorted)} dates, {len(X_samples)} samples")

    X = np.array(X_samples, dtype=np.float32)
    y = np.array(y_samples, dtype=np.float32)
    return X, y, meta


def temporal_split_by_date(
    X: np.ndarray,
    y: np.ndarray,
    meta: list[dict],
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
):
    """70/15/15 split by unique target dates (earliest → train, latest → test)."""
    unique_dates = sorted({m["target_date"] for m in meta})
    n = len(unique_dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train : n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val :])

    def mask(dateset):
        return np.array([m["target_date"] in dateset for m in meta])

    splits = {}
    for name, dset in [("train", train_dates), ("val", val_dates), ("test", test_dates)]:
        m = mask(dset)
        splits[name] = {
            "X": X[m],
            "y": y[m],
            "meta": [meta[i] for i in range(len(meta)) if m[i]],
            "dates": sorted(dset),
        }
    assert not (train_dates & val_dates) and not (train_dates & test_dates) and not (val_dates & test_dates)
    return splits
