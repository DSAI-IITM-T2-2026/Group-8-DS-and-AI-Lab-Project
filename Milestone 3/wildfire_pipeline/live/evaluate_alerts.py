#!/usr/bin/env python3
"""Cluster-level alert metrics on a date range (default: 2025 test)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live import config
from live.dataset import split_dates
from live.infer_live import load_artifacts, predict_full_grid
from live.labels import load_or_build_label
from live.regrid import build_land_mask, build_reference_grid, load_grid_meta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_alerts")


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labeled, n = ndimage.label(mask.astype(np.uint8))
    return labeled, int(n)


def cluster_centroids(labeled: np.ndarray, n: int) -> list[tuple[int, int]]:
    cents = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        cents.append((int(ys.mean()), int(xs.mean())))
    return cents


def pixel_radius(km: float, meta: dict) -> int:
    # ~1 km FIRMS cells
    return max(1, int(round(km)))


def alert_hit(
    pred_cents: list[tuple[int, int]],
    true_mask: np.ndarray,
    radius_px: int,
) -> list[bool]:
    if not pred_cents:
        return []
    # dilate true fires
    struct = ndimage.generate_binary_structure(2, 1)
    true_d = ndimage.binary_dilation(true_mask > 0, structure=struct, iterations=radius_px)
    return [bool(true_d[r, c]) for r, c in pred_cents]


def event_recall(
    true_labeled: np.ndarray,
    n_true: int,
    pred_mask: np.ndarray,
    radius_px: int,
) -> float:
    if n_true == 0:
        return float("nan")
    struct = ndimage.generate_binary_structure(2, 1)
    pred_d = ndimage.binary_dilation(pred_mask > 0, structure=struct, iterations=radius_px)
    caught = 0
    for i in range(1, n_true + 1):
        if pred_d[true_labeled == i].any():
            caught += 1
    return caught / n_true


def evaluate_dates(dates: list[str], thr: float) -> dict:
    build_reference_grid()
    land = build_land_mask()
    meta = load_grid_meta()
    radius = pixel_radius(config.ALERT_HIT_RADIUS_KM, meta)
    model, stats, calibrator, thr_file = load_artifacts()
    if thr is None:
        thr = float(thr_file.get("threshold", 0.5))

    total_alerts = 0
    total_hits = 0
    days_with_alerts = 0
    event_recalls = []
    n_days = 0

    for date_str in dates:
        try:
            probs = predict_full_grid(date_str, model, stats, calibrator)
            label = load_or_build_label(date_str)
        except Exception as e:
            logger.warning("skip %s: %s", date_str, e)
            continue
        pred = ((probs >= thr) & land).astype(np.uint8)
        true = ((label > 0) & land).astype(np.uint8)
        pred_lab, n_pred = connected_components(pred)
        true_lab, n_true = connected_components(true)
        cents = cluster_centroids(pred_lab, n_pred)
        hits = alert_hit(cents, true, radius)
        total_alerts += len(cents)
        total_hits += int(sum(hits))
        if cents:
            days_with_alerts += 1
        er = event_recall(true_lab, n_true, pred, radius)
        if not np.isnan(er):
            event_recalls.append(er)
        n_days += 1
        logger.info(
            "%s alerts=%d hits=%d true_clusters=%d",
            date_str,
            len(cents),
            int(sum(hits)),
            n_true,
        )

    precision = total_hits / total_alerts if total_alerts else 0.0
    out = {
        "n_days": n_days,
        "alerts_issued": total_alerts,
        "alert_precision": precision,
        "alerts_per_day": total_alerts / max(n_days, 1),
        "event_recall_mean": float(np.mean(event_recalls)) if event_recalls else 0.0,
        "threshold": thr,
        "hit_radius_km": config.ALERT_HIT_RADIUS_KM,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    dates = split_dates(args.split)
    if args.max_days > 0:
        dates = dates[: args.max_days]
    metrics = evaluate_dates(dates, args.threshold)
    logger.info("metrics=%s", metrics)
    out = Path(args.out) if args.out else config.ARTIFACTS_DIR / f"alert_metrics_{args.split}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
