#!/usr/bin/env python3
"""Daily live inference: fetch → features → sliding-window predict → GeoJSON alerts."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live import config
from live.dataset import apply_norm
from live.features import build_features_for_label_day
from live.gcs_fetch import ensure_dem, prefetch_era5_range
from live.model import LagFireNet
from live.regrid import build_land_mask, build_reference_grid, load_grid_meta, reference_da
from rasterio.transform import xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_live")


def load_artifacts(device=None):
    device = device or config.DEVICE
    ckpt = torch.load(
        config.ARTIFACTS_DIR / config.CHECKPOINT_NAME,
        map_location=device,
        weights_only=False,
    )
    flags = ckpt.get("flags", {})
    model = LagFireNet(
        use_era5=bool(flags.get("use_era5", True)),
        use_s5p=bool(flags.get("use_s5p", True)),
        use_s2=bool(flags.get("use_s2", True)),
        use_dem=bool(flags.get("use_dem", True)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    stats = dict(np.load(config.ARTIFACTS_DIR / config.NORM_STATS_NAME))
    import joblib

    calibrator = None
    cal_path = config.ARTIFACTS_DIR / config.CALIBRATOR_NAME
    if cal_path.exists():
        calibrator = joblib.load(cal_path)
    thr = json.loads((config.ARTIFACTS_DIR / config.THRESHOLD_NAME).read_text())
    return model, stats, calibrator, thr


@torch.no_grad()
def predict_full_grid(
    label_day: str,
    model: LagFireNet,
    stats: dict,
    calibrator=None,
    device=None,
    tile: int = config.TILE_SIZE,
    overlap: int = config.INFER_OVERLAP,
) -> np.ndarray:
    device = device or config.DEVICE
    feats = build_features_for_label_day(label_day)
    if feats is None:
        raise RuntimeError(f"Features unavailable for {label_day}")

    era5 = torch.from_numpy(feats.era5).unsqueeze(0)
    s5p = torch.from_numpy(feats.s5p).unsqueeze(0)
    s2 = torch.from_numpy(feats.s2).unsqueeze(0)
    dem = torch.from_numpy(feats.dem).unsqueeze(0)
    batch = apply_norm({"era5": era5, "s5p": s5p, "s2": s2, "dem": dem}, stats)

    H, W = dem.shape[-2:]
    stride = tile - overlap
    acc = torch.zeros(1, 1, H, W, device=device)
    wgt = torch.zeros(1, 1, H, W, device=device)

    tops = list(range(0, max(1, H - tile + 1), stride))
    lefts = list(range(0, max(1, W - tile + 1), stride))
    if tops[-1] != H - tile and H >= tile:
        tops.append(H - tile)
    if lefts[-1] != W - tile and W >= tile:
        lefts.append(W - tile)

    for top in tops:
        for left in lefts:
            sl_y = slice(top, top + tile)
            sl_x = slice(left, left + tile)
            # pad if edge smaller than tile
            e = batch["era5"][:, :, :, sl_y, sl_x]
            s = batch["s5p"][:, :, :, sl_y, sl_x]
            s2t = batch["s2"][:, :, sl_y, sl_x]
            d = batch["dem"][:, :, sl_y, sl_x]
            th, tw = d.shape[-2:]
            if th < tile or tw < tile:
                e = F.pad(e, (0, tile - tw, 0, tile - th))
                s = F.pad(s, (0, tile - tw, 0, tile - th))
                s2t = F.pad(s2t, (0, tile - tw, 0, tile - th))
                d = F.pad(d, (0, tile - tw, 0, tile - th))
            logits = model(
                e.to(device),
                s.to(device),
                s2t.to(device),
                d.to(device),
            )
            logits = logits[:, :, :th, :tw]
            acc[:, :, sl_y, sl_x] += logits
            wgt[:, :, sl_y, sl_x] += 1.0

    logits = acc / wgt.clamp_min(1.0)
    probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
    if calibrator is not None:
        flat = probs.ravel()
        probs = calibrator.predict(flat).reshape(probs.shape).astype(np.float32)
    return probs


def clusters_to_geojson(
    pred_mask: np.ndarray,
    probs: np.ndarray,
    date_str: str,
) -> dict:
    labeled, n = ndimage.label(pred_mask.astype(np.uint8))
    meta = load_grid_meta()
    from rasterio.transform import Affine

    transform = Affine(*meta["transform"])
    features = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        r, c = int(ys.mean()), int(xs.mean())
        lon, lat = xy(transform, r, c)
        conf = float(probs[ys, xs].mean())
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {
                    "date": date_str,
                    "cluster_id": i,
                    "n_pixels": int(len(ys)),
                    "confidence": conf,
                    "row": r,
                    "col": c,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {"date": date_str, "n_alerts": len(features)},
    }


def run_day(label_day: str, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (config.ARTIFACTS_DIR / "alerts")
    out_dir.mkdir(parents=True, exist_ok=True)

    build_reference_grid()
    land = build_land_mask()
    ensure_dem()
    # prefetch feature window
    d0 = date.fromisoformat(label_day)
    prefetch_era5_range(
        (d0.replace(day=1) if d0.day > 1 else d0).isoformat(),
        label_day,
    )

    model, stats, calibrator, thr_info = load_artifacts()
    thr = float(thr_info.get("threshold", 0.5))
    probs = predict_full_grid(label_day, model, stats, calibrator)
    pred = ((probs >= thr) & land).astype(np.uint8)
    gj = clusters_to_geojson(pred, probs, label_day)
    out_path = out_dir / f"alerts_{label_day}.geojson"
    out_path.write_text(json.dumps(gj, indent=2))
    # also CSV
    import csv

    csv_path = out_dir / f"alerts_{label_day}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "lat", "lon", "confidence", "n_pixels", "cluster_id"])
        w.writeheader()
        for feat in gj["features"]:
            lon, lat = feat["geometry"]["coordinates"]
            p = feat["properties"]
            w.writerow(
                {
                    "date": label_day,
                    "lat": lat,
                    "lon": lon,
                    "confidence": p["confidence"],
                    "n_pixels": p["n_pixels"],
                    "cluster_id": p["cluster_id"],
                }
            )
    logger.info("Wrote %s (%d alerts)", out_path, gj["properties"]["n_alerts"])
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Live wildfire inference for day D")
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="Label date YYYY-MM-DD (default: today UTC)",
    )
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()
    label_day = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path(args.out_dir) if args.out_dir else None
    run_day(label_day, out_dir)


if __name__ == "__main__":
    main()
