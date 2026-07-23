#!/usr/bin/env python3
"""
Statewide California wildfire risk map (teammate Milestone 3 style).

Plots patch-center confidence % on the CA outline with FIRMS-positive rings.
Uses existing local test/val/train .npy + checkpoint — no internet required.

Does NOT compete with build_dataset.py (separate process / files).
Run AFTER train:

  python scripts/map_state_risk.py
  python scripts/map_state_risk.py --date 2025-07-13
  python scripts/map_state_risk.py --split test --all-dates
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")

from src import config
from src.data.aoi import load_locked_aoi
from src.data.dataset import load_splits
from src.models.convlstm_unet import ConvLSTMUNet

GEOJSON_CANDIDATES = [
    config.DATA_DIR / "static" / "california.geojson",
    ROOT.parent / "Milestone 3" / "multimodal_fusion" / "data" / "california.geojson",
    ROOT.parent / "Milestone 3" / "cnn_lstm_fusion" / "data" / "california.geojson",
]


def _load_boundary(path: Path):
    try:
        import geopandas as gpd

        return gpd.read_file(path)
    except Exception:
        data = json.loads(path.read_text())
        polys = []

        def _coords(geom):
            t = geom["type"]
            c = geom["coordinates"]
            if t == "Polygon":
                return [c[0]]
            if t == "MultiPolygon":
                return [p[0] for p in c]
            return []

        feats = data["features"] if data.get("type") == "FeatureCollection" else [{"geometry": data}]
        for feat in feats:
            for ring in _coords(feat["geometry"]):
                polys.append(np.asarray(ring))
        return polys


def _load_model(checkpoint: Path, device: torch.device) -> ConvLSTMUNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    hidden = int(ckpt.get("hidden", config.DEFAULT_HIDDEN))
    model = ConvLSTMUNet(
        in_channels=len(config.FEATURE_CHANNEL_NAMES),
        hidden_channels=hidden,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _predict_batch(model, X: np.ndarray, device: torch.device, batch_size: int = 4) -> np.ndarray:
    """X: (N, T, H, W, C) → mean P(fire) per patch (N,)."""
    probs = []
    for i in range(0, len(X), batch_size):
        chunk = X[i : i + batch_size]
        x = torch.from_numpy(np.nan_to_num(chunk, nan=0.0)).float().to(device)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)
        probs.append(p[:, 0].mean(axis=(1, 2)))
    return np.concatenate(probs, axis=0)


def _grid_shape_from_aoi() -> tuple[int, int]:
    """
    Prefer shape recorded at build time; else infer from FIRMS reference (may need GCS).
    Smoke AOI clip was typically ~1031 x 1114.
    """
    meta_path = config.METADATA_DIR / "dataset_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        # optional future field
        if "firms_grid_shape" in meta:
            h, w = meta["firms_grid_shape"]
            return int(h), int(w)
    # Fall back: open locked AOI FIRMS ref (network once if no cache)
    try:
        from src.data.loaders import get_firms_reference

        aoi = load_locked_aoi()
        ref = get_firms_reference("2025-08-15", bounds=aoi)
        return int(ref.sizes["y"]), int(ref.sizes["x"])
    except Exception:
        return 1031, 1114


def _pixel_to_lonlat(row: float, col: float, aoi: dict, H: int, W: int) -> tuple[float, float]:
    lon = aoi["west"] + (col + 0.5) * (aoi["east"] - aoi["west"]) / W
    lat = aoi["north"] - (row + 0.5) * (aoi["north"] - aoi["south"]) / H
    return float(lon), float(lat)


def plot_day(day_rows: list[dict], boundary, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 9))

    if hasattr(boundary, "plot"):
        boundary.plot(ax=ax, facecolor="#f5f5f0", edgecolor="#333333", linewidth=0.8)
    else:
        for ring in boundary:
            ax.plot(ring[:, 0], ring[:, 1], color="#333333", linewidth=0.8)
            ax.fill(ring[:, 0], ring[:, 1], color="#f5f5f0", alpha=0.5)

    lons = [r["longitude"] for r in day_rows]
    lats = [r["latitude"] for r in day_rows]
    conf = [r["confidence_pct"] for r in day_rows]

    sc = ax.scatter(
        lons,
        lats,
        c=conf,
        s=36,
        cmap="YlOrRd",
        norm=Normalize(vmin=0, vmax=max(10.0, float(np.quantile(conf, 0.95)) if conf else 50)),
        edgecolors="k",
        linewidths=0.2,
        alpha=0.9,
        zorder=3,
    )
    pos = [r for r in day_rows if r.get("y_fire")]
    if pos:
        ax.scatter(
            [r["longitude"] for r in pos],
            [r["latitude"] for r in pos],
            facecolors="none",
            edgecolors="#0033aa",
            s=80,
            linewidths=1.2,
            label="FIRMS positive",
            zorder=4,
        )
        ax.legend(loc="lower left", frameon=True, fontsize=8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Confidence %")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    aoi = load_locked_aoi()
    ax.set_xlim(aoi["west"] - 0.3, aoi["east"] + 0.3)
    ax.set_ylim(aoi["south"] - 0.3, aoi["north"] + 0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path} ({len(day_rows)} patches)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: day with most fire patches)")
    parser.add_argument("--all-dates", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default=str(config.CHECKPOINTS_DIR / "best_convlstm_unet.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}\nRun train_models.py first.")

    geo = next((p for p in GEOJSON_CANDIDATES if p.exists()), None)
    if geo is None:
        raise SystemExit("california.geojson not found under data/static/ or Milestone 3/")

    boundary = _load_boundary(geo)
    aoi = load_locked_aoi()
    H, W = _grid_shape_from_aoi()
    print(f"AOI grid HxW ≈ {H}x{W}, geojson={geo}")

    splits, _, _ = load_splits()
    if args.split == "all":
        keys = ["train", "val", "test"]
    else:
        keys = [args.split]

    device = config.DEVICE
    model = _load_model(ckpt_path, device)

    rows: list[dict] = []
    for key in keys:
        X = splits[key]["X"]
        y = splits[key]["y"]
        meta_path = config.OUTPUT_DIR / key / f"meta_{key}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else [{}] * len(X)
        print(f"Predicting {key}: {X.shape[0]} patches...")
        mean_p = _predict_batch(model, X, device, batch_size=args.batch_size)
        for i in range(len(X)):
            m = meta[i] if i < len(meta) else {}
            top = int(m.get("top", 0))
            left = int(m.get("left", 0))
            cy = top + config.PATCH_SIZE / 2
            cx = left + config.PATCH_SIZE / 2
            lon, lat = _pixel_to_lonlat(cy, cx, aoi, H, W)
            y_patch = y[i][..., 0] if y.ndim == 4 else y[i]
            rows.append(
                {
                    "label_date": m.get("target_date", "unknown"),
                    "longitude": lon,
                    "latitude": lat,
                    "confidence_pct": float(mean_p[i] * 100.0),
                    "y_fire": bool(np.any(y_patch > 0)),
                    "split": key,
                    "top": top,
                    "left": left,
                }
            )

    if not rows:
        raise SystemExit("No patches to plot.")

    maps_dir = config.FIGURES_DIR / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # pick dates
    from collections import defaultdict

    by_date: dict[str, list] = defaultdict(list)
    for r in rows:
        by_date[str(r["label_date"])].append(r)

    if args.all_dates:
        dates = sorted(by_date.keys())
    elif args.date:
        dates = [args.date]
    else:
        # day with most FIRMS-positive patches, else highest mean confidence
        best = None
        best_score = (-1, -1.0)
        for d, rs in by_date.items():
            n_pos = sum(1 for r in rs if r["y_fire"])
            mean_c = float(np.mean([r["confidence_pct"] for r in rs]))
            score = (n_pos, mean_c)
            if score > best_score:
                best_score = score
                best = d
        dates = [best]

    for d in dates:
        day_rows = by_date.get(d, [])
        if not day_rows:
            print(f"No rows for {d}")
            continue
        title = f"California wildfire risk — {d}"
        plot_day(day_rows, boundary, maps_dir / f"risk_{d}.png", title)

    # also write a combined map of all selected split patches
    plot_day(
        rows,
        boundary,
        maps_dir / f"risk_{args.split}_all_patches.png",
        f"California wildfire risk — {args.split} patches (all dates)",
    )
    print(f"Maps directory: {maps_dir}")


if __name__ == "__main__":
    main()
