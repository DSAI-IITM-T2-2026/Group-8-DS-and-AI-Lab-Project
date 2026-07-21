#!/usr/bin/env python3
"""Train DualBranchCNN, calibrate, write alerts + test predictions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.dataset import WildfirePatchDataset
from src.model import DualBranchCNN
from src.sample import TABULAR_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cnn_train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu | cuda | mps")
    return p.parse_args()


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if y.sum() == 0 or y.sum() == len(y):
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for image, tab, y in loader:
        image = image.to(device)
        tab = tab.to(device)
        logits = model(image, tab)
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(y_all)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["output_dir"])
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(cfg["paths"]["manifest"])
    feat_cols = [c for c in TABULAR_FEATURES if c in manifest.columns]
    meta_path = out_dir / "dataset_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("feature_columns"):
            feat_cols = [c for c in meta["feature_columns"] if c in manifest.columns]

    train_df = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest.loc[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    logger.info(
        "splits train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Empty split — rebuild dataset with 2018–2021 MVP outputs")

    mean = train_df[feat_cols].astype(np.float64).mean().to_numpy()
    std = train_df[feat_cols].astype(np.float64).std(ddof=0).to_numpy()
    np.savez(model_dir / "norm_stats.npz", mean=mean, std=std, columns=np.array(feat_cols))

    patches_dir = Path(cfg["paths"]["patches_dir"])
    train_ds = WildfirePatchDataset(train_df, feat_cols, mean, std, patches_dir)
    val_ds = WildfirePatchDataset(val_df, feat_cols, mean, std, patches_dir)
    test_ds = WildfirePatchDataset(test_df, feat_cols, mean, std, patches_dir)

    mcfg = cfg["model"]
    bs = int(mcfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=int(mcfg["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=int(mcfg["num_workers"]))
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=int(mcfg["num_workers"]))

    device = _device(args.device)
    logger.info("Device: %s", device)

    model = DualBranchCNN(
        n_tabular=len(feat_cols),
        in_ch=int(cfg["patch"]["bands"]),
        cnn_embed=int(mcfg["cnn_embed_dim"]),
        mlp_embed=int(mcfg["mlp_embed_dim"]),
        dropout=float(mcfg["dropout"]),
    ).to(device)

    n_pos = max(int(train_df["y_fire"].sum()), 1)
    n_neg = max(len(train_df) - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(mcfg["lr"]),
        weight_decay=float(mcfg["weight_decay"]),
    )

    epochs = int(args.epochs or mcfg["epochs"])
    best_pr = -1.0
    best_path = model_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for image, tab, y in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            image, tab, y = image.to(device), tab.to(device), y.to(device)
            opt.zero_grad()
            logits = model(image, tab)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_logits, val_y = predict_logits(model, val_loader, device)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        vm = _metrics(val_y, val_prob)
        logger.info(
            "epoch %d  loss=%.4f  val_pr=%.4f  val_roc=%.4f",
            epoch,
            float(np.mean(losses)),
            vm["pr_auc"],
            vm["roc_auc"],
        )
        if vm["pr_auc"] == vm["pr_auc"] and vm["pr_auc"] > best_pr:
            best_pr = vm["pr_auc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_columns": feat_cols,
                    "config": {
                        "n_tabular": len(feat_cols),
                        "in_ch": int(cfg["patch"]["bands"]),
                        "cnn_embed": int(mcfg["cnn_embed_dim"]),
                        "mlp_embed": int(mcfg["mlp_embed_dim"]),
                        "dropout": float(mcfg["dropout"]),
                    },
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    val_logits, val_y = predict_logits(model, val_loader, device)
    test_logits, test_y = predict_logits(model, test_loader, device)
    val_raw = 1.0 / (1.0 + np.exp(-val_logits))
    test_raw = 1.0 / (1.0 + np.exp(-test_logits))

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_y)
    val_cal = calibrator.predict(val_raw)
    test_cal = calibrator.predict(test_raw)

    metrics = {
        "val_raw": _metrics(val_y, val_raw),
        "val_calibrated": _metrics(val_y, val_cal),
        "test_raw": _metrics(test_y, test_raw),
        "test_calibrated": _metrics(test_y, test_cal),
        "best_val_pr_auc": best_pr,
        "pos_weight": float(n_neg / n_pos),
    }
    with (model_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    joblib.dump(calibrator, model_dir / "calibrator.joblib")

    test_out = test_df.copy()
    test_out["p_fire"] = test_cal.astype("float32")
    test_out["confidence_pct"] = (test_cal * 100.0).astype("float32")
    test_out.to_parquet(model_dir / "test_predictions.parquet", index=False)

    top_k = int(cfg["alerts"]["top_k"])
    alerts = (
        test_out.sort_values(["label_date", "confidence_pct"], ascending=[True, False])
        .groupby("label_date", group_keys=False)
        .head(top_k)
    )
    alert_cols = [
        "label_date",
        "region",
        "cell_id",
        "latitude",
        "longitude",
        "confidence_pct",
        "p_fire",
        "y_fire",
    ]
    for c in alert_cols:
        if c not in alerts.columns:
            if c == "region":
                alerts["region"] = [
                    f"cell:{cid} ({lat:.2f},{lon:.2f})"
                    for cid, lat, lon in zip(
                        alerts["cell_id"], alerts["latitude"], alerts["longitude"]
                    )
                ]
    alerts[alert_cols].to_csv(model_dir / "test_alerts_topk.csv", index=False)
    alerts[alert_cols].to_parquet(model_dir / "test_alerts_topk.parquet", index=False)
    logger.info("Wrote alerts → %s", model_dir / "test_alerts_topk.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
