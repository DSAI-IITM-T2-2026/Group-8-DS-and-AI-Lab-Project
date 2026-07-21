#!/usr/bin/env python3
"""Train CNN+LSTM(+optional S5P) fusion, calibrate, write alerts + predictions."""

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

from src.config import load_config, use_sentinel5p
from src.dataset import FusionDataset
from src.model import CNNLSTMFusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fusion_train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu | cuda | mps")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-sentinel5p", action="store_true", help="Force S5P branch on")
    g.add_argument("--no-sentinel5p", action="store_true", help="Force S5P branch off")
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


def _seq_norm_stats(train_df: pd.DataFrame, sequences_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute per-feature mean/std over flattened train sequences."""
    sums = None
    sq = None
    n = 0
    seq_dim = None
    for path in train_df["sequence_path"]:
        p = Path(path)
        if not p.is_absolute():
            p = sequences_dir / p.name
        seq = np.load(p).astype(np.float64)  # [T, F]
        if seq_dim is None:
            seq_dim = seq.shape[1]
            sums = np.zeros(seq_dim, dtype=np.float64)
            sq = np.zeros(seq_dim, dtype=np.float64)
        flat = seq.reshape(-1, seq_dim)
        sums += flat.sum(axis=0)
        sq += (flat**2).sum(axis=0)
        n += flat.shape[0]
    mean = sums / max(n, 1)
    var = sq / max(n, 1) - mean**2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32), int(seq_dim)


@torch.no_grad()
def predict_logits(model, loader, device, with_s5p: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        if with_s5p:
            image, seq, s5p, y = batch
            logits = model(image.to(device), seq.to(device), s5p.to(device))
        else:
            image, seq, y = batch
            logits = model(image.to(device), seq.to(device))
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(y_all)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cli_force = True if args.use_sentinel5p else (False if args.no_sentinel5p else None)
    with_s5p = use_sentinel5p(cfg, cli_force)
    logger.info("use_sentinel5p=%s", with_s5p)

    out_dir = Path(cfg["paths"]["output_dir"])
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = Path(cfg["paths"]["patches_dir"])
    sequences_dir = Path(cfg["paths"]["sequences_dir"])

    manifest = pd.read_parquet(cfg["paths"]["manifest"])
    if "sequence_path" not in manifest.columns:
        raise SystemExit("manifest missing sequence_path — run build_sequences.py first")
    if with_s5p and "s5p_aerosol" not in manifest.columns:
        raise SystemExit(
            "S5P enabled but manifest has no s5p_aerosol — run build_s5p_features.py first"
        )

    train_df = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest.loc[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    logger.info("splits train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Empty split — rebuild dataset for 2022–2025")

    seq_mean, seq_std, seq_dim = _seq_norm_stats(train_df, sequences_dir)
    np.savez(model_dir / "seq_norm_stats.npz", mean=seq_mean, std=seq_std)

    s5p_mean, s5p_std = 0.0, 1.0
    if with_s5p:
        s5p_vals = train_df["s5p_aerosol"].astype(np.float64).to_numpy()
        s5p_mean = float(np.nanmean(s5p_vals))
        s5p_std = float(np.nanstd(s5p_vals) or 1.0)
        np.savez(model_dir / "s5p_norm_stats.npz", mean=s5p_mean, std=s5p_std)

    train_ds = FusionDataset(
        train_df, seq_mean, seq_std, patches_dir, sequences_dir, with_s5p, s5p_mean, s5p_std
    )
    val_ds = FusionDataset(
        val_df, seq_mean, seq_std, patches_dir, sequences_dir, with_s5p, s5p_mean, s5p_std
    )
    test_ds = FusionDataset(
        test_df, seq_mean, seq_std, patches_dir, sequences_dir, with_s5p, s5p_mean, s5p_std
    )

    mcfg = cfg["model"]
    bs = int(mcfg["batch_size"])
    nw = int(mcfg["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    device = _device(args.device)
    logger.info("Device: %s  seq_dim=%d", device, seq_dim)

    model = CNNLSTMFusion(
        seq_dim=seq_dim,
        in_ch=int(cfg["patch"]["bands"]),
        cnn_embed=int(mcfg["cnn_embed_dim"]),
        lstm_embed=int(mcfg["lstm_embed_dim"]),
        lstm_hidden=int(mcfg["lstm_hidden"]),
        s5p_embed=int(mcfg["s5p_embed_dim"]),
        dropout=float(mcfg["dropout"]),
        use_sentinel5p=with_s5p,
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
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            opt.zero_grad()
            if with_s5p:
                image, seq, s5p, y = batch
                logits = model(image.to(device), seq.to(device), s5p.to(device))
            else:
                image, seq, y = batch
                logits = model(image.to(device), seq.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_logits, val_y = predict_logits(model, val_loader, device, with_s5p)
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
                    "config": {
                        "seq_dim": seq_dim,
                        "in_ch": int(cfg["patch"]["bands"]),
                        "cnn_embed": int(mcfg["cnn_embed_dim"]),
                        "lstm_embed": int(mcfg["lstm_embed_dim"]),
                        "lstm_hidden": int(mcfg["lstm_hidden"]),
                        "s5p_embed": int(mcfg["s5p_embed_dim"]),
                        "dropout": float(mcfg["dropout"]),
                        "use_sentinel5p": with_s5p,
                    },
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    val_logits, val_y = predict_logits(model, val_loader, device, with_s5p)
    test_logits, test_y = predict_logits(model, test_loader, device, with_s5p)
    val_raw = 1.0 / (1.0 + np.exp(-val_logits))
    test_raw = 1.0 / (1.0 + np.exp(-test_logits))

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_y)
    val_cal = calibrator.predict(val_raw)
    test_cal = calibrator.predict(test_raw)

    metrics = {
        "use_sentinel5p": with_s5p,
        "val_raw": _metrics(val_y, val_raw),
        "val_calibrated": _metrics(val_y, val_cal),
        "test_raw": _metrics(test_y, test_raw),
        "test_calibrated": _metrics(test_y, test_cal),
        "best_val_pr_auc": best_pr,
        "pos_weight": float(n_neg / n_pos),
        "seq_dim": seq_dim,
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
    if "region" not in alerts.columns:
        alerts["region"] = [
            f"cell:{cid} ({lat:.2f},{lon:.2f})"
            for cid, lat, lon in zip(alerts["cell_id"], alerts["latitude"], alerts["longitude"])
        ]
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
    alerts[alert_cols].to_csv(model_dir / "test_alerts_topk.csv", index=False)
    alerts[alert_cols].to_parquet(model_dir / "test_alerts_topk.parquet", index=False)
    logger.info("Wrote alerts → %s", model_dir / "test_alerts_topk.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
