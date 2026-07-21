#!/usr/bin/env python3
"""Train MultimodalFusion (S2 CNN + S5P CNN + LSTM + numerical MLPs)."""

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
from src.dataset import MultimodalDataset
from src.model import MultimodalFusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("multimodal_train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
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


def _seq_norm(train_df: pd.DataFrame, sequences_dir: Path):
    sums = sq = None
    n = 0
    dim = None
    for path in train_df["sequence_path"]:
        p = Path(path)
        if not p.is_absolute():
            p = sequences_dir / p.name
        seq = np.load(p).astype(np.float64)
        if dim is None:
            dim = seq.shape[1]
            sums = np.zeros(dim)
            sq = np.zeros(dim)
        flat = seq.reshape(-1, dim)
        sums += flat.sum(0)
        sq += (flat**2).sum(0)
        n += flat.shape[0]
    mean = sums / max(n, 1)
    std = np.sqrt(np.clip(sq / max(n, 1) - mean**2, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32), int(dim)


def _vec_norm(df: pd.DataFrame, cols: list[str]):
    x = df[cols].astype(np.float64).to_numpy()
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _forward(model, batch, device, flags):
    seq = batch["seq"].to(device)
    kwargs = {"seq": seq}
    if flags["use_s2_patches"]:
        kwargs["s2_image"] = batch["s2_image"].to(device)
    if flags["use_s5p_patches"]:
        kwargs["s5p_image"] = batch["s5p_image"].to(device)
    if flags["use_s2_numerical"]:
        kwargs["s2_num"] = batch["s2_num"].to(device)
    if flags["use_s5p_numerical"]:
        kwargs["s5p_num"] = batch["s5p_num"].to(device)
    return model(**kwargs)


@torch.no_grad()
def predict(model, loader, device, flags):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        logits = _forward(model, batch, device, flags)
        logits_all.append(logits.cpu().numpy())
        y_all.append(batch["y"].numpy())
    return np.concatenate(logits_all), np.concatenate(y_all)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    flags = {
        "use_s2_patches": bool(mcfg.get("use_s2_patches", True)),
        "use_s5p_patches": bool(mcfg.get("use_s5p_patches", True)),
        "use_s2_numerical": bool(mcfg.get("use_s2_numerical", True)),
        "use_s5p_numerical": bool(mcfg.get("use_s5p_numerical", True)),
    }
    logger.info("flags=%s", flags)

    out_dir = Path(cfg["paths"]["output_dir"])
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = Path(cfg["paths"]["patches_dir"])
    s5p_patches_dir = Path(cfg["paths"]["s5p_patches_dir"])
    sequences_dir = Path(cfg["paths"]["sequences_dir"])

    manifest = pd.read_parquet(cfg["paths"]["manifest"])
    if "sequence_path" not in manifest.columns:
        raise SystemExit("Run build_sequences.py first")
    if flags["use_s2_patches"] and "patch_path" not in manifest.columns:
        raise SystemExit("Run build_dataset.py first")
    if flags["use_s5p_patches"] and "s5p_patch_path" not in manifest.columns:
        raise SystemExit("Run build_s5p_patches.py first")

    s2_cols = ["s2n_" + c for c in cfg["sources"]["s2_features"]["columns"]]
    s5_cols = ["s5n_" + c for c in cfg["sources"]["s5p_features"]["columns"]]
    s2_cols = [c for c in s2_cols if c in manifest.columns]
    s5_cols = [c for c in s5_cols if c in manifest.columns]
    if flags["use_s2_numerical"] and not s2_cols:
        raise SystemExit("S2 numerical enabled but columns missing — run build_numerical_features.py")
    if flags["use_s5p_numerical"] and not s5_cols:
        logger.warning("S5P numerical columns missing — disabling branch")
        flags["use_s5p_numerical"] = False

    train_df = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest.loc[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    logger.info("splits train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    seq_mean, seq_std, seq_dim = _seq_norm(train_df, sequences_dir)
    np.savez(model_dir / "seq_norm_stats.npz", mean=seq_mean, std=seq_std)

    s2_mean = s2_std = s5_mean = s5_std = None
    if flags["use_s2_numerical"]:
        s2_mean, s2_std = _vec_norm(train_df, s2_cols)
        np.savez(model_dir / "s2_num_norm.npz", mean=s2_mean, std=s2_std, columns=np.array(s2_cols))
    if flags["use_s5p_numerical"]:
        s5_mean, s5_std = _vec_norm(train_df, s5_cols)
        np.savez(model_dir / "s5p_num_norm.npz", mean=s5_mean, std=s5_std, columns=np.array(s5_cols))

    def make_ds(df):
        return MultimodalDataset(
            df,
            seq_mean,
            seq_std,
            patches_dir,
            s5p_patches_dir,
            sequences_dir,
            flags["use_s2_patches"],
            flags["use_s5p_patches"],
            flags["use_s2_numerical"],
            flags["use_s5p_numerical"],
            s2_cols,
            s5_cols,
            s2_mean,
            s2_std,
            s5_mean,
            s5_std,
        )

    bs = int(mcfg["batch_size"])
    nw = int(mcfg["num_workers"])
    train_loader = DataLoader(make_ds(train_df), batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(make_ds(val_df), batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(make_ds(test_df), batch_size=bs, shuffle=False, num_workers=nw)

    device = _device(args.device)
    model = MultimodalFusion(
        seq_dim=seq_dim,
        s2_num_dim=len(s2_cols),
        s5p_num_dim=len(s5_cols),
        s2_in_ch=int(cfg["patch"]["bands"]),
        s5p_in_ch=int(cfg["patch_s5p"]["bands"]),
        cnn_embed=int(mcfg["cnn_embed_dim"]),
        s5p_cnn_embed=int(mcfg["s5p_cnn_embed_dim"]),
        lstm_embed=int(mcfg["lstm_embed_dim"]),
        lstm_hidden=int(mcfg["lstm_hidden"]),
        s2_num_embed=int(mcfg["s2_num_embed_dim"]),
        s5p_num_embed=int(mcfg["s5p_num_embed_dim"]),
        dropout=float(mcfg["dropout"]),
        **flags,
    ).to(device)
    logger.info("Device=%s seq_dim=%d s2_num=%d s5_num=%d", device, seq_dim, len(s2_cols), len(s5_cols))

    n_pos = max(int(train_df["y_fire"].sum()), 1)
    n_neg = max(len(train_df) - n_pos, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=float(mcfg["lr"]), weight_decay=float(mcfg["weight_decay"]))

    epochs = int(args.epochs or mcfg["epochs"])
    best_pr = -1.0
    best_path = model_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            opt.zero_grad()
            logits = _forward(model, batch, device, flags)
            loss = criterion(logits, batch["y"].to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_logits, val_y = predict(model, val_loader, device, flags)
        val_prob = 1 / (1 + np.exp(-val_logits))
        vm = _metrics(val_y, val_prob)
        logger.info(
            "epoch %d loss=%.4f val_pr=%.4f val_roc=%.4f",
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
                    "flags": flags,
                    "s2_num_cols": s2_cols,
                    "s5p_num_cols": s5_cols,
                    "seq_dim": seq_dim,
                    "config": mcfg,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    val_logits, val_y = predict(model, val_loader, device, flags)
    test_logits, test_y = predict(model, test_loader, device, flags)
    val_raw = 1 / (1 + np.exp(-val_logits))
    test_raw = 1 / (1 + np.exp(-test_logits))
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_y)
    test_cal = calibrator.predict(test_raw)
    val_cal = calibrator.predict(val_raw)

    metrics = {
        "flags": flags,
        "val_raw": _metrics(val_y, val_raw),
        "val_calibrated": _metrics(val_y, val_cal),
        "test_raw": _metrics(test_y, test_raw),
        "test_calibrated": _metrics(test_y, test_cal),
        "best_val_pr_auc": best_pr,
    }
    with (model_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))
    joblib.dump(calibrator, model_dir / "calibrator.joblib")

    test_out = test_df.copy()
    test_out["p_fire"] = test_cal.astype("float32")
    test_out["confidence_pct"] = (test_cal * 100).astype("float32")
    test_out.to_parquet(model_dir / "test_predictions.parquet", index=False)

    top_k = int(cfg["alerts"]["top_k"])
    alerts = (
        test_out.sort_values(["label_date", "confidence_pct"], ascending=[True, False])
        .groupby("label_date", group_keys=False)
        .head(top_k)
    )
    if "region" not in alerts.columns:
        alerts["region"] = [
            f"cell:{c} ({la:.2f},{lo:.2f})"
            for c, la, lo in zip(alerts["cell_id"], alerts["latitude"], alerts["longitude"])
        ]
    cols = [
        "label_date",
        "region",
        "cell_id",
        "latitude",
        "longitude",
        "confidence_pct",
        "p_fire",
        "y_fire",
    ]
    alerts[cols].to_csv(model_dir / "test_alerts_topk.csv", index=False)
    alerts[cols].to_parquet(model_dir / "test_alerts_topk.parquet", index=False)
    logger.info("Wrote alerts → %s", model_dir / "test_alerts_topk.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
