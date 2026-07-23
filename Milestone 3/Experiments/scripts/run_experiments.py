#!/usr/bin/env python3
"""Offline multi-model experiment pack on existing local .npy splits.

Runs:
  A) HistGB baseline
  B) ConvLSTM + BCE+Dice
  C) ConvLSTM + Focal
  D) U-Net last-day + BCE+Dice

Writes data/processed/metadata/experiment_comparison.json

Usage:
  python scripts/run_experiments.py --epochs 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GS_NO_SIGN_REQUEST"] = "YES"

from src import config
from src.data.dataset import load_splits
from src.models.baseline import flatten_last_day, train_xgboost_baseline
from src.training.losses import get_loss
from src.training.metrics import write_eval_bundle
from src.training.train_loop import PatchDataset, evaluate, train_model
from torch.utils.data import DataLoader


def _slim_metrics(m: dict) -> dict:
    if not m:
        return {}
    keep = [
        "precision",
        "recall",
        "f1",
        "dice",
        "auc_pr",
        "accuracy",
        "confusion_matrix",
        "positive_rate_true",
        "positive_rate_pred",
    ]
    return {k: m[k] for k in keep if k in m}


def run_baseline(Xtr, ytr, Xva, yva, Xte, yte) -> dict:
    print("\n=== [A] Tree baseline ===")
    baseline = train_xgboost_baseline(Xtr, ytr, Xva, yva, Xte, yte)
    with open(config.METADATA_DIR / "baseline_metrics.json", "w") as f:
        json.dump(baseline["metrics"], f, indent=2)
    for split_name, Xs, ys in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        xf, yf = flatten_last_day(Xs, ys)
        prob = baseline["model"].predict_proba(xf)[:, 1]
        write_eval_bundle(
            name="baseline",
            split=split_name,
            metrics=baseline["metrics"][split_name],
            y_true=yf,
            y_prob=prob,
        )
    return baseline


def run_dl(model_name: str, loss_name: str, Xtr, ytr, Xva, yva, Xte, yte, epochs: int) -> dict:
    print(f"\n=== DL model={model_name} loss={loss_name} ===")
    ckpt_name = f"best_{model_name}_{loss_name}.pt"
    dl = train_model(
        Xtr,
        ytr,
        Xva,
        yva,
        model_name=model_name,
        loss_name=loss_name,
        epochs=epochs,
        checkpoint_name=ckpt_name,
    )
    device = next(dl["model"].parameters()).device
    loss_fn = get_loss(loss_name)
    out = {
        "config": dl["config"],
        "checkpoint": dl["checkpoint"],
        "val": dl["best_val_metrics"],
    }
    for split_name, Xs, ys in [("val", Xva, yva), ("test", Xte, yte)]:
        loader = DataLoader(PatchDataset(Xs, ys), batch_size=config.DEFAULT_BATCH_SIZE)
        _, metrics, logits, targets = evaluate(dl["model"], loader, device, loss_fn)
        out[split_name] = metrics
        write_eval_bundle(
            name=f"{model_name}_{loss_name}",
            split=split_name,
            metrics=metrics,
            logits=logits,
            targets=targets,
        )
    meta_name = (
        "dl_default_metrics.json"
        if model_name == "convlstm" and loss_name == "bce_dice"
        else f"dl_{model_name}_{loss_name}_metrics.json"
    )
    with open(config.METADATA_DIR / meta_name, "w") as f:
        json.dump(
            {"val": out["val"], "test": out["test"], "config": out["config"], "checkpoint": out["checkpoint"]},
            f,
            indent=2,
        )
    print(
        json.dumps(
            {k: v for k, v in out["test"].items() if k != "confusion_matrix"},
            indent=2,
        )
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    splits, _, meta = load_splits()
    Xtr, ytr = splits["train"]["X"], splits["train"]["y"]
    Xva, yva = splits["val"]["X"], splits["val"]["y"]
    Xte, yte = splits["test"]["X"], splits["test"]["y"]
    print(f"Loaded train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "start": meta.get("start"),
            "end": meta.get("end"),
            "shapes": meta.get("shapes"),
            "note": (
                "Patch-level FIRMS segmentation on fused 30-channel tensors. "
                "Sibling to Milestone 3 multimodal_fusion (cell-day / monthly image CNNs) — not 1:1 comparable."
            ),
        },
        "experiments": [],
    }

    if not args.skip_baseline:
        baseline = run_baseline(Xtr, ytr, Xva, yva, Xte, yte)
        comparison["experiments"].append(
            {
                "id": "A",
                "name": "baseline_histgb",
                "model": "hist_gradient_boosting",
                "loss": None,
                "val": _slim_metrics(baseline["metrics"]["val"]),
                "test": _slim_metrics(baseline["metrics"]["test"]),
            }
        )

    for exp_id, model_name, loss_name, label in [
        ("B", "convlstm", "bce_dice", "convlstm_bce_dice"),
        ("C", "convlstm", "focal", "convlstm_focal"),
        ("D", "unet_last_day", "bce_dice", "unet_last_day_bce_dice"),
    ]:
        out = run_dl(model_name, loss_name, Xtr, ytr, Xva, yva, Xte, yte, args.epochs)
        comparison["experiments"].append(
            {
                "id": exp_id,
                "name": label,
                "model": model_name,
                "loss": loss_name,
                "config": out.get("config"),
                "checkpoint": out.get("checkpoint"),
                "val": _slim_metrics(out.get("val") or {}),
                "test": _slim_metrics(out.get("test") or {}),
            }
        )

    out_path = config.METADATA_DIR / "experiment_comparison.json"
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\n=== Experiment summary (test) ===")
    print(f"{'id':3} {'name':28} {'F1':8} {'AUC-PR':8} {'Prec':8} {'Rec':8}")
    for e in comparison["experiments"]:
        t = e.get("test") or {}
        print(
            f"{e['id']:3} {e['name']:28} "
            f"{t.get('f1', float('nan')):8.4f} "
            f"{t.get('auc_pr', float('nan')):8.4f} "
            f"{t.get('precision', float('nan')):8.4f} "
            f"{t.get('recall', float('nan')):8.4f}"
        )


if __name__ == "__main__":
    main()
