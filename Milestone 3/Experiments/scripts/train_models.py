#!/usr/bin/env python3
"""Train baseline and/or DL models + optional hyperparameter search.

Examples:
  python scripts/train_models.py --model all --loss bce_dice
  python scripts/train_models.py --model convlstm --loss focal --epochs 15
  python scripts/train_models.py --model unet_last_day --loss bce_dice
  python scripts/train_models.py --model convlstm --tune --epochs 15 --trials 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GS_NO_SIGN_REQUEST"] = "YES"

from src import config
from src.data.dataset import load_splits
from src.models.baseline import flatten_last_day, train_xgboost_baseline
from src.training.losses import get_loss
from src.training.metrics import pr_curve, write_eval_bundle
from src.training.train_loop import PatchDataset, evaluate, train_model
from src.training.tune import random_search
from torch.utils.data import DataLoader


def _run_baseline(Xtr, ytr, Xva, yva, Xte, yte) -> dict:
    print("\n=== Tree baseline ===")
    baseline = train_xgboost_baseline(Xtr, ytr, Xva, yva, Xte, yte)
    with open(config.METADATA_DIR / "baseline_metrics.json", "w") as f:
        json.dump(baseline["metrics"], f, indent=2)
    print("model:", baseline["metrics"].get("model"))
    print(json.dumps({k: v for k, v in baseline["metrics"]["test"].items() if k != "confusion_matrix"}, indent=2))
    print("Floor:", json.dumps({k: v for k, v in baseline["metrics"]["floor_test"].items() if k != "confusion_matrix"}, indent=2))

    # Confusion / classification report for each split using last-day flatten
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


def _run_dl(model_name: str, loss_name: str, Xtr, ytr, Xva, yva, Xte, yte, epochs: int) -> dict:
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
    out = {"config": dl["config"], "checkpoint": dl["checkpoint"], "val": dl["best_val_metrics"]}

    for split_name, Xs, ys in [("val", Xva, yva), ("test", Xte, yte)]:
        loader = DataLoader(PatchDataset(Xs, ys), batch_size=config.DEFAULT_BATCH_SIZE)
        _, metrics, logits, targets = evaluate(dl["model"], loader, device, loss_fn)
        out[split_name] = metrics
        tag = f"{model_name}_{loss_name}"
        write_eval_bundle(
            name=tag,
            split=split_name,
            metrics=metrics,
            logits=logits,
            targets=targets,
        )
        if split_name == "test":
            out["test_logits"] = logits
            out["test_targets"] = targets

    meta_name = (
        "dl_default_metrics.json"
        if model_name == "convlstm" and loss_name == "bce_dice"
        else f"dl_{model_name}_{loss_name}_metrics.json"
    )
    serializable = {k: v for k, v in out.items() if k not in ("test_logits", "test_targets")}
    with open(config.METADATA_DIR / meta_name, "w") as f:
        json.dump(serializable, f, indent=2)
    print(json.dumps({k: v for k, v in out["test"].items() if k != "confusion_matrix"}, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["baseline", "convlstm", "unet_last_day", "all"],
        help="Which model family to train",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="bce_dice",
        choices=["bce_dice", "focal"],
        help="Loss for DL models (ignored for baseline)",
    )
    parser.add_argument("--tune", action="store_true", help="Run random hyperparameter search (ConvLSTM)")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    splits, norm_stats, meta = load_splits()
    Xtr, ytr = splits["train"]["X"], splits["train"]["y"]
    Xva, yva = splits["val"]["X"], splits["val"]["y"]
    Xte, yte = splits["test"]["X"], splits["test"]["y"]
    print(f"Loaded train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    baseline = None
    dl_default = None

    run_baseline = args.model in ("baseline", "all")
    run_convlstm = args.model in ("convlstm", "all")
    run_unet = args.model in ("unet_last_day", "all")

    if run_baseline:
        baseline = _run_baseline(Xtr, ytr, Xva, yva, Xte, yte)

    if run_convlstm:
        dl_default = _run_dl("convlstm", args.loss, Xtr, ytr, Xva, yva, Xte, yte, args.epochs)

    if run_unet:
        _run_dl("unet_last_day", args.loss, Xtr, ytr, Xva, yva, Xte, yte, args.epochs)

    # PR curve when we have both baseline and a DL test eval
    if baseline is not None and dl_default is not None and "test_logits" in dl_default:
        from sklearn.metrics import precision_recall_curve

        xs, ys = flatten_last_day(Xte, yte)
        base_prob = baseline["model"].predict_proba(xs)[:, 1]
        fig, ax = plt.subplots(figsize=(6, 5))
        bp, br, _ = precision_recall_curve(ys, base_prob)
        ax.plot(br, bp, label="Tree baseline")
        p, r, _ = pr_curve(dl_default["test_logits"], dl_default["test_targets"])
        if p is not None:
            ax.plot(r, p, label=f"ConvLSTM ({args.loss})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall: baseline vs DL")
        ax.legend()
        fig.tight_layout()
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / "pr_curve_baseline_vs_dl.png", dpi=150)
        plt.close(fig)

    if args.tune:
        if args.model not in ("convlstm", "all"):
            print("Note: --tune applies to ConvLSTM; running ConvLSTM search.")
        print("\n=== Hyperparameter random search (ConvLSTM) ===")
        random_search(
            Xtr,
            ytr,
            Xva,
            yva,
            Xte,
            yte,
            n_trials=args.trials,
            epochs=min(15, args.epochs),
            model_name="convlstm",
        )

    if config.MPS_FALLBACKS:
        with open(config.METADATA_DIR / "mps_fallbacks.json", "w") as f:
            json.dump(config.MPS_FALLBACKS, f, indent=2)
        print("MPS fallbacks logged:", config.MPS_FALLBACKS)

    print("Training complete.")


if __name__ == "__main__":
    main()
