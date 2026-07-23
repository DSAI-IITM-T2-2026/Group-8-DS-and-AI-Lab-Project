"""Random hyperparameter search (proportionate to candidate subset + local MPS)."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from torch.utils.data import DataLoader

from src import config
from src.training.losses import get_loss
from src.training.metrics import pr_curve
from src.training.train_loop import PatchDataset, evaluate, train_model


def random_search(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    n_trials: int = 8,
    seed: int = config.SEED,
    epochs: int = 15,
    model_name: str = "convlstm",
) -> dict:
    rng = np.random.default_rng(seed)
    grid = {
        "lr": [1e-4, 3e-4, 1e-3],
        "hidden": [16, 32, 64],
        "loss_name": ["bce_dice", "focal"],
        "batch_size": [2, 4, 8],
    }
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    rng.shuffle(combos)
    combos = combos[:n_trials]

    results = []
    best = None

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        print(f"\n=== Trial {i+1}/{len(combos)}: {params} ===")
        try:
            out = train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                model_name=model_name,
                hidden=params["hidden"],
                lr=params["lr"],
                batch_size=params["batch_size"],
                loss_name=params["loss_name"],
                epochs=epochs,
                checkpoint_name=f"trial_{i+1}.pt",
            )
            device = next(out["model"].parameters()).device
            test_loader = DataLoader(
                PatchDataset(X_test, y_test),
                batch_size=params["batch_size"],
                shuffle=False,
            )
            loss_fn = get_loss(params["loss_name"])
            _, test_metrics, logits, targets = evaluate(
                out["model"], test_loader, device, loss_fn
            )
            row = {
                "trial": i + 1,
                "model_name": model_name,
                **params,
                "val_metrics": out["best_val_metrics"],
                "test_metrics": test_metrics,
                "checkpoint": out["checkpoint"],
            }
            results.append(row)
            score = out["best_val_metrics"].get("auc_pr", float("nan"))
            if np.isnan(score):
                score = out["best_val_metrics"].get("f1", 0)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "row": row,
                    "logits": logits,
                    "targets": targets,
                    "out": out,
                }
        except Exception as exc:
            print(f"Trial failed: {exc}")
            results.append({"trial": i + 1, "model_name": model_name, **params, "error": str(exc)})

    out_path = config.METADATA_DIR / "tuning_results.json"
    serializable = []
    for r in results:
        serializable.append({k: v for k, v in r.items()})
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    _plot_comparison(results, best)
    return {"results": results, "best": best, "path": str(out_path)}


def _plot_comparison(results, best):
    fig_dir = config.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if "test_metrics" in r]
    if ok:
        labels = [f"t{r['trial']}" for r in ok]
        f1s = [r["test_metrics"]["f1"] for r in ok]
        dices = [r["test_metrics"]["dice"] for r in ok]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - width / 2, f1s, width, label="F1")
        ax.bar(x + width / 2, dices, width, label="Dice")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_title("Hyperparameter trials — test F1 / Dice")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "tuning_f1_dice_bars.png", dpi=150)
        plt.close(fig)

    if best is not None:
        p, r, _ = pr_curve(best["logits"], best["targets"])
        if p is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(r, p, label=f"best DL (trial {best['row']['trial']})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision–Recall — best ConvLSTM+U-Net")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "pr_curve_best_dl.png", dpi=150)
            plt.close(fig)
