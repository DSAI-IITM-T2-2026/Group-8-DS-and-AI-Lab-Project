#!/usr/bin/env python3
"""
Write pred-vs-truth risk map PNGs for the PPT / Report (teammate map_predictions pattern).

Loads local test splits + ConvLSTM checkpoint — no internet required.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")

from src import config
from src.data.dataset import load_splits
from src.models.convlstm_unet import ConvLSTMUNet


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
def _predict(model, x_np: np.ndarray, device: torch.device) -> np.ndarray:
    """x_np: (T, H, W, C) → probability map (H, W)."""
    x = torch.from_numpy(np.nan_to_num(x_np, nan=0.0)).float().unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return prob


def main():
    parser = argparse.ArgumentParser(description="Pred vs truth patch maps")
    parser.add_argument("--n", type=int, default=6, help="Number of test patches to plot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.CHECKPOINTS_DIR / "best_convlstm_unet.pt"),
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}\nRun train_models.py first.")

    splits, _, meta = load_splits()
    Xte, yte = splits["test"]["X"], splits["test"]["y"]
    print(f"Test set: X={Xte.shape} y={yte.shape}")

    device = config.DEVICE
    model = _load_model(ckpt_path, device)
    print(f"Loaded {ckpt_path.name} on {device}")

    # Prefer fire-positive patches for illustrative maps
    fire_counts = yte.reshape(len(yte), -1).sum(axis=1)
    order = np.argsort(-fire_counts)
    rng = np.random.default_rng(args.seed)
    # Mix top fire patches with a couple of random ones
    top = order[: max(args.n * 2, args.n)].tolist()
    rng.shuffle(top)
    idxs = top[: args.n]
    if len(idxs) < args.n:
        idxs = list(range(min(args.n, len(Xte))))

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    meta_test = None
    meta_path = config.OUTPUT_DIR / "test" / "meta_test.json"
    if meta_path.exists():
        import json

        meta_test = json.loads(meta_path.read_text())

    for i, idx in enumerate(idxs):
        x = Xte[idx]
        y = yte[idx][..., 0] if yte.ndim == 4 else yte[idx]
        prob = _predict(model, x, device)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        axes[0].imshow(y, cmap="Reds", vmin=0, vmax=1)
        axes[0].set_title("FIRMS label (truth)")
        axes[1].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("ConvLSTM+U-Net P(fire)")
        axes[2].imshow((prob >= 0.5).astype(float), cmap="Reds", vmin=0, vmax=1)
        axes[2].set_title("Pred @ 0.5")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        title = f"test patch idx={idx}"
        if isinstance(meta_test, list) and idx < len(meta_test):
            m = meta_test[idx]
            if isinstance(m, dict):
                title += f"  date={m.get('date', m.get('target_date', '?'))}"
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        out = config.FIGURES_DIR / f"pred_vs_truth_{i+1:02d}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out.name}  fire_px={int(fire_counts[idx])}  mean_p={prob.mean():.4f}")

    # Also a single contact sheet
    n = len(idxs)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n))
    if n == 1:
        axes = np.array([axes])
    for row, idx in enumerate(idxs):
        y = yte[idx][..., 0] if yte.ndim == 4 else yte[idx]
        prob = _predict(model, Xte[idx], device)
        axes[row, 0].imshow(y, cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[row, 2].imshow((prob >= 0.5).astype(float), cmap="Reds", vmin=0, vmax=1)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        if row == 0:
            axes[0, 0].set_title("Truth")
            axes[0, 1].set_title("P(fire)")
            axes[0, 2].set_title("Pred@0.5")
        axes[row, 0].set_ylabel(f"#{idx}", rotation=0, labelpad=20, va="center")
    fig.suptitle("Test patches — truth vs prediction", fontsize=12)
    fig.tight_layout()
    sheet = config.FIGURES_DIR / "pred_vs_truth_sheet.png"
    fig.savefig(sheet, dpi=150)
    plt.close(fig)
    print(f"Wrote contact sheet: {sheet}")
    print("Done (offline — no GCS).")


if __name__ == "__main__":
    main()
