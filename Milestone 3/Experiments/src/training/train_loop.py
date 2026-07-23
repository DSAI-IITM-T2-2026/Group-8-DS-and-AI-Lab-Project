"""MPS-aware training loop for ConvLSTM+U-Net and U-Net-last-day."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import config
from src.models.convlstm_unet import ConvLSTMUNet
from src.models.unet_last_day import UNetLastDay
from src.training.losses import get_loss
from src.training.metrics import compute_metrics


class PatchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, H, W, C)
        y = self.y[idx]  # (H, W, 1)
        x_t = torch.from_numpy(np.nan_to_num(x, nan=0.0)).float()
        y_t = torch.from_numpy(np.nan_to_num(y, nan=0.0)).float().permute(2, 0, 1)
        return x_t, y_t


def build_dl_model(
    model_name: str,
    *,
    in_channels: Optional[int] = None,
    hidden: int = config.DEFAULT_HIDDEN,
):
    """Factory for deep models used by train_models / run_experiments."""
    in_channels = in_channels or len(config.FEATURE_CHANNEL_NAMES)
    name = model_name.lower().replace("-", "_")
    if name in ("convlstm", "convlstm_unet", "default"):
        return ConvLSTMUNet(in_channels=in_channels, hidden_channels=hidden)
    if name in ("unet_last_day", "unet", "unet_lastday"):
        return UNetLastDay(in_channels=in_channels, hidden_channels=hidden)
    raise ValueError(f"Unknown DL model: {model_name}")


def _move_batch(x, y, device):
    try:
        return x.to(device), y.to(device)
    except Exception as exc:
        msg = f"batch.to({device}) failed: {exc}"
        if msg not in config.MPS_FALLBACKS:
            config.MPS_FALLBACKS.append(msg)
        return x.to("cpu"), y.to("cpu")


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits, all_targets = [], []
    for x, y in loader:
        x, y = _move_batch(x, y, device)
        model_device = next(model.parameters()).device
        if x.device != model_device:
            model = model.to(x.device)
        try:
            logits = model(x)
        except Exception as exc:
            msg = f"forward MPS fallback: {exc}"
            if msg not in config.MPS_FALLBACKS:
                config.MPS_FALLBACKS.append(msg)
            model = model.to("cpu")
            x, y = x.to("cpu"), y.to("cpu")
            logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * len(x)
        n += len(x)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(logits_np, targets_np)
    return total_loss / max(n, 1), metrics, logits_np, targets_np


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    model_name: str = "convlstm",
    build_model: Optional[Callable] = None,
    hidden: int = config.DEFAULT_HIDDEN,
    lr: float = config.DEFAULT_LR,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    loss_name: str = "bce_dice",
    epochs: int = config.DEFAULT_EPOCHS,
    patience: int = config.EARLY_STOP_PATIENCE,
    device=None,
    checkpoint_name: Optional[str] = None,
) -> dict:
    device = device or config.DEVICE
    if checkpoint_name is None:
        checkpoint_name = f"best_{model_name}_{loss_name}.pt"
    print(f"Training on device: {device}  model={model_name}  loss={loss_name}")

    train_loader = DataLoader(
        PatchDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        PatchDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if build_model is not None:
        model = build_model().to(device)
    else:
        model = build_dl_model(model_name, hidden=hidden).to(device)
    loss_fn = get_loss(loss_name)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = -1.0
    best_path = config.CHECKPOINTS_DIR / checkpoint_name
    history = []
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = _move_batch(x, y, device)
            if next(model.parameters()).device != x.device:
                model = model.to(x.device)
            opt.zero_grad()
            try:
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
            except Exception as exc:
                msg = f"train step MPS fallback: {exc}"
                if msg not in config.MPS_FALLBACKS:
                    config.MPS_FALLBACKS.append(msg)
                model = model.to("cpu")
                x, y = x.to("cpu"), y.to("cpu")
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                device = torch.device("cpu")
            running += float(loss.item()) * len(x)
            n += len(x)

        train_loss = running / max(n, 1)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, device, loss_fn)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
                if k not in ("accuracy_caveat", "confusion_matrix")
            },
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  val_auc_pr={val_metrics['auc_pr']:.4f}"
        )

        score = val_metrics["auc_pr"]
        if np.isnan(score):
            score = val_metrics["f1"]
        if score > best_auc:
            best_auc = score
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "hidden": hidden,
                    "loss_name": loss_name,
                    "lr": lr,
                    "batch_size": batch_size,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    return {
        "model": model,
        "history": history,
        "checkpoint": str(best_path),
        "best_val_metrics": ckpt["val_metrics"],
        "config": {
            "model_name": model_name,
            "hidden": hidden,
            "lr": lr,
            "batch_size": batch_size,
            "loss_name": loss_name,
            "device": str(device),
        },
    }
