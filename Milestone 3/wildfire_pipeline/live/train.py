#!/usr/bin/env python3
"""Train LagFireNet with precision@0.5 checkpointing + isotonic calibration."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# allow `python -m live.train` from wildfire_pipeline/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live import config
from live.dataset import LiveWildfireDataset, apply_norm, compute_norm_stats, split_dates
from live.gcs_fetch import ensure_dem, prefetch_era5_range
from live.losses import FocalTverskyLoss
from live.model import LagFireNet
from live.regrid import build_land_mask, build_reference_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def pixel_precision_at_threshold(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).float()
    tp = (pred * targets).sum().item()
    fp = (pred * (1 - targets)).sum().item()
    if tp + fp < 1:
        return 0.0
    return tp / (tp + fp)


@torch.no_grad()
def evaluate(model, loader, device, stats, criterion=None):
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits = []
    all_targets = []
    for batch in loader:
        batch = apply_norm(batch, stats)
        era5 = batch["era5"].to(device)
        s5p = batch["s5p"].to(device)
        s2 = batch["s2"].to(device)
        dem = batch["dem"].to(device)
        y = batch["label"].to(device)
        logits = model(era5, s5p, s2, dem)
        if criterion is not None:
            total_loss += criterion(logits, y).item()
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
        n += 1
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    prec = pixel_precision_at_threshold(logits, targets, 0.5)
    return {
        "loss": total_loss / max(n, 1),
        "precision@0.5": prec,
        "logits": logits,
        "targets": targets,
    }


def choose_threshold(logits: torch.Tensor, targets: torch.Tensor, min_prec: float) -> dict:
    probs = torch.sigmoid(logits).numpy().ravel()
    y = targets.numpy().ravel()
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for thr in np.linspace(0.05, 0.95, 37):
        pred = probs >= thr
        tp = float(((pred) & (y > 0.5)).sum())
        fp = float(((pred) & (y <= 0.5)).sum())
        fn = float(((~pred) & (y > 0.5)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        if prec >= min_prec and (prec > best["precision"] or thr < best["threshold"]):
            # smallest thr with prec >= target → track candidates then pick min thr
            if best["precision"] < min_prec or thr < best["threshold"]:
                best = {"threshold": float(thr), "precision": float(prec), "recall": float(rec)}
    # if never hit target, take max-precision thr
    if best["precision"] < min_prec:
        for thr in np.linspace(0.05, 0.95, 37):
            pred = probs >= thr
            tp = float(((pred) & (y > 0.5)).sum())
            fp = float(((pred) & (y <= 0.5)).sum())
            fn = float(((~pred) & (y > 0.5)).sum())
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            if prec > best["precision"]:
                best = {"threshold": float(thr), "precision": float(prec), "recall": float(rec)}
    # prefer smallest threshold among those with prec >= min_prec
    candidates = []
    for thr in np.linspace(0.05, 0.95, 37):
        pred = probs >= thr
        tp = float(((pred) & (y > 0.5)).sum())
        fp = float(((pred) & (y <= 0.5)).sum())
        fn = float(((~pred) & (y > 0.5)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        if prec >= min_prec:
            candidates.append((thr, prec, rec))
    if candidates:
        thr, prec, rec = min(candidates, key=lambda t: t[0])
        best = {"threshold": float(thr), "precision": float(prec), "recall": float(rec)}
    return best


def main():
    parser = argparse.ArgumentParser(description="Train live wildfire model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--crops-per-day", type=int, default=4)
    parser.add_argument("--max-train-days", type=int, default=0, help="0 = all")
    parser.add_argument("--max-val-days", type=int, default=0)
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--fire-season", action="store_true", help="Only May–Nov dates")
    parser.add_argument("--smoke", action="store_true", help="tiny run for wiring check")
    parser.add_argument(
        "--patience",
        type=int,
        default=config.EARLY_STOP_PATIENCE,
        help="epochs without val precision gain before stopping",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=config.EARLY_STOP_MIN_DELTA,
        help="min absolute gain in val precision@0.5 to reset patience",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=config.MIN_EPOCHS,
        help="never early-stop before this epoch",
    )
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = config.DEVICE
    logger.info("device=%s", device)

    logger.info("Building reference grid + DEM + land mask")
    build_reference_grid()
    build_land_mask()
    ensure_dem()

    train_dates = split_dates("train", fire_season=args.fire_season)
    val_dates = split_dates("val", fire_season=args.fire_season)
    if args.smoke:
        train_dates = train_dates[180:200] if len(train_dates) > 200 else train_dates[:20]
        val_dates = val_dates[180:190] if len(val_dates) > 190 else val_dates[:10]
        args.epochs = min(args.epochs, 2)
        args.crops_per_day = 2
    if args.max_train_days > 0:
        train_dates = train_dates[: args.max_train_days]
    if args.max_val_days > 0:
        val_dates = val_dates[: args.max_val_days]
    logger.info(
        "dates train=%d val=%d fire_season=%s range=%s..%s",
        len(train_dates),
        len(val_dates),
        args.fire_season,
        train_dates[0] if train_dates else None,
        train_dates[-1] if train_dates else None,
    )

    if not args.skip_prefetch:
        logger.info("Prefetching ERA5 for train/val span")
        prefetch_era5_range(train_dates[0], val_dates[-1])

    train_ds = LiveWildfireDataset(
        train_dates,
        split="train",
        crops_per_day=args.crops_per_day,
        fire_oversample=0.75,  # most crops centered on fire pixels
        mem_days=8,  # fewer feature rebuilds across shuffled tiles
    )
    val_ds = LiveWildfireDataset(
        val_dates,
        split="val",
        crops_per_day=max(2, args.crops_per_day // 2),
        fire_oversample=0.5,
        mem_days=8,
    )
    if len(train_ds) == 0:
        raise SystemExit(
            "Train dataset is empty — ERA5 failed to open for all days. "
            "pip install h5netcdf netCDF4 h5py  (ERA5 .nc files are ZIP archives)."
        )
    if len(val_ds) == 0:
        logger.warning("Val dataset empty — check ERA5 for val dates")

    logger.info("Computing norm stats")
    stats = compute_norm_stats(train_ds, max_batches=min(64, len(train_ds)))
    np.savez(config.ARTIFACTS_DIR / config.NORM_STATS_NAME, **stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = LagFireNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    # One-cycle-ish warmup: linear warmup 1 epoch then cosine toward 0.1× LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1
    )
    criterion = FocalTverskyLoss()

    best_prec = -1.0
    bad_epochs = 0
    history = []
    best_val_logits = None
    best_val_targets = None
    patience = args.patience
    min_delta = args.min_delta
    min_epochs = args.min_epochs
    logger.info(
        "early stop: patience=%d min_delta=%.4f min_epochs=%d (metric=val precision@0.5)",
        patience,
        min_delta,
        min_epochs,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_ds._rebuild_index()
        running = 0.0
        running_fl = 0.0
        running_tv = 0.0
        n_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = apply_norm(batch, stats)
            era5 = batch["era5"].to(device)
            s5p = batch["s5p"].to(device)
            s2 = batch["s2"].to(device)
            dem = batch["dem"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(era5, s5p, s2, dem)
            loss, fl, tv = criterion(logits, y, return_components=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            opt.step()
            running += loss.item()
            running_fl += fl.item()
            running_tv += tv.item()
            n_steps += 1
            pbar.set_postfix(
                loss=f"{running / n_steps:.4f}",
                fl=f"{running_fl / n_steps:.3f}",
                tv=f"{running_tv / n_steps:.3f}",
                fire=f"{y.mean().item():.4f}",
            )

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, stats, criterion)
        train_loss = running / max(n_steps, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_precision@0.5": val_metrics["precision@0.5"],
        }
        history.append(row)
        logger.info(
            "epoch %d train_loss=%.4f val_loss=%.4f val_prec@0.5=%.4f best=%.4f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["precision@0.5"],
            max(best_prec, 0.0),
        )

        prec = val_metrics["precision@0.5"]
        # Count as improvement only if precision rises by at least min_delta
        if prec >= best_prec + min_delta:
            best_prec = prec
            bad_epochs = 0
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_precision@0.5": prec,
                "flags": {
                    "use_era5": True,
                    "use_s5p": True,
                    "use_s2": True,
                    "use_dem": True,
                },
            }
            torch.save(ckpt, config.ARTIFACTS_DIR / config.CHECKPOINT_NAME)
            logger.info("Saved best checkpoint (prec=%.4f)", prec)
            best_val_logits = val_metrics["logits"]
            best_val_targets = val_metrics["targets"]
        else:
            bad_epochs += 1
            logger.info(
                "No meaningful improvement (need Δ ≥ %.4f); patience %d/%d",
                min_delta,
                bad_epochs,
                patience,
            )

        if epoch >= min_epochs and bad_epochs >= patience:
            logger.info(
                "Early stop at epoch %d (no val prec@0.5 gain ≥ %.4f for %d epochs; best=%.4f)",
                epoch,
                min_delta,
                patience,
                best_prec,
            )
            break

    # reload best
    ckpt_path = config.ARTIFACTS_DIR / config.CHECKPOINT_NAME
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        val_metrics = evaluate(model, val_loader, device, stats)
        best_val_logits, best_val_targets = val_metrics["logits"], val_metrics["targets"]

    # Isotonic calibration on validation
    from sklearn.isotonic import IsotonicRegression
    import joblib

    probs = torch.sigmoid(best_val_logits).numpy().ravel()
    y = best_val_targets.numpy().ravel()
    # subsample for speed
    if len(y) > 500_000:
        rng = np.random.default_rng(config.SEED)
        sel = rng.choice(len(y), 500_000, replace=False)
        probs, y = probs[sel], y[sel]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, y)
    joblib.dump(iso, config.ARTIFACTS_DIR / config.CALIBRATOR_NAME)

    thr_info = choose_threshold(best_val_logits, best_val_targets, config.DEPLOY_PRECISION_TARGET)
    thr_info["precision_target"] = config.DEPLOY_PRECISION_TARGET
    (config.ARTIFACTS_DIR / config.THRESHOLD_NAME).write_text(json.dumps(thr_info, indent=2))
    (config.ARTIFACTS_DIR / "history.json").write_text(json.dumps(history, indent=2))
    logger.info("Done. threshold=%s artifacts=%s", thr_info, config.ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
