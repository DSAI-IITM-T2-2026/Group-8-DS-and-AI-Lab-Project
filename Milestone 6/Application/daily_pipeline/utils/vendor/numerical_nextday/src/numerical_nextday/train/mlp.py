"""Small MLP secondary learner for Stage C fire_season."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from numerical_nextday.train.lgbm import _feature_cols, _load_stage_splits
from numerical_nextday.train.router import filter_bucket

logger = logging.getLogger(__name__)

MLP_TRIALS = [
    ("mlp_drop_0", {"dropout": 0.0}),
    ("mlp_drop_04", {"dropout": 0.4}),
    ("mlp_wd_0", {"weight_decay": 0.0}),
    ("mlp_wd_1e3", {"weight_decay": 1e-3}),
    ("mlp_lr_3e4", {"lr": 3e-4}),
    ("mlp_hid_64", {"hidden": 64}),
]


class FireMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if y.sum() == 0 or y.sum() == len(y):
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def train_one_mlp(cfg: dict, overrides: dict, experiment_id: str) -> dict:
    logger.info("MLP start %s", experiment_id)
    splits = _load_stage_splits(cfg, "stage_c")
    for k in splits:
        splits[k] = filter_bucket(splits[k], "fire_season", cfg)
    feat = _feature_cols(splits["train"], "C", None)  # avoid re-importing M3 mid-train
    mcfg = dict(cfg["model"]["mlp"])
    mcfg.update(overrides)

    def _xy(df):
        X = df[feat].fillna(0).to_numpy(dtype=np.float32)
        y = df["y_fire"].to_numpy(dtype=np.float32)
        return X, y

    Xtr, ytr = _xy(splits["train"])
    Xva, yva = _xy(splits["val"])
    Xte, yte = _xy(splits["test"])

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd

    device = torch.device("cpu")
    model = FireMLP(Xtr.shape[1], hidden=int(mcfg["hidden"]), dropout=float(mcfg["dropout"])).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(mcfg["lr"]), weight_decay=float(mcfg["weight_decay"])
    )
    pos = max(float(ytr.sum()), 1.0)
    neg = max(float(len(ytr) - pos), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)
    batch = int(mcfg["batch_size"])
    epochs = min(int(mcfg["epochs"]), 8)
    patience = int(mcfg["patience"])
    best_state = None
    best_pr = -1.0
    bad = 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xtr_t))
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xb = Xtr_t[idx]
            yb = ytr_t[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pva = torch.sigmoid(model(Xva_t)).numpy()
        pr = _metrics(yva, pva)["pr_auc"]
        logger.info("%s epoch %d val_pr=%.4f", experiment_id, epoch, pr if pr == pr else -1.0)
        if pr == pr and pr > best_pr:
            best_pr = pr
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = torch.sigmoid(model(torch.from_numpy(Xte))).numpy()
    metrics = {"experiment_id": experiment_id, "test": _metrics(yte, pte), "val_pr_auc": best_pr}
    out = Path(cfg["paths"]["artifacts_dir"]) / "models" / "fire_season"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": model.state_dict(), "mu": mu, "sd": sd, "feat": feat, "mcfg": mcfg},
        out / f"{experiment_id}.pt",
    )
    (out / f"{experiment_id}_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("MLP done %s test_pr=%.4f", experiment_id, metrics["test"]["pr_auc"])
    return metrics


def run_mlp_schedule(cfg: dict) -> int:
    if os.environ.get("M4_SKIP_MLP", "").strip() in ("1", "true", "yes"):
        logger.warning("M4_SKIP_MLP set — skipping MLP schedule")
        return 0
    trials = [("C_mlp_default", {})] + MLP_TRIALS
    for exp_id, overrides in trials:
        try:
            train_one_mlp(cfg, overrides, exp_id)
        except FileNotFoundError:
            logger.warning("Stage C missing — skip MLP")
            return 1
    return 0
