"""Lag-aware tile dataset for LagFireNet training / eval."""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config
from .features import build_features_for_label_day
from .labels import load_or_build_label
from .regrid import build_land_mask, reference_da

logger = logging.getLogger(__name__)

# Keep only a few full-day tensors in RAM — never dump all days to disk.
_MAX_MEM_DAYS = 3


def list_label_dates(start: str, end: str, fire_season: bool = False) -> list[str]:
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]
    if fire_season:
        dates = [
            d
            for d in dates
            if pd.Timestamp(d).month in config.FIRE_SEASON_MONTHS
        ]
    return dates


def split_dates(split: str, fire_season: bool = False) -> list[str]:
    if split == "train":
        return list_label_dates(config.TRAIN_START, config.TRAIN_END, fire_season=fire_season)
    if split == "val":
        return list_label_dates(config.VAL_START, config.VAL_END, fire_season=fire_season)
    if split == "test":
        return list_label_dates(config.TEST_START, config.TEST_END, fire_season=fire_season)
    raise ValueError(split)


class LiveWildfireDataset(Dataset):
    """
    Random 256×256 crops.

    Disk: only compact FIRMS labels (``firms/labels/*.npy``).
    RAM: LRU of a few full-day feature stacks (avoids multi‑10GB day cache).
    """

    def __init__(
        self,
        dates: list[str],
        split: str = "train",
        tile_size: int = config.TILE_SIZE,
        crops_per_day: int = 8,
        fire_oversample: float = 0.5,
        seed: int = config.SEED,
        preload_index: bool = True,
        mem_days: int = _MAX_MEM_DAYS,
    ):
        self.dates = dates
        self.split = split
        self.tile_size = tile_size
        self.crops_per_day = crops_per_day
        self.fire_oversample = fire_oversample
        self.rng = np.random.default_rng(seed)
        self.mem_days = max(1, mem_days)
        self._feat_lru: OrderedDict[str, tuple] = OrderedDict()

        ref = reference_da()
        self.H = int(ref.sizes["y"])
        self.W = int(ref.sizes["x"])
        self.land = build_land_mask(ref)

        self.samples: list[tuple[str, int, int]] = []
        if preload_index:
            self._rebuild_index()

    def _label_ok(self, date_str: str) -> np.ndarray | None:
        try:
            return load_or_build_label(date_str)
        except Exception as e:
            logger.warning("Label failed %s: %s", date_str, e)
            return None

    def _get_day_arrays(self, date_str: str):
        if date_str in self._feat_lru:
            self._feat_lru.move_to_end(date_str)
            return self._feat_lru[date_str]
        feats = build_features_for_label_day(date_str)
        if feats is None:
            raise RuntimeError(f"features unavailable: {date_str}")
        label = load_or_build_label(date_str)
        # float16 on disk-ish memory footprint while in RAM
        pack = (
            feats.era5.astype(np.float16, copy=False),
            feats.s5p.astype(np.float16, copy=False),
            feats.s2.astype(np.float16, copy=False),
            feats.dem.astype(np.float16, copy=False),
            label.astype(np.uint8, copy=False),
        )
        self._feat_lru[date_str] = pack
        while len(self._feat_lru) > self.mem_days:
            self._feat_lru.popitem(last=False)
        return pack

    def _rebuild_index(self) -> None:
        """Index uses labels only (cheap) — does not build full feature stacks."""
        self.samples = []
        ts = self.tile_size
        n_ok = 0
        for date_str in self.dates:
            label = self._label_ok(date_str)
            if label is None:
                continue
            n_ok += 1
            fire_yx = np.argwhere((label > 0) & self.land)
            for _ in range(self.crops_per_day):
                if self.split == "train" and len(fire_yx) and self.rng.random() < self.fire_oversample:
                    r, c = fire_yx[self.rng.integers(0, len(fire_yx))]
                    top = int(np.clip(r - ts // 2, 0, max(0, self.H - ts)))
                    left = int(np.clip(c - ts // 2, 0, max(0, self.W - ts)))
                else:
                    top = int(self.rng.integers(0, max(1, self.H - ts + 1)))
                    left = int(self.rng.integers(0, max(1, self.W - ts + 1)))
                self.samples.append((date_str, top, left))
        self.rng.shuffle(self.samples)
        logger.info(
            "%s dataset: %d tiles from %d/%d dates (label-only index; features on demand)",
            self.split,
            len(self.samples),
            n_ok,
            len(self.dates),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not self.samples:
            self._rebuild_index()
        if not self.samples:
            raise RuntimeError(
                f"{self.split} dataset has 0 tiles — labels/features failed for all dates."
            )
        date_str, top, left = self.samples[idx % len(self.samples)]
        era5, s5p, s2, dem, label = self._get_day_arrays(date_str)
        ts = self.tile_size
        sl_y = slice(top, top + ts)
        sl_x = slice(left, left + ts)
        return {
            "era5": torch.from_numpy(np.asarray(era5[:, :, sl_y, sl_x], dtype=np.float32)),
            "s5p": torch.from_numpy(np.asarray(s5p[:, :, sl_y, sl_x], dtype=np.float32)),
            "s2": torch.from_numpy(np.asarray(s2[:, sl_y, sl_x], dtype=np.float32)),
            "dem": torch.from_numpy(np.asarray(dem[:, sl_y, sl_x], dtype=np.float32)),
            "label": torch.from_numpy(
                np.asarray(label[sl_y, sl_x], dtype=np.float32)
            ).unsqueeze(0),
            "date": date_str,
            "top": top,
            "left": left,
        }


def compute_norm_stats(dataset: LiveWildfireDataset, max_batches: int = 200) -> dict[str, np.ndarray]:
    """Mean/std over a subset of training tiles."""
    if len(dataset) == 0:
        raise RuntimeError(
            "Cannot compute norm stats: dataset is empty (no valid feature days)."
        )
    sums = {
        "era5": np.zeros(config.ERA5_CHANNELS, dtype=np.float64),
        "s5p": np.zeros(config.S5P_CHANNELS, dtype=np.float64),
        "s2": np.zeros(config.S2_CHANNELS, dtype=np.float64),
        "dem": np.zeros(config.DEM_CHANNELS, dtype=np.float64),
    }
    sq = {k: np.zeros_like(v) for k, v in sums.items()}
    counts = {k: 0 for k in sums}

    n = min(len(dataset), max_batches)
    for i in range(n):
        item = dataset[i]
        for key in sums:
            x = item[key].numpy()
            if key in ("era5", "s5p"):
                c = x.shape[1]
                flat = x.transpose(1, 0, 2, 3).reshape(c, -1)
            else:
                c = x.shape[0]
                flat = x.reshape(c, -1)
            sums[key] += flat.mean(axis=1)
            sq[key] += (flat ** 2).mean(axis=1)
            counts[key] += 1

    stats = {}
    for key in sums:
        m = sums[key] / max(counts[key], 1)
        second = sq[key] / max(counts[key], 1)
        var = np.maximum(second - m ** 2, 1e-6)
        stats[f"{key}_mean"] = m.astype(np.float32)
        stats[f"{key}_std"] = np.sqrt(var).astype(np.float32)
    return stats


def apply_norm(batch: dict, stats: dict) -> dict:
    out = dict(batch)
    for key in ("era5", "s5p", "s2", "dem"):
        mean = torch.as_tensor(stats[f"{key}_mean"], dtype=out[key].dtype)
        std = torch.as_tensor(stats[f"{key}_std"], dtype=out[key].dtype)
        x = out[key]
        if key in ("era5", "s5p"):
            shape = [1] * x.dim()
            shape[-3] = -1
            mean = mean.view(*shape)
            std = std.view(*shape)
        else:
            if x.dim() == 3:
                mean = mean.view(-1, 1, 1)
                std = std.view(-1, 1, 1)
            else:
                mean = mean.view(1, -1, 1, 1)
                std = std.view(1, -1, 1, 1)
        out[key] = (x - mean) / std.clamp_min(1e-6)
    return out
