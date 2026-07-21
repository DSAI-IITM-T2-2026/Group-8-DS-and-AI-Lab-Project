from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    """Returns (image, seq[, s5p], y)."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        seq_mean: np.ndarray,
        seq_std: np.ndarray,
        patches_dir: Path | None = None,
        sequences_dir: Path | None = None,
        use_sentinel5p: bool = False,
        s5p_mean: float = 0.0,
        s5p_std: float = 1.0,
    ):
        self.df = manifest.reset_index(drop=True)
        self.seq_mean = seq_mean.astype(np.float32)
        self.seq_std = np.clip(seq_std.astype(np.float32), 1e-6, None)
        self.patches_dir = Path(patches_dir) if patches_dir else None
        self.sequences_dir = Path(sequences_dir) if sequences_dir else None
        self.use_sentinel5p = use_sentinel5p
        self.s5p_mean = float(s5p_mean)
        self.s5p_std = float(max(s5p_std, 1e-6))

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, row) -> np.ndarray:
        patch_path = Path(row["patch_path"])
        if not patch_path.is_absolute() and self.patches_dir is not None:
            patch_path = self.patches_dir / patch_path.name
        image = np.load(patch_path).astype(np.float32)
        if np.nanmax(image) > 2.0:
            image = np.clip(image, 0, 10000) / 10000.0
        return np.nan_to_num(image, nan=0.0)

    def _load_seq(self, row) -> np.ndarray:
        seq_path = Path(row["sequence_path"])
        if not seq_path.is_absolute() and self.sequences_dir is not None:
            seq_path = self.sequences_dir / seq_path.name
        seq = np.load(seq_path).astype(np.float32)  # [T, F]
        seq = (seq - self.seq_mean) / self.seq_std
        return np.nan_to_num(seq, nan=0.0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = torch.from_numpy(self._load_image(row))
        seq = torch.from_numpy(self._load_seq(row))
        y = torch.tensor(np.float32(row["y_fire"]))
        if not self.use_sentinel5p:
            return image, seq, y
        s5p_val = float(row.get("s5p_aerosol", 0.0))
        if not np.isfinite(s5p_val):
            s5p_val = self.s5p_mean
        s5p = np.float32((s5p_val - self.s5p_mean) / self.s5p_std)
        return image, seq, torch.tensor(s5p), y
