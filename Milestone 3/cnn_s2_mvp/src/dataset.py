from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WildfirePatchDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_cols: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        patches_dir: Path | None = None,
    ):
        self.df = manifest.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.mean = mean.astype(np.float32)
        self.std = np.clip(std.astype(np.float32), 1e-6, None)
        self.patches_dir = Path(patches_dir) if patches_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patch_path = Path(row["patch_path"])
        if not patch_path.is_absolute() and self.patches_dir is not None:
            patch_path = self.patches_dir / patch_path.name
        image = np.load(patch_path).astype(np.float32)
        # Per-band simple scale (reflectance-ish): clip and /10000 if large
        if np.nanmax(image) > 2.0:
            image = np.clip(image, 0, 10000) / 10000.0
        image = np.nan_to_num(image, nan=0.0)

        tab = row[self.feature_cols].to_numpy(dtype=np.float32)
        tab = (tab - self.mean) / self.std
        tab = np.nan_to_num(tab, nan=0.0)

        y = np.float32(row["y_fire"])
        return (
            torch.from_numpy(image),
            torch.from_numpy(tab),
            torch.tensor(y),
        )
