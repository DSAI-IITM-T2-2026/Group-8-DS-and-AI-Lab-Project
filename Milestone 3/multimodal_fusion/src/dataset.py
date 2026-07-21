from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """Returns dict-like batch fields for flexible fusion training."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        seq_mean: np.ndarray,
        seq_std: np.ndarray,
        patches_dir: Path | None = None,
        s5p_patches_dir: Path | None = None,
        sequences_dir: Path | None = None,
        use_s2_patches: bool = True,
        use_s5p_patches: bool = True,
        use_s2_numerical: bool = False,
        use_s5p_numerical: bool = False,
        s2_num_cols: list[str] | None = None,
        s5p_num_cols: list[str] | None = None,
        s2_num_mean: np.ndarray | None = None,
        s2_num_std: np.ndarray | None = None,
        s5p_num_mean: np.ndarray | None = None,
        s5p_num_std: np.ndarray | None = None,
    ):
        self.df = manifest.reset_index(drop=True)
        self.seq_mean = seq_mean.astype(np.float32)
        self.seq_std = np.clip(seq_std.astype(np.float32), 1e-6, None)
        self.patches_dir = Path(patches_dir) if patches_dir else None
        self.s5p_patches_dir = Path(s5p_patches_dir) if s5p_patches_dir else None
        self.sequences_dir = Path(sequences_dir) if sequences_dir else None
        self.use_s2_patches = use_s2_patches
        self.use_s5p_patches = use_s5p_patches
        self.use_s2_numerical = use_s2_numerical
        self.use_s5p_numerical = use_s5p_numerical
        self.s2_num_cols = s2_num_cols or []
        self.s5p_num_cols = s5p_num_cols or []
        self.s2_num_mean = s2_num_mean
        self.s2_num_std = s2_num_std
        self.s5p_num_mean = s5p_num_mean
        self.s5p_num_std = s5p_num_std

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path_val, patches_dir: Path | None, reflectance_scale: bool) -> np.ndarray:
        p = Path(path_val)
        if not p.is_absolute() and patches_dir is not None:
            p = patches_dir / p.name
        image = np.load(p).astype(np.float32)
        if reflectance_scale and np.nanmax(image) > 2.0:
            image = np.clip(image, 0, 10000) / 10000.0
        return np.nan_to_num(image, nan=0.0)

    def _load_seq(self, row) -> np.ndarray:
        seq_path = Path(row["sequence_path"])
        if not seq_path.is_absolute() and self.sequences_dir is not None:
            seq_path = self.sequences_dir / seq_path.name
        seq = np.load(seq_path).astype(np.float32)
        seq = (seq - self.seq_mean) / self.seq_std
        return np.nan_to_num(seq, nan=0.0)

    def _load_vec(self, row, cols, mean, std) -> np.ndarray:
        x = row[cols].to_numpy(dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0)
        if mean is not None and std is not None:
            x = (x - mean) / np.clip(std, 1e-6, None)
        return x.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = torch.tensor(np.float32(row["y_fire"]))
        seq = torch.from_numpy(self._load_seq(row))
        out = {"seq": seq, "y": y}

        if self.use_s2_patches:
            out["s2_image"] = torch.from_numpy(
                self._load_image(row["patch_path"], self.patches_dir, reflectance_scale=True)
            )
        if self.use_s5p_patches:
            out["s5p_image"] = torch.from_numpy(
                self._load_image(
                    row["s5p_patch_path"], self.s5p_patches_dir, reflectance_scale=False
                )
            )
        if self.use_s2_numerical:
            out["s2_num"] = torch.from_numpy(
                self._load_vec(row, self.s2_num_cols, self.s2_num_mean, self.s2_num_std)
            )
        if self.use_s5p_numerical:
            out["s5p_num"] = torch.from_numpy(
                self._load_vec(row, self.s5p_num_cols, self.s5p_num_mean, self.s5p_num_std)
            )
        return out
