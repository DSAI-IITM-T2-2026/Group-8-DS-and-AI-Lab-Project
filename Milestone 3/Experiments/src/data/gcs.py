"""GCS helpers and device utilities."""
from __future__ import annotations

import os
from typing import Optional

import gcsfs
import numpy as np
import torch

from src import config

os.environ["GS_NO_SIGN_REQUEST"] = "YES"

_fs: Optional[gcsfs.GCSFileSystem] = None


def get_fs() -> gcsfs.GCSFileSystem:
    global _fs
    if _fs is None:
        _fs = gcsfs.GCSFileSystem(token="anon")
    return _fs


def list_bucket_files(bucket: str, prefix: str, suffix: Optional[str] = None) -> list[str]:
    fs = get_fs()
    path = f"{bucket}/{prefix}"
    files = fs.ls(path)
    if suffix:
        files = [f for f in files if f.endswith(suffix)]
    return files


def to_device(arr, device=None):
    """Move array to DEVICE; fall back to CPU for unsupported MPS ops."""
    device = device or config.DEVICE
    t = torch.as_tensor(np.asarray(arr), dtype=torch.float32)
    try:
        return t.to(device)
    except Exception as exc:  # pragma: no cover
        msg = f"to_device fallback to CPU: {exc}"
        if msg not in config.MPS_FALLBACKS:
            config.MPS_FALLBACKS.append(msg)
        return t.to("cpu")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def safe_mps_op(fn, *args, **kwargs):
    """Run fn on DEVICE; on failure retry on CPU and log."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        if config.DEVICE.type != "mps":
            raise
        msg = f"MPS op fallback ({getattr(fn, '__name__', fn)}): {exc}"
        if msg not in config.MPS_FALLBACKS:
            config.MPS_FALLBACKS.append(msg)
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        cpu_kwargs = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()
        }
        out = fn(*cpu_args, **cpu_kwargs)
        if isinstance(out, torch.Tensor):
            try:
                return out.to(config.DEVICE)
            except Exception:
                return out
        return out
