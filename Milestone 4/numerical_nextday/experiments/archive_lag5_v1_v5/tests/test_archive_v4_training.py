from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_archive_v4.py"
    spec = importlib.util.spec_from_file_location("train_archive_v4", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fast_recall_at_k_respects_daily_groups() -> None:
    module = _module()
    target = np.array([1, 0, 0, 0, 1, 0], dtype="int8")
    score = np.array([0.9, 0.8, 0.1, 0.9, 0.8, 0.1])
    groups = np.array([3, 3])
    assert module.fast_recall_at_k(target, score, groups, 1) == 0.5
    assert module.fast_recall_at_k(target, score, groups, 2) == 1.0


def test_recall25_target_is_fifty_percent() -> None:
    module = _module()
    assert module.TARGET_RECALL_25 == 0.50
