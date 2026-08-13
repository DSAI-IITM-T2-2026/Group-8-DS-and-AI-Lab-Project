"""Load Milestone 3 modules by file path (avoid colliding top-level `src` packages)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load(name: str, path: Path) -> ModuleType:
    path = path.resolve()
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # Ensure package parent for relative imports inside M3 packages
    spec.loader.exec_module(mod)
    return mod


def _load_pkg_module(unique: str, file_path: Path, package_root: Path) -> ModuleType:
    """Load a module that does relative imports within package_root/src."""
    pkg_root = package_root.resolve()
    src_dir = pkg_root / "src"
    # Register fake package hierarchy so relative imports work
    pkg_name = unique
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [str(src_dir)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    mod_name = f"{pkg_name}.{file_path.stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(
        mod_name,
        file_path,
        submodule_search_locations=[str(src_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(mod_name)
    mod = importlib.util.module_from_spec(spec)
    # Point relative imports: from .cells → mvp_src.cells
    mod.__package__ = pkg_name
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_mvp_modules(cfg: dict) -> dict:
    root = Path(cfg["paths"]["mvp_era5_dem_root"])
    src = root / "src"
    # Preload dependency modules for relative imports
    for name in ("cells", "config", "era5_daily", "firms_labels", "assemble"):
        _load_pkg_module("mvp_src", src / f"{name}.py", root)
    return {
        "cells": sys.modules["mvp_src.cells"],
        "era5_daily": sys.modules["mvp_src.era5_daily"],
        "firms_labels": sys.modules["mvp_src.firms_labels"],
        "assemble": sys.modules["mvp_src.assemble"],
    }


def load_numerical_features(cfg: dict) -> ModuleType:
    root = Path(cfg["paths"]["multimodal_fusion_root"])
    return _load_pkg_module("mm_src", root / "src" / "numerical_features.py", root)
