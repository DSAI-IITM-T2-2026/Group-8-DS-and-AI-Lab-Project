"""Score Stage C test rows with month router + fire_season fallback."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from numerical_nextday.train.router import bucket_for_month

logger = logging.getLogger(__name__)


def _predict_bundle(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    model, iso, feat = bundle["model"], bundle["iso"], bundle["feature_cols"]
    raw = model.predict(df[feat], num_iteration=model.best_iteration)
    return np.asarray(iso.predict(raw), dtype=float)


def load_bucket_bundles(art: Path, cfg: dict) -> dict[str, dict]:
    """Load available C_* / C_default joblibs for each bucket."""
    bundles: dict[str, dict] = {}
    fs_path = art / "models" / "fire_season" / "C_default.joblib"
    if not fs_path.exists():
        raise FileNotFoundError(f"Missing fire_season model: {fs_path}")
    bundles["fire_season"] = joblib.load(fs_path)

    for bucket in ["jan", "feb", "mar", "dec"]:
        bp = art / "models" / bucket / f"C_{bucket}.joblib"
        fb = art / "models" / bucket / "FALLBACK_fire_season.txt"
        if bp.exists():
            bundles[bucket] = joblib.load(bp)
        else:
            if fb.exists():
                logger.info("Bucket %s uses fire_season fallback", bucket)
            bundles[bucket] = bundles["fire_season"]
    return bundles


def score_test_frame(cfg: dict, test: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return test rows with: label_date, lat/lon, y_fire, proba, confidence_pct, bucket.
    Uses month-router models when available.
    """
    from numerical_nextday.data.s2_s5p import _lag_prefix

    art = Path(cfg["paths"]["artifacts_dir"])
    if test is None:
        test = pd.read_parquet(_lag_prefix(cfg) / "stage_c" / "test.parquet")
    out = test.copy()
    out["label_date"] = pd.to_datetime(out["label_date"])
    bundles = load_bucket_bundles(art, cfg)
    out["bucket"] = out["label_date"].dt.month.map(lambda m: bucket_for_month(int(m), cfg))

    proba_s = pd.Series(index=out.index, dtype=float)
    for bucket, idxs in out.groupby("bucket").groups.items():
        b = bundles.get(str(bucket), bundles["fire_season"])
        proba_s.loc[idxs] = _predict_bundle(b, out.loc[idxs])

    out["proba"] = proba_s.astype(float).to_numpy()
    out["confidence_pct"] = out["proba"] * 100.0
    return out
