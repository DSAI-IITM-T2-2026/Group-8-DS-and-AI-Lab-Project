"""Causal EO attach: latest window_end <= D (no covering-window lookahead)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def attach_causal_window_end(
    samples: pd.DataFrame,
    window_tables: list[pd.DataFrame],
    value_cols: list[str],
    date_col: str = "eo_asof_date",
    max_lag_days: int | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """
    For each sample day D, attach features from the latest window with
    window_end <= D. Optionally reject if (D - window_end) > max_lag_days.
    """
    out_cols = [prefix + c for c in value_cols] + [prefix + "available", prefix + "lag_days"]
    if not window_tables:
        out = samples.copy()
        for c in value_cols:
            out[prefix + c] = np.nan
        out[prefix + "available"] = 0
        out[prefix + "lag_days"] = np.nan
        return out

    panels = []
    for w in window_tables:
        cols = [c for c in value_cols if c in w.columns]
        if "cell_id" not in w.columns or "window_end" not in w.columns:
            continue
        part = w[["cell_id", "window_end"] + cols].copy()
        panels.append(part)
    if not panels:
        out = samples.copy()
        for c in value_cols:
            out[prefix + c] = np.nan
        out[prefix + "available"] = 0
        out[prefix + "lag_days"] = np.nan
        return out

    panel = pd.concat(panels, ignore_index=True)
    panel["window_end"] = pd.to_datetime(panel["window_end"]).dt.normalize()
    panel = panel.sort_values(["cell_id", "window_end"]).drop_duplicates(
        ["cell_id", "window_end"], keep="last"
    )

    samples = samples.copy()
    samples[date_col] = pd.to_datetime(samples[date_col]).dt.normalize()

    # merge_asof per cell is slow; use searchsorted per cell (same as M3 but causal-only)
    attached_rows: list[dict] = []
    for cell_id, grp in samples.groupby("cell_id", sort=False):
        cell_panel = panel.loc[panel["cell_id"] == cell_id]
        if cell_panel.empty:
            for _, row in grp.iterrows():
                rec = row.to_dict()
                for c in value_cols:
                    rec[prefix + c] = np.nan
                rec[prefix + "available"] = 0
                rec[prefix + "lag_days"] = np.nan
                attached_rows.append(rec)
            continue

        ends = cell_panel["window_end"].to_numpy()
        for _, row in grp.iterrows():
            D = row[date_col]
            # latest window_end <= D
            idx = int(np.searchsorted(ends, np.datetime64(D), side="right") - 1)
            if idx < 0:
                rec = row.to_dict()
                for c in value_cols:
                    rec[prefix + c] = np.nan
                rec[prefix + "available"] = 0
                rec[prefix + "lag_days"] = np.nan
                attached_rows.append(rec)
                continue
            wrow = cell_panel.iloc[idx]
            lag = int((D - wrow["window_end"]).days)
            if max_lag_days is not None and lag > max_lag_days:
                rec = row.to_dict()
                for c in value_cols:
                    rec[prefix + c] = np.nan
                rec[prefix + "available"] = 0
                rec[prefix + "lag_days"] = lag
                attached_rows.append(rec)
                continue
            rec = row.to_dict()
            for c in value_cols:
                rec[prefix + c] = wrow[c] if c in wrow.index else np.nan
            rec[prefix + "available"] = 1
            rec[prefix + "lag_days"] = lag
            attached_rows.append(rec)

    result = pd.DataFrame(attached_rows)
    logger.info(
        "Causal attach prefix=%s available=%s / %s",
        prefix,
        int(result[prefix + "available"].sum()) if prefix + "available" in result.columns else 0,
        len(result),
    )
    # silence unused
    _ = out_cols
    return result


def apply_train_median_fill(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, ...]:
    """Fill NaNs in cols using train medians; apply same values to all frames."""
    medians: dict[str, float] = {}
    for c in cols:
        if c not in train.columns:
            continue
        medians[c] = float(np.nanmedian(train[c].to_numpy(dtype=float)))

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, m in medians.items():
            if c in out.columns:
                out[c] = out[c].fillna(m)
        return out

    return (_fill(train),) + tuple(_fill(df) for df in others)
