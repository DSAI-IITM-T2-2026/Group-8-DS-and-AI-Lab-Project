from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .plots import plot_importance


def write_lightgbm_explanations(
    bundle,
    frame: pd.DataFrame,
    artifact_dir: Path,
    max_rows: int = 2000,
    top_local_rows: int = 25,
    top_local_features: int = 8,
) -> dict[str, str]:
    """Write global and local TreeSHAP values using LightGBM's native implementation."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sample = frame
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=bundle.random_seed)
    matrix = bundle.preprocessor.transform(sample[bundle.feature_columns])
    contributions = np.asarray(
        bundle.booster.predict(
            matrix,
            pred_contrib=True,
            num_iteration=bundle.best_iteration or bundle.booster.best_iteration,
        )
    )
    values = contributions[:, :-1]
    global_table = pd.DataFrame(
        {
            "feature": bundle.feature_columns,
            "mean_abs_contribution": np.abs(values).mean(axis=0),
            "mean_contribution": values.mean(axis=0),
            "gain_importance": bundle.booster.feature_importance(importance_type="gain"),
            "split_importance": bundle.booster.feature_importance(importance_type="split"),
        }
    ).sort_values("mean_abs_contribution", ascending=False)
    global_path = artifact_dir / "global_feature_explanations.csv"
    global_table.to_csv(global_path, index=False)
    plot_importance(global_table, artifact_dir / "global_feature_explanations.png")

    probabilities = bundle.predict_proba(sample)
    ranked = np.argsort(probabilities)[::-1][:top_local_rows]
    local_records = []
    identity_columns = [
        column
        for column in ("label_date", "cell_id", "latitude", "longitude", "y_fire")
        if column in sample
    ]
    for rank, row_position in enumerate(ranked, start=1):
        order = np.argsort(np.abs(values[row_position]))[::-1][:top_local_features]
        identity = sample.iloc[row_position][identity_columns].to_dict()
        for feature_rank, feature_position in enumerate(order, start=1):
            local_records.append(
                {
                    "alert_rank": rank,
                    "feature_rank": feature_rank,
                    "p_fire": float(probabilities[row_position]),
                    "feature": bundle.feature_columns[feature_position],
                    "feature_value": float(matrix[row_position, feature_position]),
                    "contribution_log_odds": float(values[row_position, feature_position]),
                    **identity,
                }
            )
    local_path = artifact_dir / "top_alert_local_explanations.csv"
    pd.DataFrame(local_records).to_csv(local_path, index=False)
    return {"global": str(global_path), "local": str(local_path)}
