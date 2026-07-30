from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from numerical_nextday.train.calibration import fit_platt_calibrator
from numerical_nextday.train.preprocessing import TabularPreprocessor


def test_imputation_is_fitted_only_from_training_data() -> None:
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [10.0, 20.0, 30.0]})
    validation = pd.DataFrame({"x": [1000.0, np.nan], "y": [40.0, np.nan]})
    processor = TabularPreprocessor(["x", "y"]).fit(train)
    transformed = processor.transform(validation)
    assert processor.medians.to_dict() == {"x": 2.0, "y": 20.0}
    assert transformed[1].tolist() == [2.0, 20.0]


def test_platt_calibration_preserves_probability_order() -> None:
    raw = np.array([0.01, 0.05, 0.2, 0.7, 0.9, 0.95])
    target = np.array([0, 0, 0, 1, 1, 1])
    calibrator = fit_platt_calibrator(raw, target, min_positives=1)
    calibrated = calibrator.predict(raw)
    assert np.all(np.diff(calibrated) > 0)
    assert average_precision_score(target, raw) == average_precision_score(
        target, calibrated
    )
