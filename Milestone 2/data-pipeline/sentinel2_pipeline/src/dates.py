"""
Date windowing.

Generates the (start, end, label) windows the pipeline steps through. This
logic is duplicated (not shared) in the sentinel5p_pipeline folder on
purpose — the two pipelines are meant to be independently cloneable, so
neither imports from the other.
"""

from datetime import datetime, timedelta
from typing import Iterator, Tuple


def descending_date_steps(cfg: dict) -> Iterator[Tuple[str, str, str]]:
    """
    Yields (start_date, end_date, label) tuples stepping backward in
    `step_days`-sized windows from start_year Dec 31 down to end_year Jan 1.

    Config uses inverted year names on purpose: start_year is the newest year,
    end_year is the oldest. load_config() rejects start_year < end_year.

    `label` is the window's END date, formatted YYYY_MM_DD — this matches
    the naming convention verified in the production GCS bucket.
    GEE filterDate end is exclusive, so the yielded end is cursor_end + 1 day.
    """
    t = cfg["temporal"]
    step_days = t["step_days"]
    range_start = datetime(t["end_year"], 1, 1)
    range_end = datetime(t["start_year"], 12, 31)

    cursor_end = range_end
    while cursor_end >= range_start:
        cursor_start = max(range_start, cursor_end - timedelta(days=step_days - 1))
        label = cursor_end.strftime("%Y_%m_%d")
        yield (
            cursor_start.strftime("%Y-%m-%d"),
            (cursor_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            label,
        )
        cursor_end = cursor_start - timedelta(days=1)
