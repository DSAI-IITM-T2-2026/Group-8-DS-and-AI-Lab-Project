#!/usr/bin/env python3
"""Write ROC/PR and related metric comparison charts into artifacts/figures/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    from numerical_nextday.config import load_config
    from numerical_nextday.eval.metrics_charts import write_metrics_charts

    cfg = load_config(args.config)
    paths = write_metrics_charts(cfg)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
