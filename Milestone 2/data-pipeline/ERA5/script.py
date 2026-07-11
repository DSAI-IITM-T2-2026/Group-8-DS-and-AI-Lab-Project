#!/usr/bin/env python3
"""Legacy entry point — use run_pipeline.py instead."""

import runpy

if __name__ == "__main__":
    runpy.run_path("run_pipeline.py", run_name="__main__")
