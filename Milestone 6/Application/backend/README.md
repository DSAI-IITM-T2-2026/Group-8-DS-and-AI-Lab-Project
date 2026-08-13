# Wildfire IQ Pipeline and Inference API

The single public FastAPI service for launching `Milestone 6/Application/daily_pipeline` and scoring its validated parquet through `Milestone 6/Application/api`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
```

The worker invokes `Milestone 6/Application/daily_pipeline/run_daily.py all --label-date YYYY-MM-DD`, stores run state in SQLite under `.state/`, and writes per-run logs under `.state/logs/`. Configure the ignored `WILDFIRE_MODEL_URI` with the champion model's `gs://` object and enable `WILDFIRE_ALLOW_GCS`. The backend streams both the model and `final_processed/YYYY-MM-DD_test.parquet` into memory; neither needs to be copied into the deployed filesystem. The runtime identity needs object read access and the pipeline additionally needs object create/update access. Credentials and artifacts are never exposed to the browser.

Use Python 3.12 with the pinned numerical packages in `requirements.txt`; these versions match the model's training manifest.
