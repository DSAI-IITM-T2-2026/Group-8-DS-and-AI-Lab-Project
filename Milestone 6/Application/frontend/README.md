# Wildfire IQ Prediction Data Studio

Focused React/TypeScript workflow for preparing a prediction date, automatically scoring the validated daily parquet, and inspecting the California priority grid.

## Run locally

```bash
npm install
npm run dev
```

Quality checks:

```bash
npm run typecheck
npm test
npm run build
npm run test:sites
```

## Local API

The development server proxies `/api` to the FastAPI backend on port 8000. `VITE_API_BASE_URL` can point a production build at another API origin. The browser never receives GCP or Earth Engine credentials.

The application prepares and validates `final_processed/YYYY-MM-DD_test.parquet`, then calls the unified backend for calibrated probability, blended daily priority, top-25 alerts, and per-cell details.
