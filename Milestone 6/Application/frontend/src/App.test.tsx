import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const config = {
  minPredictionDate: "2019-01-01",
  maxPredictionDate: "2026-08-17",
  timezone: "America/Los_Angeles",
  cutoffLocalTime: "06:30",
  lookbackDays: 30,
  expectedFeatureCount: 86,
};

const evaluation = {
  evaluationVersion: "milestone-5-champion-2025-v1",
  modelVersion: "champion-v1",
  split: "Held-out 2025 test set",
  labelYear: 2025,
  rows: 93518,
  positives: 1325,
  baseline: { label: "Naive constant-rate baseline", prAuc: 0.0093 },
  metrics: [],
  provenance: "held_out_evaluation",
};

function jsonResponse(value: unknown) {
  return Promise.resolve(new Response(JSON.stringify(value), { status: 200 }));
}

describe("App tomorrow workflow", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => vi.restoreAllMocks());

  it("selects tomorrow from the backend config by default", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/pipeline/config")) return jsonResponse(config);
      if (url.endsWith("/model/evaluation")) return jsonResponse(evaluation);
      throw new Error(`Unexpected request: ${url}`);
    });

    render(<App />);
    const input = await screen.findByLabelText("Prediction day (California time)");
    await waitFor(() => expect(input).toHaveValue("2026-08-17"));
    expect(screen.getByRole("button", { name: /tomorrow/i })).toHaveClass("is-selected");
    expect(screen.getByText("All prediction dates and data cutoffs use California time.")).toBeInTheDocument();
    expect(screen.getByText("California now")).toBeInTheDocument();
    expect(screen.getByText(/Data cutoff: 06:30 California time/)).toBeInTheDocument();
  });

  it("renders missing tomorrow data as a neutral unavailable state", async () => {
    localStorage.setItem("wildfire-iq-pipeline-run", "run-1");
    const unavailableRun = {
      runId: "run-1",
      predictionDate: "2026-08-17",
      status: "unavailable",
      stage: "inventory",
      message: "Tomorrow's data is not available yet.",
      progressCompleted: 0,
      progressTotal: 0,
      sourceInventory: {},
      createdAt: "2026-08-16T00:00:00Z",
      finishedAt: "2026-08-16T00:00:01Z",
    };
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/pipeline/config")) return jsonResponse(config);
      if (url.endsWith("/model/evaluation")) return jsonResponse(evaluation);
      if (url.endsWith("/pipeline-runs/run-1")) return jsonResponse(unavailableRun);
      throw new Error(`Unexpected request: ${url}`);
    });

    render(<App />);
    expect(await screen.findByText("Provisional tomorrow forecast is unavailable")).toBeInTheDocument();
    expect(screen.queryByText("Preparation needs attention")).not.toBeInTheDocument();
    expect(screen.queryByText("Prediction results")).not.toBeInTheDocument();
  });

  it("stops an active forecast and renders a neutral terminal state", async () => {
    localStorage.setItem("wildfire-iq-pipeline-run", "run-1");
    const runningRun = {
      runId: "run-1", predictionDate: "2026-08-16", status: "running", stage: "era5",
      message: "Ensuring ERA5 history.", progressCompleted: 3, progressTotal: 38,
      sourceInventory: {}, createdAt: "2026-08-16T00:00:00Z", startedAt: "2026-08-16T00:00:01Z",
    };
    const stoppedRun = {
      ...runningRun, status: "interrupted", message: "Forecast preparation was stopped by the user.",
      errorCode: "cancelled_by_user", finishedAt: "2026-08-16T00:01:00Z",
    };
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/pipeline/config")) return jsonResponse(config);
      if (url.endsWith("/model/evaluation")) return jsonResponse(evaluation);
      if (url.endsWith("/pipeline-runs/run-1/cancel")) return jsonResponse(stoppedRun);
      if (url.endsWith("/pipeline-runs/run-1")) return jsonResponse(runningRun);
      throw new Error(`Unexpected request: ${url}`);
    });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Stop forecast" }));

    expect(await screen.findByText("Forecast stopped")).toBeInTheDocument();
    expect(screen.getByText("stopped")).toBeInTheDocument();
    expect(screen.queryByText("Preparation needs attention")).not.toBeInTheDocument();
  });

  it("clearly labels a successful one-day ERA5 fallback", async () => {
    localStorage.setItem("wildfire-iq-pipeline-run", "run-1");
    const fallbackRun = {
      runId: "run-1", predictionDate: "2026-08-17", status: "succeeded", stage: "completed",
      message: "Prediction data is ready.", progressCompleted: 1, progressTotal: 1,
      sourceInventory: {}, createdAt: "2026-08-16T14:00:00Z", finishedAt: "2026-08-16T14:01:00Z",
      artifact: {
        objectUri: "gs://test/final_processed/2026-08-17_test.parquet", rowCount: 437,
        featureCount: 86, cellCount: 437, labelDate: "2026-08-17", eoAsOfDate: "2026-08-16",
        featureEndDate: "2026-08-10", requiredFeatureEndDate: "2026-08-11",
        createdAt: "2026-08-16T14:01:00Z", forecastMode: "provisional_tomorrow",
        artifactQuality: "era5_fallback", needsRefresh: true,
        sourceSnapshots: { era5: { required: 38, available: 37, missing: 1, scheduled: 0, pending: 0, ready: true, requiredThroughDate: "2026-08-11", selectedThroughDate: "2026-08-10", ageDays: 1, mode: "latest_causal", exactAvailable: false } },
      },
    };
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/pipeline/config")) return jsonResponse(config);
      if (url.endsWith("/model/evaluation")) return jsonResponse(evaluation);
      if (url.endsWith("/pipeline-runs/run-1")) return jsonResponse(fallbackRun);
      if (url.includes("/risk-map") || url.includes("/risk-cells") || url.includes("/validation")) return jsonResponse({ code: "not_ready" });
      throw new Error(`Unexpected request: ${url}`);
    });

    render(<App />);

    expect((await screen.findAllByText("Weather fallback forecast")).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/one day older than the required 2026-08-11 endpoint/)).toBeInTheDocument();
    expect(screen.getByText(/will be regenerated when exact ERA5 data becomes available/)).toBeInTheDocument();
    expect(screen.getByText(/Required 2026-08-11 · selected 2026-08-10/)).toBeInTheDocument();
  });
});
