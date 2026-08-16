import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const config = {
  minPredictionDate: "2019-01-01",
  maxPredictionDate: "2026-08-17",
  timezone: "America/Los_Angeles",
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
    const input = await screen.findByLabelText("Prediction day");
    await waitFor(() => expect(input).toHaveValue("2026-08-17"));
    expect(screen.getByRole("button", { name: /tomorrow/i })).toHaveClass("is-selected");
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
    expect(await screen.findByText("Tomorrow’s data is not available yet")).toBeInTheDocument();
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
});
