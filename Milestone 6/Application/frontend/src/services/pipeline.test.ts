import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpPipelineService, PipelineApiError } from "./pipeline";

describe("HttpPipelineService", () => {
  afterEach(() => vi.restoreAllMocks());

  it("creates a run using the public prediction-date contract", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ runId: "run-1", predictionDate: "2025-08-01", status: "queued" }), { status: 202 }),
    );
    await new HttpPipelineService("/api/v1").createRun("2025-08-01");
    expect(fetchMock).toHaveBeenCalledWith("/api/v1/pipeline-runs", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ predictionDate: "2025-08-01" }),
    }));
  });

  it("surfaces safe field errors from the backend", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: { code: "unsupported_prediction_date", message: "Choose a supported date.", fieldErrors: { predictionDate: "Outside range." } } }), { status: 422 }),
    );
    await expect(new HttpPipelineService("/api/v1").createRun("2016-11-21")).rejects.toMatchObject({
      code: "unsupported_prediction_date",
      fieldErrors: { predictionDate: "Outside range." },
    } satisfies Partial<PipelineApiError>);
  });

  it("cancels an active run through the public run endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ runId: "run-1", status: "interrupted", errorCode: "cancelled_by_user" }), { status: 200 }),
    );

    await new HttpPipelineService("/api/v1").cancelRun("run-1");

    expect(fetchMock).toHaveBeenCalledWith("/api/v1/pipeline-runs/run-1/cancel", expect.objectContaining({ method: "POST" }));
  });
});
