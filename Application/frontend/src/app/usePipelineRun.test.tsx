import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { usePipelineRun } from "./usePipelineRun";
import type { PipelineRun } from "../domain/pipeline";

const config = { minPredictionDate: "2019-01-01", maxPredictionDate: "2026-08-13", timezone: "America/Los_Angeles", lookbackDays: 30, expectedFeatureCount: 86 };
const queued: PipelineRun = { runId: "run-1", predictionDate: "2025-08-01", status: "queued", stage: "validating", message: "Queued", progressCompleted: 0, progressTotal: 0, sourceInventory: {}, createdAt: "2026-08-13T00:00:00Z" };

describe("usePipelineRun", () => {
  beforeEach(() => localStorage.clear());

  it("loads date constraints and preserves an active run id", async () => {
    const service = {
      getConfig: vi.fn().mockResolvedValue(config),
      getRun: vi.fn().mockResolvedValue(queued),
      createRun: vi.fn().mockResolvedValue(queued),
    } as any;
    const { result } = renderHook(() => usePipelineRun(service));
    await waitFor(() => expect(result.current.selectedDate).toBe(config.maxPredictionDate));
    act(() => result.current.selectDate("2025-08-01"));
    await act(async () => result.current.start());
    expect(localStorage.getItem("wildfire-iq-pipeline-run")).toBe("run-1");
  });

  it("restores an active run after reload", async () => {
    localStorage.setItem("wildfire-iq-pipeline-run", "run-1");
    const service = {
      getConfig: vi.fn().mockResolvedValue(config),
      getRun: vi.fn().mockResolvedValue(queued),
      createRun: vi.fn(),
    } as any;
    const { result } = renderHook(() => usePipelineRun(service));
    await waitFor(() => expect(result.current.run?.runId).toBe("run-1"));
    expect(service.getRun).toHaveBeenCalledWith("run-1");
  });
});
