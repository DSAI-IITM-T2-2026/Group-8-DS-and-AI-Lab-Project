import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useInference } from "./useInference";

const geometry = {
  regionId: "california",
  geometryVersion: "v1",
  geojson: { type: "FeatureCollection", features: [] },
} as const;

const riskMap = {
  regionId: "california",
  timestamp: "2025-08-01T12:00:00Z",
  geometryVersion: "v1",
  provenance: "model",
  items: [{
    areaId: "cell-a", areaName: "Grid cell cell-a", probability: 0.2,
    rawProbability: 0.3, alertScore: 0.9, priorityRank: 1,
    alertTop25: true, riskClass: "very_high", updatedAt: "2025-08-01T12:00:00Z",
  }],
} as const;

const validation = {
  status: "available", date: "2025-08-01", modelVersion: "model-v1",
  labelSource: "historical_archive", message: "Ready", provenance: "firms_observation",
  items: [], summary: { observedFireCells: 0, capturedInTop25: 0, recallAt25: null, precisionAt25: 0, falseAlerts: 1, top25Count: 1 },
} as const;

describe("useInference", () => {
  it("automatically scores and loads the highest-priority cell", async () => {
    const service = {
      getGeometry: vi.fn().mockResolvedValue(geometry),
      getRiskMap: vi.fn().mockResolvedValue(riskMap),
      getDailyValidation: vi.fn().mockResolvedValue(validation),
      getPrediction: vi.fn().mockResolvedValue({ regionId: "cell-a" }),
    } as any;
    const { result } = renderHook(() => useInference(service, "2025-08-01"));

    await waitFor(() => expect(result.current.riskMap?.items[0].areaId).toBe("cell-a"));
    await waitFor(() => expect(result.current.prediction?.regionId).toBe("cell-a"));
    await waitFor(() => expect(result.current.validation?.status).toBe("available"));
    expect(service.getRiskMap).toHaveBeenCalledWith("2025-08-01");
    expect(service.getPrediction).toHaveBeenCalledWith("cell-a", "2025-08-01");
    expect(service.getDailyValidation).toHaveBeenCalledWith("2025-08-01");
  });

  it("keeps inference errors separate from pipeline state", async () => {
    const service = {
      getGeometry: vi.fn().mockResolvedValue(geometry),
      getRiskMap: vi.fn().mockRejectedValue(new Error("Model unavailable")),
      getDailyValidation: vi.fn().mockResolvedValue(validation),
      getPrediction: vi.fn(),
    } as any;
    const { result } = renderHook(() => useInference(service, "2025-08-01"));
    await waitFor(() => expect(result.current.error?.message).toBe("Model unavailable"));
    expect(result.current.riskMap).toBeUndefined();
  });

  it("publishes the initial map, validation, and cell detail together", async () => {
    let resolvePrediction: (value: { regionId: string }) => void = () => undefined;
    const detail = new Promise<{ regionId: string }>((resolve) => { resolvePrediction = resolve; });
    const service = {
      getGeometry: vi.fn().mockResolvedValue(geometry),
      getRiskMap: vi.fn().mockResolvedValue(riskMap),
      getDailyValidation: vi.fn().mockResolvedValue(validation),
      getPrediction: vi.fn().mockReturnValue(detail),
    } as any;
    const { result } = renderHook(() => useInference(service, "2025-08-01"));

    await waitFor(() => expect(service.getPrediction).toHaveBeenCalled());
    expect(result.current.isLoading).toBe(true);
    expect(result.current.riskMap).toBeUndefined();
    expect(result.current.validation).toBeUndefined();

    await act(async () => resolvePrediction({ regionId: "cell-a" }));
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.riskMap).toBe(riskMap);
    expect(result.current.validation).toBe(validation);
    expect(result.current.prediction?.regionId).toBe("cell-a");
  });
});
