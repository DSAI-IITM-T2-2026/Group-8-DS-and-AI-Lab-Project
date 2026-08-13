import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpInferenceService, InferenceApiError } from "./inference";

describe("HttpInferenceService", () => {
  afterEach(() => vi.restoreAllMocks());

  it("requests the complete daily risk map", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ items: [] }), { status: 200 }));
    const service = new HttpInferenceService("/api/v1");
    await service.getRiskMap("2025-08-01");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/risk-map?region_id=california&timestamp=2025-08-01T12%3A00%3A00Z&mode=forecast_24h",
      expect.objectContaining({ headers: expect.objectContaining({ Accept: "application/json" }) }),
    );
  });

  it("preserves safe inference error codes", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ code: "model_unavailable", message: "Model is not loaded." }), { status: 503 }));
    const service = new HttpInferenceService("/api/v1");
    await expect(service.getGeometry()).rejects.toEqual(expect.objectContaining<Partial<InferenceApiError>>({ code: "model_unavailable" }));
  });
});
