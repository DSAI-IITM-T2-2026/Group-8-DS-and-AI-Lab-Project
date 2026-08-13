import type {
  InferenceServiceError,
  PredictionResponse,
  RegionGeometryResponse,
  RiskMapResponse,
} from "../domain/inference";

export class InferenceApiError extends Error {
  constructor(readonly code: string, message: string, readonly requestId?: string) {
    super(message);
    this.name = "InferenceApiError";
  }
}

export class HttpInferenceService {
  constructor(private readonly baseUrl: string) {}

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
      },
    });
    if (!response.ok) {
      const raw = (await response.json().catch(() => ({}))) as Partial<InferenceServiceError>;
      throw new InferenceApiError(
        raw.code ?? "inference_request_failed",
        raw.message ?? `Inference request failed with status ${response.status}.`,
        raw.requestId,
      );
    }
    return response.json() as Promise<T>;
  }

  getGeometry(): Promise<RegionGeometryResponse> {
    return this.request("/regions/california/geometry");
  }

  getRiskMap(predictionDate: string): Promise<RiskMapResponse> {
    const timestamp = encodeURIComponent(`${predictionDate}T12:00:00Z`);
    return this.request(`/risk-map?region_id=california&timestamp=${timestamp}&mode=forecast_24h`);
  }

  getPrediction(cellId: string, predictionDate: string): Promise<PredictionResponse> {
    return this.request("/predictions", {
      method: "POST",
      body: JSON.stringify({
        regionId: cellId,
        timestamp: `${predictionDate}T12:00:00Z`,
        mode: "forecast_24h",
      }),
    });
  }
}
