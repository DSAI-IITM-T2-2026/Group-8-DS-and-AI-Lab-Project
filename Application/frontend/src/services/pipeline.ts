import type { PipelineConfig, PipelineRun, ServiceError } from "../domain/pipeline";

export class PipelineApiError extends Error {
  constructor(readonly code: string, message: string, readonly fieldErrors?: Record<string, string>) {
    super(message);
    this.name = "PipelineApiError";
  }
}

export class HttpPipelineService {
  constructor(private readonly baseUrl: string) {}

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}) },
    });
    if (!response.ok) {
      const raw = await response.json().catch(() => ({}));
      const detail = raw.detail ?? raw;
      const error = detail as Partial<ServiceError>;
      throw new PipelineApiError(error.code ?? "request_failed", error.message ?? `Request failed with status ${response.status}.`, error.fieldErrors);
    }
    return response.json() as Promise<T>;
  }

  getConfig(): Promise<PipelineConfig> { return this.request("/pipeline/config"); }
  createRun(predictionDate: string): Promise<PipelineRun> {
    return this.request("/pipeline-runs", { method: "POST", body: JSON.stringify({ predictionDate }) });
  }
  getRun(runId: string): Promise<PipelineRun> { return this.request(`/pipeline-runs/${encodeURIComponent(runId)}`); }
  listRuns(predictionDate?: string): Promise<PipelineRun[]> {
    const query = predictionDate ? `?predictionDate=${encodeURIComponent(predictionDate)}&limit=10` : "?limit=10";
    return this.request(`/pipeline-runs${query}`);
  }
}
