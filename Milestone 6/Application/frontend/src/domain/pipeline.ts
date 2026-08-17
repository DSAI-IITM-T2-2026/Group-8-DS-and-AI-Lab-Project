export type RunStatus = "queued" | "running" | "waiting_external" | "succeeded" | "unavailable" | "failed" | "interrupted";
export type PipelineStage = "validating" | "inventory" | "era5" | "firms" | "sentinel2" | "sentinel5p" | "preprocessing" | "exporting" | "completed";

export interface PipelineConfig {
  minPredictionDate: string;
  maxPredictionDate: string;
  timezone: string;
  cutoffLocalTime: string;
  lookbackDays: number;
  expectedFeatureCount: number;
}

export interface SourceInventoryItem {
  required: number;
  available: number;
  missing: number;
  scheduled: number;
  pending: number;
  requiredThroughDate?: string;
  selectedThroughDate?: string;
  selectedWindowStartDate?: string;
  ageDays?: number;
  mode?: "exact" | "latest_causal" | "imputed" | "static";
  ready?: boolean;
  exactAvailable?: boolean;
  exactArrivedAfterCutoff?: boolean;
  message?: string;
}

export interface ArtifactSummary {
  objectUri: string;
  rowCount: number;
  featureCount: number;
  cellCount: number;
  labelDate: string;
  eoAsOfDate: string;
  featureEndDate: string;
  createdAt: string;
  cutoffAt?: string;
  timezone?: string;
  forecastMode?: string;
  sourceSnapshots?: Record<string, SourceInventoryItem>;
  provenanceUri?: string;
  immutableProvenanceUri?: string;
  artifactQuality?: "exact" | "era5_fallback";
  needsRefresh?: boolean;
  requiredFeatureEndDate?: string;
  availabilityPolicy?: "cutoff_snapshot" | "late_exact_refresh";
  refreshedAt?: string;
  supersedesProvenanceUri?: string;
  generation?: string;
}

export interface PipelineRun {
  runId: string;
  predictionDate: string;
  status: RunStatus;
  stage: PipelineStage;
  message: string;
  progressCompleted: number;
  progressTotal: number;
  sourceInventory: Record<string, SourceInventoryItem>;
  artifact?: ArtifactSummary;
  errorCode?: string;
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
}

export interface ServiceError { code: string; message: string; fieldErrors?: Record<string, string> }
