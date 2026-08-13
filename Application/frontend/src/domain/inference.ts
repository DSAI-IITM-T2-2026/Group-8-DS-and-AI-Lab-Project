export type RiskClass = "very_low" | "low" | "moderate" | "high" | "very_high";

export interface RiskMapItem {
  areaId: string;
  areaName: string;
  probability: number;
  rawProbability: number;
  alertScore: number;
  priorityRank: number;
  alertTop25: boolean;
  riskClass: RiskClass;
  updatedAt: string;
}

export interface RiskMapResponse {
  regionId: string;
  timestamp: string;
  geometryVersion: string;
  items: RiskMapItem[];
  provenance: "model";
}

export interface GridFeature {
  type: "Feature";
  geometry: { type: "Polygon"; coordinates: number[][][] };
  properties: {
    id: string;
    name: string;
    stateFips: string;
    fireRegionCategory: string;
    elevation: number;
  };
}

export interface RegionGeometryResponse {
  regionId: string;
  geometryVersion: string;
  geojson: { type: "FeatureCollection"; features: GridFeature[] };
}

export interface FeatureSnapshotValue {
  key: string;
  displayName: string;
  value: number;
  unit?: string;
  source: string;
  observedAt?: string;
}

export interface PredictionResponse {
  predictionId: string;
  regionId: string;
  timestamp: string;
  inferenceMode: string;
  probability: number;
  rawProbability: number;
  alertScore: number;
  priorityRank: number;
  alertTop25: boolean;
  riskClass: RiskClass;
  modelVersion: string;
  dataTimestamp: string;
  featureSnapshot: { values: FeatureSnapshotValue[] };
  explanation?: {
    confidence: number;
    featureImportance: Array<{ feature: string; displayName: string; importance: number }>;
    contributions: Array<{ feature: string; displayName: string; contribution: number }>;
    provenance: "model";
  };
  provenance: "model";
}

export interface InferenceServiceError {
  code: string;
  message: string;
  requestId?: string;
  fieldErrors?: Record<string, string>;
}
