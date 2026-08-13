# Wildfire IQ application implementation plan

Status: repository audit and Phase 1 UI foundation complete; real model/GCP integration remains pending source availability.

Last updated: 2026-08-11

## 1. Objective

Build the approved Wildfire IQ Prediction Studio as a desktop analyst workbench. The first milestone will use explicit development adapters, but its components will consume stable domain contracts so the original wildfire model, preprocessing pipeline, model artifacts, and GCP-hosted data can replace mocks without a frontend rewrite.

This plan follows the supplied `wildfire-prediction-ui-integration` skill. It does not assume that human-readable UI labels are model features, does not redesign the model, and does not invent production GCP resources.

## 2. Repository audit

### 2.1 Audit scope and evidence

The initial empty workspace audited before repository integration was:

```text
an empty local Git workspace
```

The complete worktree contains only `.git/`. Git reports:

- no commits on `master`;
- no tracked files;
- no untracked application files;
- no local branches with commits;
- no remotes;
- no stashes or alternate source trees visible in the repository;
- only Git's canonical empty-tree object (`4b825dc...`) in the object database.

Repository-wide filename and content searches therefore found no wildfire implementation to inspect or preserve.

### 2.2 Findings by required area

| Area | Finding | Consequence |
| --- | --- | --- |
| Frontend framework | None present | Use the supplied skill's fallback: React, TypeScript, and Vite. |
| Backend/API framework | None present | Reserve a Python/FastAPI backend boundary, but do not fabricate production inference in Phase 1. |
| ML training/inference | None present | No real inference entry point can yet be imported or validated. |
| Model features/order/types | None present | UI feature keys, ranges, units, transforms, and order must not be claimed as production schema. |
| Preprocessing | None present | Missing-value handling, normalization, encoding, scaling, and thresholds remain unknown. |
| Model artifacts | None present | Artifact format, versioning, loading, and cache behavior remain to be determined when artifacts are supplied. |
| Configuration | None present | Add frontend data-mode configuration in Phase 1; add only verified backend/GCP variables later. |
| GCP integration | None present | Define repository ports now; choose GCS, BigQuery, Earth Engine, or other adapters only from future evidence. |
| Deployment | None present | Defer production deployment decisions; keep the first frontend build portable. |
| Tests/linting | None present | Establish focused TypeScript tests and normal Vite checks for Phase 1. |
| Map/geospatial library | None present | Phase 1 uses MapLibre GL for the interactive basemap and a D3-projected county overlay when the local browser's GeoJSON worker did not complete. |
| UI reference | Supplied after the audit and preserved at `frontend/reference/prediction-studio-approved.png` | Implemented and visually verified at the source's 1672 × 941 viewport. |

### 2.3 What is intentionally not inferred

The following are unknown and must be recovered from the original model repository or artifact bundle before real integration:

- exact feature names and ordering;
- numeric/categorical types and allowed values;
- units, source datasets, valid ranges, and freshness constraints;
- normalization, imputation, encoding, aggregation, and temporal-window logic;
- geospatial grid definition and coordinate reference system;
- classification threshold and risk-class boundaries;
- calibration and confidence semantics;
- whether SHAP or another explanation path exists;
- artifact format and required runtime/library versions;
- GCP project, region, buckets, datasets, tables, object paths, and identities;
- historical-validation schema and ground-truth labeling rules.

Mock fixtures may demonstrate the UI contract, but they must be visibly development-only and must not be represented as model-derived facts.

## 3. Architecture decision

Because the repository establishes no stack, use the documented fallback with a small, replaceable boundary:

```text
React UI components
        |
Frontend feature controllers/hooks
        |
Typed service interfaces
        |
Adapter selected by VITE_DATA_MODE
   |                         |
Mock adapters            HTTP adapters
                             |
                        /api/v1 contracts
                             |
                      Backend services
                             |
       Existing ML pipeline + infrastructure repositories
                             |
                         Verified GCP sources
```

Rules for this boundary:

1. React components receive domain objects and callbacks; they do not import fixtures or call `fetch`.
2. Mock adapters and HTTP adapters implement the same service interfaces.
3. Adapter selection occurs once in the application composition root.
4. The browser never receives model artifacts or GCP credentials.
5. Feature metadata drives scenario controls; controls do not define production feature semantics.
6. Risk-class display metadata is centralized. Production thresholds come from backend model metadata.
7. Geometry and changing risk values remain separable so boundaries can be cached.
8. Missing explanations remain unavailable; they are never replaced with fabricated SHAP data.

## 4. Proposed project structure

No current repository convention exists, so the first implementation will use this structure without creating pass-through layers that have no responsibility:

```text
frontend/
  src/
    app/
      App.tsx
      composition.ts
      config.ts
    domain/
      prediction.ts
      regions.ts
      map.ts
      validation.ts
      model.ts
    services/
      contracts.ts
      http/
        httpClient.ts
        HttpPredictionService.ts
        HttpRegionService.ts
        HttpMapDataService.ts
        HttpValidationService.ts
        HttpModelMetadataService.ts
      mock/
        MockPredictionService.ts
        MockRegionService.ts
        MockMapDataService.ts
        MockValidationService.ts
        MockModelMetadataService.ts
    fixtures/
      california.ts
      predictions.ts
      validation.ts
      feature-metadata.ts
    features/
      prediction-studio/
      map/
      scenario/
      explainability/
      validation/
    components/
    styles/
    test/
  .env.example
  package.json
  vite.config.ts
backend/
  app/
    api/
    schemas/
    services/
    repositories/
    ml/
    infrastructure/gcp/
  tests/
docs/
  wildfire-app-implementation-plan.md
```

Only `frontend/` is implemented during Phase 1. `backend/` is shown as the target real-integration boundary and should be created in Phase 2 or when the original Python pipeline is supplied.

## 5. Frontend domain contracts

The exact property names may be adjusted before implementation if the original pipeline is supplied, but UI-facing contracts should retain these responsibilities.

### 5.1 Regions

```ts
type RegionAvailability = 'supported' | 'experimental' | 'validation_only' | 'unavailable';

interface RegionSummary {
  id: string;
  name: string;
  country?: string;
  regionType: 'state' | 'county' | 'grid' | 'custom';
  geometryId?: string;
  availability: RegionAvailability;
  availabilityReason?: string;
}
```

California is the default fixture region, not a condition embedded throughout map components.

### 5.2 Model and feature metadata

```ts
interface DataFreshness {
  status: 'fresh' | 'stale' | 'partial' | 'unavailable';
  observedAt?: string;
  message?: string;
}

interface ModelMetadata {
  modelVersion: string;
  threshold: number | null;
  updatedAt?: string;
  explanationCapability: 'available' | 'unavailable' | 'unknown';
  dataFreshness?: DataFreshness;
}

interface FeatureMetadata {
  key: string;
  displayName: string;
  type: 'number' | 'category';
  unit?: string;
  source?: string;
  min?: number;
  max?: number;
  step?: number;
  editableInScenario: boolean;
}
```

Development feature metadata will use clearly fictional fixture keys such as `mock.temperature_c`, not names that imply they are the trained model schema.

### 5.3 Predictions

```ts
type PredictionMode = 'live' | 'forecast_24h' | 'forecast_7d' | 'historical' | 'scenario';
type RiskClass = 'very_low' | 'low' | 'moderate' | 'high' | 'very_high';

interface PredictionRequest {
  regionId: string;
  timestamp: string;
  mode: PredictionMode;
  latitude?: number;
  longitude?: number;
  featureOverrides?: Record<string, number | string>;
}

interface FeatureValue {
  key: string;
  displayName: string;
  value: number | string | null;
  unit?: string;
  source?: string;
  observedAt?: string;
}

interface PredictionExplanation {
  confidence?: number;
  featureImportance?: Array<{
    feature: string;
    displayName: string;
    importance: number;
  }>;
  contributions?: Array<{
    feature: string;
    displayName: string;
    contribution: number;
  }>;
}

interface PredictionResponse {
  predictionId: string;
  regionId: string;
  timestamp: string;
  inferenceMode: PredictionMode;
  probability: number;
  riskClass: RiskClass;
  threshold: number | null;
  modelVersion: string;
  dataTimestamp?: string;
  featureSnapshot: { values: FeatureValue[] };
  explanation?: PredictionExplanation;
  provenance: 'development_fixture' | 'model';
}
```

`provenance` lets the product clearly label simulated fixture output during Phase 1.

### 5.4 Risk map and validation

```ts
interface RiskMapQuery {
  regionId: string;
  timestamp: string;
  mode: Exclude<PredictionMode, 'scenario'>;
}

interface RiskMapResponse {
  regionId: string;
  timestamp: string;
  geometryVersion: string;
  items: Array<{
    areaId: string;
    probability: number;
    riskClass: RiskClass;
  }>;
}

interface ValidationEvent {
  id: string;
  date: string;
  regionId: string;
  regionName: string;
  actualEvent: boolean;
  predictedProbability: number;
  predictedRiskClass: RiskClass;
  outcome: 'true_positive' | 'true_negative' | 'false_positive' | 'false_negative';
  modelVersion: string;
}

interface ValidationQuery {
  regionId?: string;
  startDate?: string;
  endDate?: string;
  modelVersion?: string;
  actualOutcome?: boolean;
  predictedClass?: RiskClass;
  cursor?: string;
  limit?: number;
}

interface ValidationPage {
  items: ValidationEvent[];
  nextCursor?: string;
  total?: number;
}
```

The frontend displays backend-supplied evaluation labels rather than recomputing them with an independent threshold.

## 6. Frontend service ports

```ts
interface RegionDetail extends RegionSummary {
  center: [longitude: number, latitude: number];
  bounds: [west: number, south: number, east: number, north: number];
}

interface RegionGeometry {
  regionId: string;
  geometryVersion: string;
  geojson: GeoJSON.FeatureCollection;
}

interface PredictionService {
  predict(request: PredictionRequest, signal?: AbortSignal): Promise<PredictionResponse>;
}

interface RegionService {
  listRegions(signal?: AbortSignal): Promise<RegionSummary[]>;
  getRegion(id: string, signal?: AbortSignal): Promise<RegionDetail>;
}

interface MapDataService {
  getGeometry(regionId: string, signal?: AbortSignal): Promise<RegionGeometry>;
  getRiskMap(query: RiskMapQuery, signal?: AbortSignal): Promise<RiskMapResponse>;
}

interface HistoricalValidationService {
  listEvents(query: ValidationQuery, signal?: AbortSignal): Promise<ValidationPage>;
}

interface ModelMetadataService {
  getMetadata(signal?: AbortSignal): Promise<ModelMetadata>;
  listFeatures(signal?: AbortSignal): Promise<FeatureMetadata[]>;
}
```

`VITE_DATA_MODE=mock|api` selects the adapter set. `VITE_API_BASE_URL` is read only by the HTTP composition layer. Components do not branch on data mode.

## 7. API contract to design toward

The HTTP adapters will target these versioned operations:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Liveness/readiness summary. |
| `GET` | `/api/v1/model/metadata` | Version, threshold, capability, and freshness metadata. |
| `GET` | `/api/v1/model/features` | Scenario-safe feature metadata derived from the real schema. |
| `GET` | `/api/v1/regions` | Region availability and coverage. |
| `GET` | `/api/v1/regions/{regionId}` | Region bounds and boundary metadata. |
| `GET` | `/api/v1/regions/{regionId}/geometry` | Versioned boundary geometry used by the current HTTP map adapter. |
| `GET` | `/api/v1/risk-map` | Batched area-level risk values for a region and timestamp. |
| `GET` | `/api/v1/features/snapshot` | Baseline feature snapshot resolved by the backend. |
| `POST` | `/api/v1/predictions` | Baseline or explicitly marked scenario inference. |
| `GET` | `/api/v1/validation/events` | Filtered, paginated historical validation events. |

Responses should use camelCase at the API boundary or a single documented case-conversion policy. Runtime response validation can be added once the project dependency set is established.

Errors should have a stable shape with `code`, `message`, optional `requestId`, and optional field-level details. UI states must distinguish unavailable, empty, stale, partial, and failed data.

## 8. Future backend integration seams

### 8.1 Domain services

- `PredictionService` orchestrates feature resolution, preprocessing, inference, thresholding, and response provenance.
- `FeatureService` returns an ordered, validated feature snapshot for a region/time.
- `RiskMapService` batches area-level inference or reads a precomputed risk surface.
- `ValidationService` reads authoritative observations and precomputed evaluation labels.
- `ModelMetadataService` exposes safe, traceable model information.

### 8.2 Repository ports

- `ModelArtifactRepository`
- `FeatureDataRepository`
- `BoundaryRepository`
- `HistoricalObservationRepository`
- `PredictionResultRepository` only if persistence is demonstrated to be necessary

GCS, BigQuery, Earth Engine, or another data product may implement a port only after its actual use is found in the original project. Bucket names, dataset IDs, and credential paths must remain configuration, never business logic.

### 8.3 ML boundary

When the original pipeline is supplied, identify one importable inference entry point and make it the only source of truth for:

- feature order and schema validation;
- missing-value behavior;
- encoding/scaling/transforms;
- artifact loading;
- probability extraction;
- threshold and risk-class mapping;
- explanation generation, if genuinely supported.

Artifacts should load once per backend process and be versioned/traceable. A golden test must compare the API path with the existing notebook/script output for an identical sample.

## 9. Phase 1: Prediction Studio UI foundation

### 9.1 Visual implementation result

The approved UI image was supplied after the initial audit. Phase 1 is implemented under `frontend/` and passed same-viewport visual QA. Evidence and comparison history are recorded in `frontend/design-qa.md` and `frontend/qa/`.

The implementation includes both mock and HTTP service bundles selected at the composition root. Mock predictions and illustrative explanation fixtures are explicitly labeled development data. The production HTTP adapter accepts absent explanations and the UI renders an unavailable state instead of substituting fixture values.

### 9.2 Scope once the reference is available

- desktop application shell and navigation;
- Prediction Studio mode tabs: Live, 24h Forecast, 7-day Forecast, Historical Validation;
- reusable California-default map with hover, selection, search, zoom, reset, legend, and time control;
- region selector with availability status;
- scenario panel generated from mock `FeatureMetadata` through the service interface;
- explicit Run prediction interaction (not a request per slider movement);
- baseline/scenario state and clear simulated-output labeling;
- probability, risk class, threshold, model version, and data freshness cards;
- explanation widgets that render only when explanation data is present;
- paginated/incremental validation table;
- loading, empty, stale, partial, and error states for major widgets;
- responsive desktop/laptop behavior and keyboard-visible focus states.

### 9.3 Mock-adapter rules

- Fixtures live only under `frontend/src/fixtures/`.
- Mock services are the only modules allowed to import fixtures.
- Fixture results use `provenance: 'development_fixture'`.
- Mock model/feature names are visibly labeled development metadata.
- Components never contain fixed prediction values.
- The API adapter never falls back silently to mock success after an error.
- Switching to `VITE_DATA_MODE=api` requires no component edits.

### 9.4 Phase 1 tests

- risk-class labels/colors are centralized and exhaustive;
- probability formatting handles `0`, `1`, and missing/invalid data;
- scenario edits generate only permitted `featureOverrides`;
- reset restores the baseline feature snapshot;
- service failures render a retryable error rather than mock numbers;
- unavailable explanation data renders an explicit unavailable state;
- region/map selection state stays synchronized;
- validation results preserve backend-supplied outcomes and pagination;
- mock and HTTP adapters satisfy the same TypeScript contracts.

## 10. Later phases

### Phase 2: Contract and backend shell

- add backend request/response schemas;
- implement health/readiness and development providers;
- add HTTP adapter contract tests;
- document environment configuration without secrets.

### Phase 3: Original model integration

- import/refactor the existing inference entry point without duplicating preprocessing;
- implement cached, versioned artifact loading;
- map model-safe results to API domain responses;
- add missing-feature and artifact-failure tests;
- add the golden inference test.

### Phase 4: Verified GCP data adapters

- implement only the GCP repositories found in the source project;
- use Application Default Credentials/service identity;
- validate source timestamps, missing coverage, and region availability;
- keep GCP SDK imports inside infrastructure adapters.

### Phase 5: Production risk map

- select stable/versioned boundary geometry;
- implement batched or precomputed regional risk retrieval;
- cache static geometry separately from time-varying values;
- validate hover/selection area IDs against backend records.

### Phase 6: Validation and explainability

- connect authoritative validation events and filters;
- expose explanations only if the original pipeline supports them;
- verify display names without leaking raw tensors.

### Phase 7: Deployment hardening

- container/runtime decisions based on actual model dependencies;
- readiness that fails when required model/data dependencies are unusable;
- structured prediction logs with request/prediction/model/region/latency fields;
- least-privilege service identity;
- CI for frontend, backend, contract, and golden inference tests.

## 11. Configuration plan

Phase 1 frontend example:

```text
VITE_DATA_MODE=mock
VITE_API_BASE_URL=/api/v1
```

Potential backend variables such as `GCP_PROJECT_ID`, `MODEL_BUCKET`, `MODEL_MANIFEST_PATH`, `DATA_BUCKET`, and `MODEL_VERSION` are not added until the supplied model/GCP sources prove they are needed. No service-account JSON or credentials belong in the repository or frontend.

## 12. Definition of done for milestone 1

Milestone 1 is complete when:

1. the rendered Prediction Studio has been compared with the approved source image at the same viewport and passes visual QA;
2. California fixture map interactions and region selection work;
3. scenario controls are metadata-driven and results are marked simulated;
4. all UI data comes through the five typed service interfaces;
5. mock/API mode changes at the composition root only;
6. loading, empty, stale, partial, error, and retry states are implemented;
7. frontend contract/state tests and production build pass;
8. no production model feature claims, fabricated explanations, GCP credentials, or direct artifact access exist in frontend code;
9. the original pipeline and GCP integration remain explicit unresolved inputs, not guessed implementations.

## 13. Inputs required for real integration

Before Phases 3-6, provide or mount the original model repository/artifact bundle and its environment/configuration documentation. At minimum it must make the inference entry point, preprocessing implementation, saved feature schema, model artifact(s), a known inference sample, and existing GCP references inspectable.

The approved Prediction Studio reference is now stored under `frontend/reference/`. No additional visual input is required for Phase 1.
