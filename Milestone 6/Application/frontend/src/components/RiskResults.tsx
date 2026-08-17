import { useEffect, useMemo, useState } from "react";
import { ArrowClockwise, CheckCircle, Fire, MapPin, Ranking, WarningCircle } from "@phosphor-icons/react";
import type { DailyValidationResponse, GridFeature, PredictionResponse, RegionGeometryResponse, RiskMapItem, RiskMapResponse, ValidationOutcome } from "../domain/inference";

interface RiskResultsProps {
  predictionDate: string;
  cutoffAt?: string;
  forecastMode?: string;
  geometry?: RegionGeometryResponse;
  riskMap?: RiskMapResponse;
  prediction?: PredictionResponse;
  validation?: DailyValidationResponse;
  selectedCellId?: string;
  error?: { message: string; code: string };
  isLoading: boolean;
  isLoadingDetail: boolean;
  isLoadingValidation: boolean;
  validationError?: string;
  onRetry: () => void;
  onSelectCell: (cellId: string) => void;
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function scoreColor(score?: number) {
  if (score === undefined) return "#e8eeec";
  if (score >= 0.8) return "#a84232";
  if (score >= 0.6) return "#d47a45";
  if (score >= 0.4) return "#e8b767";
  if (score >= 0.2) return "#a9c7a8";
  return "#dbe8df";
}

const OUTCOME_LABELS: Record<ValidationOutcome, string> = {
  true_positive: "Observed and captured",
  false_negative: "Observed but missed",
  false_positive: "Top-25 without observation",
  true_negative: "Neither observed nor alerted",
};

function outcomeFill(outcome?: ValidationOutcome) {
  if (outcome === "true_positive") return "url(#outcome-tp)";
  if (outcome === "false_negative") return "url(#outcome-fn)";
  if (outcome === "false_positive") return "url(#outcome-fp)";
  return "url(#outcome-tn)";
}

function mapGeometry(features: GridFeature[], width: number, height: number) {
  const points = features.flatMap((feature) => feature.geometry.coordinates[0]);
  const longitudes = points.map(([longitude]) => longitude);
  const latitudes = points.map(([, latitude]) => latitude);
  const minLon = Math.min(...longitudes);
  const maxLon = Math.max(...longitudes);
  const minLat = Math.min(...latitudes);
  const maxLat = Math.max(...latitudes);
  const padding = 24;
  const x = (longitude: number) => padding + ((longitude - minLon) / (maxLon - minLon)) * (width - padding * 2);
  const y = (latitude: number) => height - padding - ((latitude - minLat) / (maxLat - minLat)) * (height - padding * 2);
  return (feature: GridFeature) => `${feature.geometry.coordinates[0].map(([lon, lat], index) => `${index ? "L" : "M"}${x(lon).toFixed(2)},${y(lat).toFixed(2)}`).join(" ")} Z`;
}

function PriorityMap({ geometry, items, validation, mode, selectedCellId, onSelectCell }: {
  geometry: RegionGeometryResponse;
  items: RiskMapItem[];
  validation?: DailyValidationResponse;
  mode: "risk" | "validation";
  selectedCellId?: string;
  onSelectCell: (cellId: string) => void;
}) {
  const width = 720;
  const height = 520;
  const paths = useMemo(() => mapGeometry(geometry.geojson.features, width, height), [geometry]);
  const results = useMemo(() => new Map(items.map((item) => [item.areaId, item])), [items]);
  const outcomes = useMemo(() => new Map(validation?.items.map((item) => [item.areaId, item]) ?? []), [validation]);

  return (
    <div className="priority-map-wrap">
      <svg className="priority-map" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="California wildfire daily priority grid">
        <defs>
          <pattern id="outcome-tp" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#4b9a75" /><path d="M0 8L8 0" stroke="#d9f0e5" strokeWidth="2" /></pattern>
          <pattern id="outcome-fn" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#c95445" /><path d="M0 0L8 8M8 0L0 8" stroke="#fae4df" strokeWidth="1.5" /></pattern>
          <pattern id="outcome-fp" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#daa04f" /><circle cx="2" cy="2" r="1.5" fill="#fff1d2" /><circle cx="6" cy="6" r="1.5" fill="#fff1d2" /></pattern>
          <pattern id="outcome-tn" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#e3e8e8" /><path d="M0 4H8" stroke="#c8d0d0" strokeWidth="1" /></pattern>
        </defs>
        {geometry.geojson.features.map((feature) => {
          const item = results.get(feature.properties.id);
          const observed = outcomes.get(feature.properties.id);
          const selected = feature.properties.id === selectedCellId;
          const accessibleLabel = mode === "validation" && observed
            ? `${item?.areaName}, ${OUTCOME_LABELS[observed.outcome]}`
            : item ? `${item.areaName}, priority ${item.priorityRank}, ${percent(item.probability)} probability` : undefined;
          return (
            <path
              key={feature.properties.id}
              d={paths(feature)}
              fill={mode === "validation" ? outcomeFill(observed?.outcome) : scoreColor(item?.alertScore)}
              className={`${mode === "risk" && item?.alertTop25 ? "is-alert" : ""} ${selected ? "is-selected" : ""}`}
              role={item ? "button" : undefined}
              tabIndex={item ? 0 : undefined}
              aria-label={accessibleLabel}
              onClick={() => item && onSelectCell(item.areaId)}
              onKeyDown={(event) => {
                if (item && (event.key === "Enter" || event.key === " ")) {
                  event.preventDefault();
                  onSelectCell(item.areaId);
                }
              }}
            >
              {item ? <title>{accessibleLabel}</title> : null}
            </path>
          );
        })}
      </svg>
      {mode === "risk" ? <div className="map-legend" aria-label="Daily priority legend"><span>Lower priority</span><i className="legend-scale" /><span>Higher priority</span><b><i /> Top 25</b></div> : (
        <div className="map-legend validation-legend" aria-label="Actual versus Top-25 legend">
          <b className="legend-tp"><i /> Captured</b><b className="legend-fn"><i /> Missed</b><b className="legend-fp"><i /> False alert</b><b className="legend-tn"><i /> Neither</b>
        </div>
      )}
    </div>
  );
}

export function RiskResults(props: RiskResultsProps) {
  const [mapMode, setMapMode] = useState<"risk" | "validation">("risk");
  const ranked = props.riskMap?.items.slice().sort((a, b) => a.priorityRank - b.priorityRank) ?? [];
  const selected = ranked.find((item) => item.areaId === props.selectedCellId);
  const selectedValidation = props.validation?.items.find((item) => item.areaId === props.selectedCellId);
  const validationAvailable = props.validation?.status === "available";

  useEffect(() => { setMapMode("risk"); }, [props.predictionDate]);

  return (
    <section className="results-section" aria-live="polite">
      <div className="section-heading">
        <div><small>{props.forecastMode === "provisional_tomorrow" ? "Provisional tomorrow forecast" : "Step 03"}</small><h2>Prediction results</h2>{props.cutoffAt ? <p className="results-time-context">Data cutoff: {new Date(props.cutoffAt).toLocaleString("en-US", { timeZone: "America/Los_Angeles" })} California time · America/Los_Angeles</p> : <p className="results-time-context">California time · America/Los_Angeles</p>}</div>
        {props.riskMap ? <span className="status-pill status-pill--succeeded">Model scored</span> : null}
      </div>

      {props.isLoading ? (
        <div className="results-loading card"><span className="loading-ring" /><div><strong>Scoring the prepared grid</strong><p>Applying the calibrated classifier and daily ranker blend for {props.predictionDate}.</p></div></div>
      ) : null}

      {props.error ? (
        <div className="error-panel inference-error" role="alert">
          <WarningCircle weight="fill" />
          <div><strong>Prediction scoring needs attention</strong><p>{props.error.message}</p><small>{props.error.code}</small></div>
          <button type="button" onClick={props.onRetry}><ArrowClockwise /> Retry scoring</button>
        </div>
      ) : null}

      {props.geometry && props.riskMap ? (
        <div className="results-grid">
          <article className="card map-card">
            <div className="results-card-heading"><div><MapPin /><span><small>{mapMode === "risk" ? "Daily priority map" : "Post-event validation"}</small><strong>California model grid</strong></span></div><p>{props.riskMap.items.length} scored cells</p></div>
            <div className="map-mode-toggle" aria-label="Map display mode">
              <button type="button" className={mapMode === "risk" ? "is-selected" : ""} onClick={() => setMapMode("risk")}>Forecast risk</button>
              <button type="button" className={mapMode === "validation" ? "is-selected" : ""} disabled={!validationAvailable} onClick={() => setMapMode("validation")}>Actual vs Top-25</button>
            </div>
            {props.isLoadingValidation ? <p className="validation-availability">Checking completed FIRMS observations…</p> : null}
            {!props.isLoadingValidation && props.validation?.status !== "available" ? <p className="validation-availability"><WarningCircle /> {props.validation?.message ?? props.validationError ?? "Observed labels are temporarily unavailable."}</p> : null}
            <PriorityMap geometry={props.geometry} items={props.riskMap.items} validation={props.validation} mode={mapMode} selectedCellId={props.selectedCellId} onSelectCell={props.onSelectCell} />
          </article>

          <aside className="card ranking-card">
            <div className="results-card-heading"><div><Ranking /><span><small>Model priority</small><strong>Highest-ranked cells</strong></span></div></div>
            <div className="ranking-list">
              {ranked.map((item) => (
                <button key={item.areaId} type="button" className={item.areaId === props.selectedCellId ? "is-selected" : ""} onClick={() => props.onSelectCell(item.areaId)}>
                  <b>#{item.priorityRank}</b>
                  <span><strong>{item.areaId}</strong><small>{item.alertTop25 ? "Top-25 alert" : item.riskClass.replace("_", " ")}</small></span>
                  <em>{percent(item.probability)}</em>
                </button>
              ))}
            </div>
          </aside>
        </div>
      ) : null}

      {mapMode === "validation" && props.validation?.summary ? (
        <div className="validation-summary" aria-label="Selected-day validation summary">
          <div><small>FIRMS-observed cells</small><strong>{props.validation.summary.observedFireCells}</strong></div>
          <div><small>Captured in Top-25</small><strong>{props.validation.summary.capturedInTop25}</strong></div>
          <div><small>Recall@25</small><strong>{props.validation.summary.recallAt25 == null ? "—" : percent(props.validation.summary.recallAt25)}</strong></div>
          <div><small>Precision@25</small><strong>{props.validation.summary.precisionAt25 == null ? "—" : percent(props.validation.summary.precisionAt25)}</strong></div>
          <div><small>False alerts</small><strong>{props.validation.summary.falseAlerts}</strong></div>
        </div>
      ) : null}

      {selected ? (
        <article className="card cell-detail">
          <div className="cell-summary">
            <div className="cell-title"><Fire weight="fill" /><div><small>Selected grid cell</small><h3>{selected.areaId}</h3></div></div>
            <div><small>Daily priority</small><strong>#{selected.priorityRank}</strong></div>
            <div><small>Calibrated probability</small><strong>{percent(selected.probability)}</strong></div>
            <div><small>Priority score</small><strong>{percent(selected.alertScore)}</strong></div>
            <div><small>Alert state</small><strong className={selected.alertTop25 ? "alert-text" : ""}>{selected.alertTop25 ? "Top 25" : "Monitor"}</strong></div>
          </div>
          <div className="feature-detail">
            {props.isLoadingDetail ? <p>Loading model details…</p> : props.prediction?.regionId === selected.areaId ? (
              <>
                <div><small>Model version</small><code>{props.prediction.modelVersion}</code></div>
                <div className="drivers"><small>Strongest model drivers</small>{props.prediction.explanation?.contributions.slice(0, 5).map((entry) => <span key={entry.feature}><b>{entry.displayName}</b><em>{entry.contribution > 0 ? "+" : ""}{entry.contribution.toFixed(3)}</em></span>) ?? <p>Feature-level explanation is unavailable for this model artifact.</p>}</div>
              </>
            ) : <p>Model details are not available for the selected cell.</p>}
          </div>
          {selectedValidation && props.validation?.status === "available" ? (
            <div className="observation-detail">
              <small>FIRMS observation</small>
              <strong className={`outcome-text outcome-text--${selectedValidation.outcome}`}>{OUTCOME_LABELS[selectedValidation.outcome]}</strong>
              <span>{selectedValidation.actualEvent ? <CheckCircle weight="fill" /> : <WarningCircle />} {selectedValidation.actualEvent ? "Fire detected" : "No qualifying detection"}</span>
              <span>Pixels: {selectedValidation.firmsPixelCount ?? "—"} · Max confidence: {selectedValidation.firmsMaxConfidence?.toFixed(0) ?? "—"}</span>
              <code>{props.validation.labelSource?.replaceAll("_", " ")}</code>
            </div>
          ) : props.validation || props.isLoadingValidation || props.validationError ? (
            <div className="observation-detail observation-detail--unavailable">
              <small>Observed-label status</small>
              <strong>{props.isLoadingValidation ? "Checking FIRMS observations" : "Observed labels not available yet"}</strong>
              <span><WarningCircle /> {props.validation?.message ?? props.validationError ?? "Checking whether completed FIRMS labels are available."}</span>
            </div>
          ) : null}
        </article>
      ) : null}
    </section>
  );
}
