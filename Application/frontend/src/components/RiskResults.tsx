import { useMemo } from "react";
import { ArrowClockwise, Fire, MapPin, Ranking, WarningCircle } from "@phosphor-icons/react";
import type { GridFeature, PredictionResponse, RegionGeometryResponse, RiskMapItem, RiskMapResponse } from "../domain/inference";

interface RiskResultsProps {
  predictionDate: string;
  geometry?: RegionGeometryResponse;
  riskMap?: RiskMapResponse;
  prediction?: PredictionResponse;
  selectedCellId?: string;
  error?: { message: string; code: string };
  isLoading: boolean;
  isLoadingDetail: boolean;
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

function PriorityMap({ geometry, items, selectedCellId, onSelectCell }: {
  geometry: RegionGeometryResponse;
  items: RiskMapItem[];
  selectedCellId?: string;
  onSelectCell: (cellId: string) => void;
}) {
  const width = 720;
  const height = 520;
  const paths = useMemo(() => mapGeometry(geometry.geojson.features, width, height), [geometry]);
  const results = useMemo(() => new Map(items.map((item) => [item.areaId, item])), [items]);

  return (
    <div className="priority-map-wrap">
      <svg className="priority-map" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="California wildfire daily priority grid">
        {geometry.geojson.features.map((feature) => {
          const item = results.get(feature.properties.id);
          const selected = feature.properties.id === selectedCellId;
          return (
            <path
              key={feature.properties.id}
              d={paths(feature)}
              fill={scoreColor(item?.alertScore)}
              className={`${item?.alertTop25 ? "is-alert" : ""} ${selected ? "is-selected" : ""}`}
              role={item ? "button" : undefined}
              tabIndex={item ? 0 : undefined}
              aria-label={item ? `${item.areaName}, priority ${item.priorityRank}, ${percent(item.probability)} probability` : undefined}
              onClick={() => item && onSelectCell(item.areaId)}
              onKeyDown={(event) => {
                if (item && (event.key === "Enter" || event.key === " ")) {
                  event.preventDefault();
                  onSelectCell(item.areaId);
                }
              }}
            >
              {item ? <title>{`${item.areaName} · #${item.priorityRank} · ${percent(item.probability)}`}</title> : null}
            </path>
          );
        })}
      </svg>
      <div className="map-legend" aria-label="Daily priority legend">
        <span>Lower priority</span>
        <i className="legend-scale" />
        <span>Higher priority</span>
        <b><i /> Top 25</b>
      </div>
    </div>
  );
}

export function RiskResults(props: RiskResultsProps) {
  const ranked = props.riskMap?.items.slice().sort((a, b) => a.priorityRank - b.priorityRank) ?? [];
  const selected = ranked.find((item) => item.areaId === props.selectedCellId);

  return (
    <section className="results-section" aria-live="polite">
      <div className="section-heading">
        <div><small>Step 03</small><h2>Prediction results</h2></div>
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
            <div className="results-card-heading"><div><MapPin /><span><small>Daily priority map</small><strong>California model grid</strong></span></div><p>{props.riskMap.items.length} scored cells</p></div>
            <PriorityMap geometry={props.geometry} items={props.riskMap.items} selectedCellId={props.selectedCellId} onSelectCell={props.onSelectCell} />
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
        </article>
      ) : null}
    </section>
  );
}
