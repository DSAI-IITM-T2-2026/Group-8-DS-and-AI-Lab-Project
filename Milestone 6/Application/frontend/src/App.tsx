import { useMemo } from "react";
import {
  ArrowClockwise,
  CalendarBlank,
  CheckCircle,
  CloudArrowDown,
  ClockCountdown,
  Database,
  FileArrowDown,
  Fire,
  HourglassMedium,
  Play,
  StopCircle,
  WarningCircle,
} from "@phosphor-icons/react";
import { appConfig } from "./app/config";
import { useInference } from "./app/useInference";
import { useModelEvaluation } from "./app/useModelEvaluation";
import { usePipelineRun } from "./app/usePipelineRun";
import { ModelEvaluationScorecard } from "./components/ModelEvaluationScorecard";
import { RiskResults } from "./components/RiskResults";
import { HttpInferenceService } from "./services/inference";
import { HttpPipelineService } from "./services/pipeline";
import type { PipelineStage, SourceInventoryItem } from "./domain/pipeline";

const STAGES: Array<{ id: PipelineStage; label: string }> = [
  { id: "validating", label: "Validate date" },
  { id: "inventory", label: "Check storage" },
  { id: "era5", label: "ERA5 weather" },
  { id: "firms", label: "FIRMS history" },
  { id: "sentinel2", label: "Sentinel-2" },
  { id: "sentinel5p", label: "Sentinel-5P" },
  { id: "preprocessing", label: "Build features" },
  { id: "exporting", label: "Export parquet" },
  { id: "completed", label: "Ready" },
];

const SOURCES = [
  ["era5", "ERA5", "Weather history through D−6"],
  ["firms", "FIRMS", "Prior fire context through D−1"],
  ["sentinel2", "Sentinel-2", "Optical snapshot through D−1"],
  ["sentinel5p", "Sentinel-5P", "Atmospheric snapshot through D−1"],
] as const;

function dateShift(value: string, amount: number) {
  const date = new Date(`${value}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + amount);
  return date.toISOString().slice(0, 10);
}

function sourceSummary(item?: SourceInventoryItem) {
  if (!item) return "Checked when the run starts";
  if (item.missing === 0) return `${item.available} of ${item.required} already available`;
  return `${item.available} available · ${item.missing} to prepare`;
}

export function App() {
  const service = useMemo(() => new HttpPipelineService(appConfig.apiBaseUrl), []);
  const inferenceService = useMemo(() => new HttpInferenceService(appConfig.apiBaseUrl), []);
  const pipeline = usePipelineRun(service);
  const modelEvaluation = useModelEvaluation(inferenceService);
  const inferenceDate = pipeline.run?.status === "succeeded" ? pipeline.run.artifact?.labelDate : undefined;
  const inference = useInference(inferenceService, inferenceDate);
  const activeIndex = STAGES.findIndex((stage) => stage.id === pipeline.run?.stage);
  const isActive = pipeline.run && ["queued", "running", "waiting_external"].includes(pipeline.run.status);
  const wasCancelled = pipeline.run?.status === "interrupted" && pipeline.run.errorCode === "cancelled_by_user";
  const hasError = pipeline.error || pipeline.run?.status === "failed" || (pipeline.run?.status === "interrupted" && !wasCancelled);

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Wildfire IQ home">
          <img src="/assets/wildfire-iq-shield.png?v=2" alt="" />
          <span><strong>WILDFIRE <em>IQ</em></strong><small>California Forecasting Studio</small></span>
        </a>
        <div className="system-state"><span /> Pipeline workspace <b>Local / VM</b></div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="eyebrow"><Fire weight="fill" /> California wildfire forecasting</div>
          <h1>Generate a wildfire forecast<br /><span>for a prediction day.</span></h1>
          <p>Choose the day you want to predict. Wildfire IQ reuses a valid model-ready parquet when available, prepares only missing causal inputs when needed, then scores California with the champion model to produce a risk map and ranked cell predictions.</p>
        </section>

        <section className="workspace-grid">
          <article className="card request-card">
            <div className="card-heading"><span className="icon-box"><CalendarBlank /></span><div><small>Step 01</small><h2>Select prediction date</h2></div></div>
            <div className="date-label-row">
              <label htmlFor="prediction-date">Prediction day</label>
              <button
                type="button"
                className={pipeline.selectedDate === pipeline.config?.maxPredictionDate ? "tomorrow-shortcut is-selected" : "tomorrow-shortcut"}
                onClick={() => pipeline.config && pipeline.selectDate(pipeline.config.maxPredictionDate)}
                disabled={!pipeline.config || Boolean(isActive)}
              >
                <ClockCountdown /> Tomorrow
              </button>
            </div>
            <div className="date-row">
              <input
                id="prediction-date"
                type="date"
                value={pipeline.selectedDate}
                min={pipeline.config?.minPredictionDate}
                max={pipeline.config?.maxPredictionDate}
                onChange={(event) => pipeline.selectDate(event.target.value)}
                disabled={Boolean(isActive)}
              />
              <button type="button" className="primary-button" onClick={() => void pipeline.start()} disabled={!pipeline.config || pipeline.isSubmitting || Boolean(isActive)}>
                <Play weight="fill" /> {pipeline.isSubmitting ? "Queuing…" : isActive ? "Forecast in progress" : "Generate wildfire forecast"}
              </button>
            </div>
            {pipeline.config ? (
              <div className="window-card">
                <div><small>30-day history starts</small><strong>{dateShift(pipeline.selectedDate, -pipeline.config.lookbackDays)}</strong></div>
                <i />
                <div><small>ERA5 feature end</small><strong>{dateShift(pipeline.selectedDate, -6)}</strong></div>
                <i />
                <div><small>Satellite / fire as-of</small><strong>{dateShift(pipeline.selectedDate, -1)}</strong></div>
                <i />
                <div className="target-date"><small>Prediction day</small><strong>{pipeline.selectedDate}</strong></div>
              </div>
            ) : <div className="window-card window-card--loading">Connecting to the forecasting service…</div>}
            <p className="causal-note"><CheckCircle weight="fill" /> Only information available before the prediction day is used. No future fire labels enter the model features.</p>
          </article>

          <aside className="card readiness-card">
            <div className="card-heading"><span className="icon-box"><Database /></span><div><small>Prediction contract</small><h2>What powers the forecast</h2></div></div>
            <div className="contract-number"><strong>{pipeline.config?.expectedFeatureCount ?? 86}</strong><span>locked model features<br />across high & medium-risk cells</span></div>
            <dl>
              <div><dt>Supported dates</dt><dd>{pipeline.config ? `${pipeline.config.minPredictionDate} — ${pipeline.config.maxPredictionDate}` : "Loading…"}</dd></div>
              <div><dt>Time reference</dt><dd>California · {pipeline.config?.timezone ?? "America/Los_Angeles"}</dd></div>
              <div><dt>Final object</dt><dd><code>final_processed/D_test.parquet</code></dd></div>
            </dl>
          </aside>
        </section>

        <ModelEvaluationScorecard
          evaluation={modelEvaluation.evaluation}
          isLoading={modelEvaluation.isLoading}
          isUnavailable={modelEvaluation.isUnavailable}
        />

        {(pipeline.run || hasError) ? (
          <section className="run-section" aria-live="polite">
            <div className="section-heading">
              <div><small>Step 02</small><h2>Forecast run</h2></div>
              <div className="run-actions">
                {pipeline.run ? <span className={`status-pill status-pill--${pipeline.run.status} ${wasCancelled ? "status-pill--cancelled" : ""}`}>{wasCancelled ? "stopped" : pipeline.run.status.replace("_", " ")}</span> : null}
                {isActive ? <button type="button" className="stop-run-button" disabled={pipeline.isStopping} onClick={() => void pipeline.stop()}><StopCircle weight="fill" /> {pipeline.isStopping ? "Stopping…" : "Stop forecast"}</button> : null}
              </div>
            </div>

            {hasError ? (
              <div className="error-panel" role="alert"><WarningCircle weight="fill" /><div><strong>Preparation needs attention</strong><p>{pipeline.error || pipeline.run?.message}</p><small>{pipeline.run?.errorCode}</small></div><button type="button" onClick={() => void pipeline.retry()}><ArrowClockwise /> Retry</button></div>
            ) : null}

            {pipeline.run?.status === "waiting_external" ? (
              <div className="waiting-panel"><HourglassMedium /><div><strong>Waiting for Google Earth Engine</strong><p>Satellite exports may take 30 minutes or longer. This page can be closed—the worker will continue and the run will be restored when you return.</p></div></div>
            ) : null}

            {pipeline.run?.status === "unavailable" ? (
              <div className="data-unavailable-panel"><ClockCountdown /><div><strong>Tomorrow’s data is not available yet</strong><p>The required causal source files are not complete. No cloud preparation was started; check again later.</p></div></div>
            ) : null}

            {wasCancelled ? (
              <div className="stopped-panel"><StopCircle weight="fill" /><div><strong>Forecast stopped</strong><p>The local preparation process was stopped. Cloud exports submitted before cancellation may finish and will be reused by a later run.</p></div></div>
            ) : null}

            {pipeline.run && pipeline.run.status !== "unavailable" ? (
              <div className="progress-card card">
                <div className="progress-header"><div><small>Current activity</small><strong>{pipeline.run.message}</strong></div>{pipeline.run.progressTotal > 0 ? <b>{pipeline.run.progressCompleted}/{pipeline.run.progressTotal}</b> : null}</div>
                <ol className="stage-track">
                  {STAGES.map((stage, index) => {
                    const complete = pipeline.run?.status === "succeeded" || index < activeIndex;
                    const current = index === activeIndex && pipeline.run?.status !== "succeeded";
                    return <li key={stage.id} className={complete ? "is-complete" : current ? "is-current" : ""}><span>{complete ? <CheckCircle weight="fill" /> : index + 1}</span><small>{stage.label}</small></li>;
                  })}
                </ol>
                <div className="source-grid">
                  {SOURCES.map(([key, name, detail]) => {
                    const item = pipeline.run?.sourceInventory[key];
                    return <div className="source-card" key={key}><CloudArrowDown /><div><strong>{name}</strong><small>{detail}</small><p>{sourceSummary(item)}</p></div></div>;
                  })}
                </div>
              </div>
            ) : null}

            {pipeline.run?.status === "succeeded" && pipeline.run.artifact ? (
              <div className="success-panel"><div className="success-icon"><FileArrowDown weight="fill" /></div><div className="success-copy"><small>Model input ready</small><h2>{pipeline.run.artifact.labelDate}_test.parquet</h2><p>The causal feature contract passed. Wildfire IQ is scoring the complete daily grid from this verified artifact.</p><code>{pipeline.run.artifact.objectUri}</code></div><div className="success-metrics"><div><strong>{pipeline.run.artifact.featureCount}</strong><small>features</small></div><div><strong>{pipeline.run.artifact.cellCount}</strong><small>cells</small></div><div><strong>{pipeline.run.artifact.rowCount}</strong><small>rows</small></div></div></div>
            ) : null}
          </section>
        ) : null}

        {inferenceDate ? (
          <RiskResults
            predictionDate={inferenceDate}
            geometry={inference.geometry}
            riskMap={inference.riskMap}
            prediction={inference.prediction}
            validation={inference.validation}
            selectedCellId={inference.selectedCellId}
            error={inference.error}
            isLoading={inference.isLoading}
            isLoadingDetail={inference.isLoadingDetail}
            isLoadingValidation={inference.isLoadingValidation}
            validationError={inference.validationError}
            onRetry={() => void inference.retry()}
            onSelectCell={(cellId) => void inference.selectCell(cellId)}
          />
        ) : null}
      </main>
      <footer>Wildfire IQ · Group 8 DSAI Lab · Causal data preparation and model inference</footer>
    </div>
  );
}
