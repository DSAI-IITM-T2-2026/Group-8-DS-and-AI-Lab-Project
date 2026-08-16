import { ChartLineUp, Info } from "@phosphor-icons/react";
import type { ModelEvaluationResponse } from "../domain/inference";

interface ModelEvaluationScorecardProps {
  evaluation?: ModelEvaluationResponse;
  isLoading: boolean;
  isUnavailable: boolean;
}

export function ModelEvaluationScorecard({ evaluation, isLoading, isUnavailable }: ModelEvaluationScorecardProps) {
  return (
    <section className="evaluation-section" aria-labelledby="evaluation-title">
      <div className="section-heading evaluation-heading">
        <div><small>Model evidence</small><h2 id="evaluation-title">Champion model evaluation</h2></div>
        <span className="evaluation-split">{evaluation?.split ?? "Held-out 2025 test set"}</span>
      </div>

      <article className="card evaluation-card">
        {isLoading ? <div className="evaluation-state"><span className="loading-ring" /> Loading held-out evaluation…</div> : null}
        {isUnavailable ? <div className="evaluation-state"><Info /> Model evaluation metrics are temporarily unavailable.</div> : null}
        {evaluation ? (
          <>
            <div className="evaluation-intro">
              <ChartLineUp />
              <div><strong>Performance on unseen 2025 labels</strong><p>{evaluation.rows.toLocaleString()} cell-days · {evaluation.positives.toLocaleString()} positive cell-days</p></div>
            </div>
            <div className="metric-grid">
              {evaluation.metrics.map((metric) => (
                <div className="metric-tile" key={metric.key} title={metric.description}>
                  <small>{metric.label}</small>
                  <strong>{metric.displayValue}</strong>
                  <p>{metric.description}</p>
                </div>
              ))}
            </div>
            <p className="evaluation-note"><Info weight="fill" /> These are historical model-level evaluation results, not measured performance for the selected future prediction.</p>
          </>
        ) : null}
      </article>
    </section>
  );
}
