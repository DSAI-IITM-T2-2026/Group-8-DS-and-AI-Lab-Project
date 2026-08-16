import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ModelEvaluationScorecard } from "./ModelEvaluationScorecard";

const evaluation = {
  evaluationVersion: "milestone-5-champion-2025-v1",
  modelVersion: "champion-v1",
  split: "Held-out 2025 test set",
  labelYear: 2025,
  rows: 93518,
  positives: 1325,
  baseline: { label: "Naive constant-rate baseline", prAuc: 0.0093 },
  provenance: "held_out_evaluation",
  metrics: [
    { key: "pr_auc", label: "PR-AUC", value: 0.1451, displayValue: "0.1451", description: "Precision-recall performance." },
    { key: "recall_at_25", label: "Recall@25", value: 0.3638, displayValue: "36.38%", description: "Top-25 capture." },
    { key: "pr_auc_lift", label: "PR-AUC lift", value: 15.6, displayValue: "15.6×", description: "Lift over baseline." },
  ],
} as const;

describe("ModelEvaluationScorecard", () => {
  it("labels held-out metrics separately from a future prediction", () => {
    render(<ModelEvaluationScorecard evaluation={evaluation as any} isLoading={false} isUnavailable={false} />);
    expect(screen.getByText("Held-out 2025 test set")).toBeInTheDocument();
    expect(screen.getByText("93,518 cell-days · 1,325 positive cell-days")).toBeInTheDocument();
    expect(screen.getByText("0.1451")).toBeInTheDocument();
    expect(screen.getByText("36.38%")).toBeInTheDocument();
    expect(screen.getByText("15.6×")).toBeInTheDocument();
    expect(screen.getByText(/not measured performance for the selected future prediction/i)).toBeInTheDocument();
  });
});
