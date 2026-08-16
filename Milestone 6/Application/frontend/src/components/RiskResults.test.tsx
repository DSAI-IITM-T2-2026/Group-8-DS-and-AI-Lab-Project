import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RiskResults } from "./RiskResults";

const geometry = {
  regionId: "california",
  geometryVersion: "v1",
  geojson: {
    type: "FeatureCollection",
    features: [{
      type: "Feature",
      geometry: { type: "Polygon", coordinates: [[[-121, 39], [-120.75, 39], [-120.75, 39.25], [-121, 39.25], [-121, 39]]] },
      properties: { id: "cell-a", name: "Grid cell cell-a", stateFips: "06", fireRegionCategory: "High", elevation: 100 },
    }],
  },
} as const;

const riskMap = {
  regionId: "california", timestamp: "2025-08-01T12:00:00Z", geometryVersion: "v1", provenance: "model",
  items: [{ areaId: "cell-a", areaName: "Grid cell cell-a", probability: 0.25, rawProbability: 0.3, alertScore: 0.95, priorityRank: 1, alertTop25: true, riskClass: "very_high", updatedAt: "2025-08-01T12:00:00Z" }],
} as const;

describe("RiskResults", () => {
  it("renders daily priority, calibrated probability, and top-25 state", () => {
    const select = vi.fn();
    render(<RiskResults predictionDate="2025-08-01" geometry={geometry as any} riskMap={riskMap as any} selectedCellId="cell-a" isLoading={false} isLoadingDetail={false} isLoadingValidation={false} onRetry={vi.fn()} onSelectCell={select} />);
    expect(screen.getByText("Prediction results")).toBeInTheDocument();
    expect(screen.getAllByText("25.0%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Top 25").length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByRole("button", { name: /cell-a/i })[0]);
    expect(select).toHaveBeenCalledWith("cell-a");
  });

  it("renders a recoverable model error without hiding preparation", () => {
    render(<RiskResults predictionDate="2025-08-01" error={{ code: "model_unavailable", message: "Mount the model artifact." }} isLoading={false} isLoadingDetail={false} isLoadingValidation={false} onRetry={vi.fn()} onSelectCell={vi.fn()} />);
    expect(screen.getByText("Prediction scoring needs attention")).toBeInTheDocument();
    expect(screen.getByText("model_unavailable")).toBeInTheDocument();
  });

  it("toggles to actual-vs-top-25 outcomes and renders operational metrics", () => {
    const ids = ["cell-a", "cell-b", "cell-c", "cell-d"];
    const expandedGeometry = {
      ...geometry,
      geojson: {
        ...geometry.geojson,
        features: ids.map((id, index) => ({
          ...geometry.geojson.features[0],
          geometry: { type: "Polygon", coordinates: [[[-121 + index * .25, 39], [-120.75 + index * .25, 39], [-120.75 + index * .25, 39.25], [-121 + index * .25, 39.25], [-121 + index * .25, 39]]] },
          properties: { ...geometry.geojson.features[0].properties, id, name: `Grid cell ${id}` },
        })),
      },
    };
    const expandedRisk = {
      ...riskMap,
      items: ids.map((id, index) => ({ ...riskMap.items[0], areaId: id, areaName: `Grid cell ${id}`, priorityRank: index + 1, alertTop25: index < 2 })),
    };
    const outcomes = ["true_positive", "false_positive", "false_negative", "true_negative"] as const;
    const validation = {
      status: "available", date: "2025-08-01", modelVersion: "model-v1",
      labelSource: "historical_archive", message: "Ready", provenance: "firms_observation",
      items: ids.map((areaId, index) => ({ areaId, actualEvent: index === 0 || index === 2, firmsPixelCount: index === 0 ? 3 : null, firmsMaxConfidence: index === 0 ? 88 : null, alertTop25: index < 2, outcome: outcomes[index] })),
      summary: { observedFireCells: 2, capturedInTop25: 1, recallAt25: .5, precisionAt25: .5, falseAlerts: 1, top25Count: 2 },
    };

    render(<RiskResults predictionDate="2025-08-01" geometry={expandedGeometry as any} riskMap={expandedRisk as any} validation={validation as any} selectedCellId="cell-a" isLoading={false} isLoadingDetail={false} isLoadingValidation={false} onRetry={vi.fn()} onSelectCell={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Actual vs Top-25" }));

    expect(screen.getByRole("button", { name: /cell-a, observed and captured/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cell-b, top-25 without observation/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cell-c, observed but missed/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cell-d, neither observed nor alerted/i })).toBeInTheDocument();
    expect(screen.getByText("FIRMS-observed cells")).toBeInTheDocument();
    expect(screen.getAllByText("50.0%")).toHaveLength(2);
    expect(screen.getByText("Observed and captured")).toBeInTheDocument();
    expect(screen.getByText(/Pixels: 3/)).toBeInTheDocument();
  });

  it("keeps the risk map usable when observed labels are not mature", () => {
    const validation = {
      status: "not_mature", date: "2026-08-17", modelVersion: "model-v1",
      labelSource: null, message: "Observed labels are available only after the California day has ended.",
      items: [], summary: null, provenance: "firms_observation",
    };
    render(<RiskResults predictionDate="2026-08-17" geometry={geometry as any} riskMap={riskMap as any} validation={validation as any} selectedCellId="cell-a" isLoading={false} isLoadingDetail={false} isLoadingValidation={false} onRetry={vi.fn()} onSelectCell={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Actual vs Top-25" })).toBeDisabled();
    expect(screen.getAllByText(/available only after the California day has ended/i)).toHaveLength(2);
    expect(screen.getByText("Observed-label status")).toBeInTheDocument();
    expect(screen.getByText("Observed labels not available yet")).toBeInTheDocument();
    expect(screen.getByLabelText("California wildfire daily priority grid")).toBeInTheDocument();
  });
});
