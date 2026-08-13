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
    render(<RiskResults predictionDate="2025-08-01" geometry={geometry as any} riskMap={riskMap as any} selectedCellId="cell-a" isLoading={false} isLoadingDetail={false} onRetry={vi.fn()} onSelectCell={select} />);
    expect(screen.getByText("Prediction results")).toBeInTheDocument();
    expect(screen.getAllByText("25.0%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Top 25").length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByRole("button", { name: /cell-a/i })[0]);
    expect(select).toHaveBeenCalledWith("cell-a");
  });

  it("renders a recoverable model error without hiding preparation", () => {
    render(<RiskResults predictionDate="2025-08-01" error={{ code: "model_unavailable", message: "Mount the model artifact." }} isLoading={false} isLoadingDetail={false} onRetry={vi.fn()} onSelectCell={vi.fn()} />);
    expect(screen.getByText("Prediction scoring needs attention")).toBeInTheDocument();
    expect(screen.getByText("model_unavailable")).toBeInTheDocument();
  });
});
