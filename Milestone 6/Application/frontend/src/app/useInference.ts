import { useCallback, useEffect, useRef, useState } from "react";
import type { DailyValidationResponse, PredictionResponse, RegionGeometryResponse, RiskMapResponse } from "../domain/inference";
import { InferenceApiError, type HttpInferenceService } from "../services/inference";

function errorDetails(error: unknown) {
  if (error instanceof InferenceApiError) return { message: error.message, code: error.code };
  if (error instanceof Error) return { message: error.message, code: "inference_failed" };
  return { message: "Prediction scoring is temporarily unavailable.", code: "inference_failed" };
}

export function useInference(service: HttpInferenceService, predictionDate?: string) {
  const [geometry, setGeometry] = useState<RegionGeometryResponse>();
  const [riskMap, setRiskMap] = useState<RiskMapResponse>();
  const [prediction, setPrediction] = useState<PredictionResponse>();
  const [validation, setValidation] = useState<DailyValidationResponse>();
  const [selectedCellId, setSelectedCellId] = useState<string>();
  const [error, setError] = useState<{ message: string; code: string }>();
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [isLoadingValidation, setIsLoadingValidation] = useState(false);
  const [validationError, setValidationError] = useState<string>();
  const geometryRef = useRef<RegionGeometryResponse | undefined>(undefined);

  const load = useCallback(async () => {
    if (!predictionDate) return;
    setIsLoading(true);
    setError(undefined);
    setPrediction(undefined);
    setValidation(undefined);
    setValidationError(undefined);
    try {
      const [nextGeometry, nextRiskMap] = await Promise.all([
        geometryRef.current ? Promise.resolve(geometryRef.current) : service.getGeometry(),
        service.getRiskMap(predictionDate),
      ]);
      geometryRef.current = nextGeometry;
      setGeometry(nextGeometry);
      setRiskMap(nextRiskMap);
      setIsLoadingValidation(true);
      try {
        setValidation(await service.getDailyValidation(predictionDate));
      } catch (caught) {
        setValidationError(errorDetails(caught).message);
      } finally {
        setIsLoadingValidation(false);
      }
      const firstCellId = nextRiskMap.items[0]?.areaId;
      setSelectedCellId(firstCellId);
      if (firstCellId) {
        setIsLoadingDetail(true);
        try {
          setPrediction(await service.getPrediction(firstCellId, predictionDate));
        } catch (caught) {
          setError(errorDetails(caught));
        } finally {
          setIsLoadingDetail(false);
        }
      }
    } catch (caught) {
      setError(errorDetails(caught));
      setRiskMap(undefined);
    } finally {
      setIsLoading(false);
    }
  }, [predictionDate, service]);

  useEffect(() => {
    void load();
  }, [load]);

  const selectCell = useCallback(async (cellId: string) => {
    if (!predictionDate) return;
    setSelectedCellId(cellId);
    setIsLoadingDetail(true);
    try {
      setPrediction(await service.getPrediction(cellId, predictionDate));
    } catch (caught) {
      setError(errorDetails(caught));
    } finally {
      setIsLoadingDetail(false);
    }
  }, [predictionDate, service]);

  return {
    geometry,
    riskMap,
    prediction,
    validation,
    selectedCellId,
    error,
    isLoading,
    isLoadingDetail,
    isLoadingValidation,
    validationError,
    retry: load,
    selectCell,
  };
}
