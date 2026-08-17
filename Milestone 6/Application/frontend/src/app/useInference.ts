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
  const [loadedDate, setLoadedDate] = useState<string>();
  const geometryRef = useRef<RegionGeometryResponse | undefined>(undefined);

  const load = useCallback(async () => {
    if (!predictionDate) return;
    setIsLoading(true);
    setError(undefined);
    setRiskMap(undefined);
    setPrediction(undefined);
    setValidation(undefined);
    setSelectedCellId(undefined);
    setValidationError(undefined);
    setIsLoadingValidation(true);
    setIsLoadingDetail(true);
    try {
      const validationRequest = service.getDailyValidation(predictionDate)
        .then((value) => ({ value, error: undefined }))
        .catch((caught) => ({ value: undefined, error: errorDetails(caught).message }));
      const [nextGeometry, nextRiskMap, validationResult] = await Promise.all([
        geometryRef.current ? Promise.resolve(geometryRef.current) : service.getGeometry(),
        service.getRiskMap(predictionDate),
        validationRequest,
      ]);
      geometryRef.current = nextGeometry;
      const firstCellId = nextRiskMap.items[0]?.areaId;
      let nextPrediction: PredictionResponse | undefined;
      if (firstCellId) {
        try {
          nextPrediction = await service.getPrediction(firstCellId, predictionDate);
        } catch (caught) {
          setError(errorDetails(caught));
        }
      }
      // Publish the complete initial result together so the map, ranking,
      // validation toggle, and selected-cell explanation do not pop in apart.
      setGeometry(nextGeometry);
      setRiskMap(nextRiskMap);
      setValidation(validationResult.value);
      setValidationError(validationResult.error);
      setSelectedCellId(firstCellId);
      setPrediction(nextPrediction);
      setLoadedDate(predictionDate);
    } catch (caught) {
      setError(errorDetails(caught));
      setRiskMap(undefined);
      setLoadedDate(predictionDate);
    } finally {
      setIsLoading(false);
      setIsLoadingValidation(false);
      setIsLoadingDetail(false);
    }
  }, [predictionDate, service]);

  useEffect(() => {
    void load();
  }, [load]);

  const selectCell = useCallback(async (cellId: string) => {
    if (!predictionDate) return;
    setSelectedCellId(cellId);
    setError(undefined);
    setPrediction(undefined);
    setIsLoadingDetail(true);
    try {
      setPrediction(await service.getPrediction(cellId, predictionDate));
    } catch (caught) {
      setError(errorDetails(caught));
    } finally {
      setIsLoadingDetail(false);
    }
  }, [predictionDate, service]);

  const isTransitioning = Boolean(predictionDate && loadedDate !== predictionDate);

  return {
    geometry,
    riskMap: isTransitioning ? undefined : riskMap,
    prediction,
    validation,
    selectedCellId,
    error,
    isLoading: isLoading || isTransitioning,
    isLoadingDetail,
    isLoadingValidation,
    validationError,
    retry: load,
    selectCell,
  };
}
