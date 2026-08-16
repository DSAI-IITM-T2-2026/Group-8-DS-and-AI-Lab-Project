import { useEffect, useState } from "react";
import type { ModelEvaluationResponse } from "../domain/inference";
import type { HttpInferenceService } from "../services/inference";

export function useModelEvaluation(service: HttpInferenceService) {
  const [evaluation, setEvaluation] = useState<ModelEvaluationResponse>();
  const [isLoading, setIsLoading] = useState(true);
  const [isUnavailable, setIsUnavailable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    service.getModelEvaluation()
      .then((result) => {
        if (!cancelled) setEvaluation(result);
      })
      .catch(() => {
        if (!cancelled) setIsUnavailable(true);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => { cancelled = true; };
  }, [service]);

  return { evaluation, isLoading, isUnavailable };
}
