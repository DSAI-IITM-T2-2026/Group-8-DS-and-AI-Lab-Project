import { useCallback, useEffect, useState } from "react";
import type { PipelineConfig, PipelineRun } from "../domain/pipeline";
import { PipelineApiError, type HttpPipelineService } from "../services/pipeline";

const RUN_KEY = "wildfire-iq-pipeline-run";
const ACTIVE = new Set(["queued", "running", "waiting_external"]);

function messageFrom(error: unknown) {
  return error instanceof PipelineApiError || error instanceof Error ? error.message : "The forecasting service is unavailable.";
}

export function usePipelineRun(service: HttpPipelineService) {
  const [config, setConfig] = useState<PipelineConfig>();
  const [selectedDate, setSelectedDate] = useState("");
  const [run, setRun] = useState<PipelineRun>();
  const [error, setError] = useState<string>();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);

  useEffect(() => {
    let cancelled = false;
    service.getConfig().then((next) => {
      if (cancelled) return;
      setConfig(next);
      setSelectedDate((current) => current || next.maxPredictionDate);
    }).catch((caught) => !cancelled && setError(messageFrom(caught)));
    const saved = localStorage.getItem(RUN_KEY);
    if (saved) service.getRun(saved).then((next) => !cancelled && setRun(next)).catch(() => localStorage.removeItem(RUN_KEY));
    return () => { cancelled = true; };
  }, [service]);

  useEffect(() => {
    if (!run) return;
    localStorage.setItem(RUN_KEY, run.runId);
    if (!ACTIVE.has(run.status)) return;
    const timer = window.setInterval(() => {
      service.getRun(run.runId).then(setRun).catch((caught) => setError(messageFrom(caught)));
    }, 5000);
    return () => window.clearInterval(timer);
  }, [run, service]);

  const start = useCallback(async () => {
    if (!selectedDate) return;
    setIsSubmitting(true); setError(undefined);
    try {
      const next = await service.createRun(selectedDate);
      setRun(next);
      localStorage.setItem(RUN_KEY, next.runId);
    } catch (caught) { setError(messageFrom(caught)); }
    finally { setIsSubmitting(false); }
  }, [selectedDate, service]);

  const retry = useCallback(async () => { setRun(undefined); await start(); }, [start]);
  const stop = useCallback(async () => {
    if (!run || !ACTIVE.has(run.status)) return;
    setIsStopping(true);
    setError(undefined);
    try {
      setRun(await service.cancelRun(run.runId));
    } catch (caught) {
      setError(messageFrom(caught));
    } finally {
      setIsStopping(false);
    }
  }, [run, service]);
  const selectDate = useCallback((value: string) => { setSelectedDate(value); setRun(undefined); setError(undefined); localStorage.removeItem(RUN_KEY); }, []);
  return { config, selectedDate, run, error, isSubmitting, isStopping, start, retry, stop, selectDate };
}
