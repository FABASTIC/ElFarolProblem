import { useEffect, useRef, useState } from "react";
import { parseCsv } from "./csv";
import { attachMinds, framesFromRows, type Frame } from "./model";

async function optionalText(url: string, signal: AbortSignal): Promise<string | null> {
  const response = await fetch(url, { cache: "no-store", signal });
  if (response.status === 204 || response.status === 404 || !response.ok) return null;
  return response.text();
}

export type FeedState = "connecting" | "live" | "missing" | "error";

export interface Polled<T> {
  data: T | null;
  state: FeedState;
  receivedAt: number | null;
  error: string | null;
}

export function usePolledJson<T>(path: string, intervalMs: number): Polled<T> {
  const [snapshot, setSnapshot] = useState<Polled<T>>({ data: null, state: "connecting", receivedAt: null, error: null });
  const lastText = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const controller = new AbortController();

    const tick = async () => {
      try {
        const response = await fetch(`/data/${path}?optional=1`, { cache: "no-store", signal: controller.signal });
        if (response.status === 204 || response.status === 404) {
          if (!cancelled) {
            lastText.current = null;
            setSnapshot({ data: null, state: "missing", receivedAt: null, error: null });
          }
        } else if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        } else {
          const text = await response.text();
          if (!cancelled) {
            if (text !== lastText.current) {
              const parsed = JSON.parse(text) as T;
              lastText.current = text;
              setSnapshot({ data: parsed, state: "live", receivedAt: Date.now(), error: null });
            } else {
              setSnapshot((prev) => ({ ...prev, state: "live", receivedAt: Date.now(), error: null }));
            }
          }
        }
      } catch (error) {
        if (cancelled || (error instanceof DOMException && error.name === "AbortError")) return;
        setSnapshot((prev) => ({ ...prev, state: prev.data ? prev.state : "error", error: String(error) }));
      } finally {
        if (!cancelled) timer = window.setTimeout(tick, document.hidden ? intervalMs * 4 : intervalMs);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, [path, intervalMs]);

  return snapshot;
}

export type ReplayState = "idle" | "loading" | "ready" | "missing" | "error";

const replayCache = new Map<string, Frame[]>();

export function useReplay(trialDir: string | null, revision: string): { frames: Frame[]; state: ReplayState } {
  const cacheKey = trialDir ? `${trialDir}@${revision}` : null;
  const [result, setResult] = useState<{ key: string | null; frames: Frame[]; state: ReplayState }>({
    key: null,
    frames: [],
    state: "idle",
  });

  useEffect(() => {
    if (!trialDir || !cacheKey) {
      setResult({ key: null, frames: [], state: "idle" });
      return;
    }
    const cached = replayCache.get(cacheKey);
    if (cached) {
      setResult({ key: cacheKey, frames: cached, state: "ready" });
      return;
    }
    const controller = new AbortController();
    setResult((prev) => ({ key: cacheKey, frames: prev.key === cacheKey ? prev.frames : [], state: "loading" }));
    const base = `/data/${encodeURIComponent(trialDir)}`;
    Promise.all([
      optionalText(`${base}/agent_epoch_details.csv?optional=1`, controller.signal),
      optionalText(`${base}/mind_trace.csv?optional=1`, controller.signal),
      optionalText(`${base}/minds.json?optional=1`, controller.signal),
    ])
      .then(([agentsText, traceText, mindsText]) => {
        if (agentsText == null) {
          setResult({ key: cacheKey, frames: [], state: "missing" });
          return;
        }
        let minds = null;
        try {
          minds = mindsText ? JSON.parse(mindsText) : null;
        } catch {
          minds = null;
        }
        const frames = attachMinds(framesFromRows(parseCsv(agentsText)), traceText ? parseCsv(traceText) : [], minds);
        replayCache.set(cacheKey, frames);
        setResult({ key: cacheKey, frames, state: frames.length ? "ready" : "missing" });
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setResult({ key: cacheKey, frames: [], state: "error" });
      });
    return () => controller.abort();
  }, [trialDir, cacheKey]);

  return { frames: result.key === cacheKey ? result.frames : [], state: result.key === cacheKey ? result.state : trialDir ? "loading" : "idle" };
}
