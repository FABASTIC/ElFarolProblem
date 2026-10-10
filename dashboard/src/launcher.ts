import { useCallback, useEffect, useRef, useState } from "react";
import { request } from "./engine/backend";
import type { LaunchConfig, LauncherStatus } from "./setup";

export const ACTIVE_STATES = new Set(["launching", "running", "stopping", "analyzing", "detached"]);

export interface Launcher {
  status: LauncherStatus | null;
  reachable: boolean | null;
  pending: boolean;
  error: string | null;
  launch: (config: LaunchConfig) => Promise<boolean>;
  stop: () => Promise<boolean>;
  probe: () => Promise<void>;
  clearError: () => void;
}

async function post(path: string, body: unknown): Promise<{ ok: boolean; data: LauncherStatus | { error?: string } | null }> {
  const response = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  return { ok: response.ok, data };
}

export function useLauncher(): Launcher {
  const [status, setStatus] = useState<LauncherStatus | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const kick = useRef<() => void>(() => undefined);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const controller = new AbortController();
    const tick = async () => {
      if (timer) window.clearTimeout(timer);
      let next = 4000;
      try {
        const response = await request("/api/sim/status", { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as LauncherStatus;
        if (cancelled) return;
        setStatus(data);
        setReachable(true);
        next = ACTIVE_STATES.has(data.state) ? 1200 : data.capabilities.llm.available === null ? 2000 : 4000;
      } catch (err) {
        if (cancelled || (err instanceof DOMException && err.name === "AbortError")) return;
        setReachable(false);
        next = 8000;
      }
      if (!cancelled) timer = window.setTimeout(tick, document.hidden ? next * 3 : next);
    };
    kick.current = () => void tick();
    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  const run = useCallback(async (path: string, body: unknown) => {
    setPending(true);
    setError(null);
    try {
      const { ok, data } = await post(path, body);
      if (!ok) {
        setError((data as { error?: string } | null)?.error ?? "The launcher refused the request.");
        return false;
      }
      if (data && "state" in data) setStatus(data as LauncherStatus);
      return true;
    } catch (err) {
      setError(`Launcher unreachable: ${String(err)}`);
      return false;
    } finally {
      setPending(false);
      kick.current();
    }
  }, []);

  const launch = useCallback((config: LaunchConfig) => run("/api/sim/launch", config), [run]);
  const stop = useCallback(() => run("/api/sim/stop", {}), [run]);
  const probe = useCallback(async () => {
    await run("/api/sim/probe", {});
  }, [run]);
  const clearError = useCallback(() => setError(null), []);

  return { status, reachable, pending, error, launch, stop, probe, clearError };
}
