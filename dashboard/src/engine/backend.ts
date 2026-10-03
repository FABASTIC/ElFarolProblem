import { BROWSER_ENGINE, HOSTED } from "../env";
import { sanitizeConfig, type LaunchConfig, type LauncherState, type LauncherStatus } from "../setup";
import type { WorkerEvent, WorkerRequest } from "./worker";

export const BROWSER_PACE_MS = 250;
const LOG_REPLY = 120;
const MIME: Record<string, string> = {
  json: "application/json; charset=utf-8",
  csv: "text/csv; charset=utf-8",
};

interface BrowserRun {
  state: LauncherState;
  config: LaunchConfig;
  startedAt: number;
  endedAt: number | null;
  exitCode: number | null;
  error: string | null;
}

const files = new Map<string, string>();
const staticCache = new Map<string, Promise<{ status: number; text: string | null; type: string | null }>>();
const log: string[] = [];
let mounted = false;
let worker: Worker | null = null;
let run: BrowserRun | null = null;

function cores(): number {
  return typeof navigator !== "undefined" && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 1;
}

function status(): LauncherStatus {
  return {
    state: run?.state ?? "idle",
    engine: run?.config.engine ?? null,
    config: run?.config ?? null,
    pid: null,
    startedAt: run?.startedAt ?? null,
    endedAt: run?.endedAt ?? null,
    exitCode: run?.exitCode ?? null,
    archivedTo: null,
    error: run?.error ?? null,
    log: log.slice(-LOG_REPLY),
    model: "In-browser tensor brains",
    dataDir: "this browser tab",
    capabilities: {
      llm: { available: true, detail: `this device · ${cores()} cores` },
      rehearsal: { available: false, detail: "the paced CPU preview needs the Python engine" },
    },
  };
}

function json(body: unknown, code = 200): Response {
  return new Response(JSON.stringify(body), { status: code, headers: { "Content-Type": "application/json; charset=utf-8" } });
}

function finish(state: LauncherState, exitCode: number | null, error?: string) {
  if (!run) return;
  run.state = state;
  run.exitCode = exitCode;
  run.endedAt = Date.now() / 1000;
  if (error) run.error = error;
  worker?.terminate();
  worker = null;
}

function onEvent(message: WorkerEvent) {
  if (message.type === "file") {
    files.set(message.path, message.text);
    return;
  }
  if (message.type === "log") {
    log.push(message.line);
    log.splice(0, Math.max(0, log.length - LOG_REPLY * 4));
    return;
  }
  if (!run) return;
  if (message.state === "exited" || message.state === "failed") finish(message.state, message.exitCode, message.error);
  else if (run.state !== "stopping") run.state = message.state;
}

function launch(body: unknown): Response {
  const { config, errors } = sanitizeConfig(body);
  if (!config) return json({ error: errors.join(" ") }, 400);
  if (worker) return json({ error: "A sweep is already running. Stop it before starting another." }, 409);
  if (config.engine !== "llm") return json({ error: "The paced rehearsal engine needs the Python engine. Use the tensor brains." }, 412);
  files.clear();
  log.length = 0;
  mounted = true;
  run = { state: "launching", config, startedAt: Date.now() / 1000, endedAt: null, exitCode: null, error: null };
  try {
    worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  } catch (error) {
    finish("failed", null, `Could not start the in-browser engine: ${String(error)}`);
    return json({ error: run.error }, 500);
  }
  worker.onmessage = (event: MessageEvent<WorkerEvent>) => onEvent(event.data);
  worker.onerror = (event) => {
    log.push(`[launcher] engine crashed: ${event.message}`);
    finish("failed", 1, event.message || "The in-browser engine crashed.");
  };
  const request: WorkerRequest = { type: "launch", config: { members: config.members, epochs: config.epochs, seeds: config.seeds, paceMs: BROWSER_PACE_MS } };
  worker.postMessage(request);
  log.push(`[launcher] in-browser engine :: ${config.members} brains x ${config.epochs} nights x ${config.seeds.length * 2} towns on ${cores()} cores`);
  return json(status(), 202);
}

function stop(): Response {
  if (!worker || !run) return json({ error: "Nothing is running." }, 409);
  const message: WorkerRequest = { type: "stop" };
  worker.postMessage(message);
  if (run.state === "running" || run.state === "launching") {
    run.state = "stopping";
    log.push("[launcher] stop requested: finishing the current night, then tearing down");
  }
  return json(status(), 202);
}

async function readBody(init?: RequestInit): Promise<unknown> {
  if (typeof init?.body !== "string" || !init.body) return {};
  try {
    return JSON.parse(init.body);
  } catch {
    return null;
  }
}

async function api(route: string, init?: RequestInit): Promise<Response> {
  const method = (init?.method ?? "GET").toUpperCase();
  if (method === "GET" && route === "status") return json(status());
  if (method !== "POST") return json({ error: "unknown launcher route" }, 404);
  if (route === "launch") return launch(await readBody(init));
  if (route === "stop") return stop();
  if (route === "probe") return json(status());
  return json({ error: "unknown launcher route" }, 404);
}

function served(path: string, optional: boolean): Response {
  const text = files.get(path);
  if (text == null) return new Response(null, { status: optional ? 204 : 404 });
  const ext = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
  return new Response(text, { status: 200, headers: { "Content-Type": MIME[ext] ?? "text/plain; charset=utf-8" } });
}

async function cached(url: string, init?: RequestInit): Promise<Response> {
  const key = url.split("?")[0];
  let entry = staticCache.get(key);
  if (!entry) {
    entry = fetch(url, { cache: "no-store" }).then(async (response) => ({
      status: response.status,
      text: response.status === 204 ? null : await response.text(),
      type: response.headers.get("Content-Type"),
    }));
    staticCache.set(key, entry);
    entry.catch(() => staticCache.delete(key));
  }
  const signal = init?.signal;
  if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
  const result = await entry;
  if (result.status >= 500) staticCache.delete(key);
  return new Response(result.text, { status: result.status, headers: result.type ? { "Content-Type": result.type } : undefined });
}

export function request(input: string, init?: RequestInit): Promise<Response> {
  if (!BROWSER_ENGINE) return fetch(input, init);
  const url = new URL(input, window.location.href);
  if (url.origin !== window.location.origin) return fetch(input, init);
  if (url.pathname.startsWith("/api/sim/")) return api(url.pathname.slice("/api/sim/".length), init);
  if (url.pathname.startsWith("/data/")) {
    if (mounted) return Promise.resolve(served(decodeURIComponent(url.pathname.slice("/data/".length)), url.searchParams.has("optional")));
    return HOSTED ? cached(input, init) : fetch(input, init);
  }
  return fetch(input, init);
}
