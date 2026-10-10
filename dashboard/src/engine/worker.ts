import { runSweep, type SweepConfig } from "./town";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";

export type WorkerRequest = { type: "launch"; config: SweepConfig } | { type: "stop" };

export type WorkerEvent =
  | { type: "file"; path: string; text: string }
  | { type: "log"; line: string }
  | { type: "state"; state: "running" | "analyzing" | "exited" | "failed"; exitCode: number | null; error?: string };

interface PyodideRuntime {
  loadPackage(names: string[]): Promise<void>;
  runPython(code: string): unknown;
  FS: {
    mkdirTree(path: string): void;
    writeFile(path: string, data: string): void;
    readFile(path: string, options: { encoding: "utf8" }): string;
  };
}

const scope = self as unknown as {
  postMessage(message: WorkerEvent): void;
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  location: Location;
};

const files = new Map<string, string>();
let stopping = false;
let started = false;

function emit(message: WorkerEvent) {
  scope.postMessage(message);
}

function log(line: string) {
  emit({ type: "log", line });
}

function write(path: string, text: string) {
  files.set(path, text);
  emit({ type: "file", path, text });
}

async function analyze() {
  emit({ type: "state", state: "analyzing", exitCode: 0 });
  log("[launcher] running analyzer.py on the sweep");
  const module = (await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`)) as { loadPyodide(options: { indexURL: string }): Promise<PyodideRuntime> };
  const py = await module.loadPyodide({ indexURL: PYODIDE_URL });
  await py.loadPackage(["numpy", "pandas"]);
  const response = await fetch(new URL("/engine/analyzer.py", scope.location.origin));
  if (!response.ok) throw new Error(`analyzer.py unavailable (HTTP ${response.status})`);
  py.FS.mkdirTree("/elfarol");
  py.FS.writeFile("/elfarol/analyzer.py", await response.text());
  for (const [path, text] of files) {
    if (path === "live_state.json") continue;
    const target = `/experiment/${path}`;
    py.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")));
    py.FS.writeFile(target, text);
  }
  py.runPython(
    [
      "import sys",
      "sys.path.insert(0, '/elfarol')",
      "import analyzer",
      "report = analyzer.build_report('/experiment')",
      "analyzer.write_report(report, '/experiment/analytics_report.json')",
    ].join("\n"),
  );
  write("analytics_report.json", py.FS.readFile("/experiment/analytics_report.json", { encoding: "utf8" }));
  log("[launcher] analytics report ready");
}

async function launch(config: SweepConfig) {
  emit({ type: "state", state: "running", exitCode: null });
  let outcome: "complete" | "interrupted";
  try {
    outcome = await runSweep(config, {
      write,
      log,
      stopRequested: () => stopping,
      rest: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    });
  } catch (error) {
    log(`[launcher] experiment failed: ${String(error)}`);
    emit({ type: "state", state: "failed", exitCode: 1, error: String(error) });
    return;
  }
  log(`[launcher] experiment exited with ${outcome === "complete" ? 0 : 130}`);
  if (outcome === "complete" || files.has("comparison.json")) {
    try {
      await analyze();
    } catch (error) {
      log(`[launcher] analyzer could not run: ${String(error)}`);
    }
  }
  emit({ type: "state", state: "exited", exitCode: outcome === "complete" ? 0 : 130 });
}

scope.onmessage = (event) => {
  const message = event.data;
  if (message.type === "stop") {
    stopping = true;
    return;
  }
  if (message.type === "launch" && !started) {
    started = true;
    void launch(message.config);
  }
};
