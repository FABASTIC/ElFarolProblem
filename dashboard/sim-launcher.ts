import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type { Connect, Plugin } from "vite";
import { sanitizeConfig, type Engine, type EngineCapability, type LaunchConfig, type LauncherState, type LauncherStatus } from "./src/setup";

const LOG_REPLY = 120;
const LOG_WINDOW = 160 * 1024;
const BODY_LIMIT = 16 * 1024;
const STALE_AFTER_S = 900;
const RUNNING_LIVE = new Set(["booting", "loading_engine", "running"]);
const RECORD_FILE = ".launcher.json";
const STOP_FILE = ".stop";
const LOG_FILE = "sweep.log";

interface RunRecord {
  state: LauncherState;
  engine: Engine;
  config: LaunchConfig;
  host: "wsl" | "native";
  pid: number | null;
  startedAt: number;
  endedAt: number | null;
  exitCode: number | null;
  archivedTo: string | null;
  error: string | null;
  logOffset: number;
  analyzed: boolean;
}

interface LauncherOptions {
  repoRoot: string;
  dataDir: string;
}

function hostPython(): string {
  return process.env.ELFAROL_PYTHON || (process.platform === "win32" ? "python" : "python3");
}

function torchHost(): "wsl" | "native" {
  const chosen = process.env.ELFAROL_TORCH_HOST;
  if (chosen === "wsl" || chosen === "native") return chosen;
  return process.platform === "win32" ? "wsl" : "native";
}

function wslDistro(): string {
  return process.env.ELFAROL_WSL_DISTRO || "Ubuntu-24.04";
}

function wslPython(): string {
  return process.env.ELFAROL_WSL_PYTHON || "~/.venv/bin/python";
}

function engineLabel(): string {
  return "Isolated PyTorch Tensors";
}

const TORCH_PROBE = "import torch; print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')";

function toWslPath(target: string): string {
  const resolved = path.resolve(target);
  const drive = /^([A-Za-z]):[\\/](.*)$/.exec(resolved);
  if (!drive) return resolved.replace(/\\/g, "/");
  return `/mnt/${drive[1].toLowerCase()}/${drive[2].replace(/\\/g, "/")}`;
}

function longPath(target: string): string {
  try {
    return fs.realpathSync.native(target);
  } catch {
    return path.resolve(target);
  }
}

function stamp(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function probe(command: string, args: string[], timeoutMs: number): Promise<{ ok: boolean; output: string }> {
  return new Promise((resolve) => {
    let output = "";
    let settled = false;
    const finish = (ok: boolean) => {
      if (settled) return;
      settled = true;
      resolve({ ok, output: output.trim() });
    };
    try {
      const child = spawn(command, args, { windowsHide: true });
      const timer = setTimeout(() => {
        child.kill();
        finish(false);
      }, timeoutMs);
      child.stdout?.on("data", (d: Buffer) => (output += d.toString("utf8")));
      child.stderr?.on("data", (d: Buffer) => (output += d.toString("utf8")));
      child.on("error", (error) => {
        clearTimeout(timer);
        output += error.message;
        finish(false);
      });
      child.on("exit", (code) => {
        clearTimeout(timer);
        finish(code === 0);
      });
    } catch (error) {
      output += String(error);
      finish(false);
    }
  });
}

class Launcher {
  private readonly repoRoot: string;
  private readonly dataDir: string;
  private run: RunRecord | null = null;
  private child: ChildProcess | null = null;
  private forceTimer: NodeJS.Timeout | null = null;
  private capabilities: Record<Engine, EngineCapability> = {
    llm: { available: null, detail: "verified when you press Begin" },
    rehearsal: { available: null, detail: "verified when you press Begin" },
  };
  private probedAt = 0;
  private wslPulse: { pid: number; at: number; alive: boolean | null } | null = null;

  constructor(options: LauncherOptions) {
    this.repoRoot = longPath(options.repoRoot);
    fs.mkdirSync(options.dataDir, { recursive: true });
    this.dataDir = longPath(options.dataDir);
    this.restore();
  }

  private file(name: string) {
    return path.join(this.dataDir, name);
  }

  private restore() {
    try {
      const saved = JSON.parse(fs.readFileSync(this.file(RECORD_FILE), "utf8")) as RunRecord;
      if (["launching", "running", "stopping", "analyzing", "detached"].includes(saved.state)) saved.state = "detached";
      saved.logOffset = saved.logOffset ?? 0;
      saved.analyzed = saved.analyzed ?? false;
      this.run = saved;
    } catch {
      this.run = null;
    }
  }

  private persist() {
    if (!this.run) return;
    try {
      fs.mkdirSync(this.dataDir, { recursive: true });
      fs.writeFileSync(this.file(RECORD_FILE), JSON.stringify(this.run, null, 2));
    } catch {
      return;
    }
  }

  private note(message: string) {
    try {
      fs.appendFileSync(this.file(LOG_FILE), `[launcher] ${message}\n`);
    } catch {
      return;
    }
  }

  private readLog(): string[] {
    const run = this.run;
    if (!run) return [];
    let text = "";
    try {
      const size = fs.statSync(this.file(LOG_FILE)).size;
      const start = Math.max(run.logOffset, size - LOG_WINDOW);
      if (size <= start) return [];
      const buffer = Buffer.alloc(size - start);
      const fd = fs.openSync(this.file(LOG_FILE), "r");
      try {
        fs.readSync(fd, buffer, 0, buffer.length, start);
      } finally {
        fs.closeSync(fd);
      }
      text = buffer.toString("utf8");
    } catch {
      return [];
    }
    const lines = text.split(/\r?\n/).filter((line) => line.trim());
    if (run.pid == null && run.host === "wsl") {
      const found = lines.map((line) => /^ELFAROL_PID=(\d+)$/.exec(line.trim())).find(Boolean);
      if (found) {
        run.pid = Number(found[1]);
        this.persist();
      }
    }
    return lines.filter((line) => !line.startsWith("ELFAROL_PID=")).slice(-LOG_REPLY);
  }

  private liveState(): { status?: string; updated_at?: number } | null {
    try {
      return JSON.parse(fs.readFileSync(this.file("live_state.json"), "utf8"));
    } catch {
      return null;
    }
  }

  private liveRunning(): boolean {
    const live = this.liveState();
    if (!live?.status || !RUNNING_LIVE.has(live.status)) return false;
    return typeof live.updated_at === "number" && Date.now() / 1000 - live.updated_at < STALE_AFTER_S;
  }

  unverified(engine: Engine | undefined): boolean {
    return this.capabilities[engine ?? "llm"]?.available !== true;
  }

  async refreshCapabilities(force = false) {
    if (!force && Date.now() - this.probedAt < 120_000) return;
    this.probedAt = Date.now();
    const python = hostPython();
    const local = await probe(python, ["-c", TORCH_PROBE], 20_000);
    this.capabilities.rehearsal = local.ok
      ? { available: true, detail: `${python} · torch ${local.output}` }
      : { available: false, detail: `GPU ONLY · ${python} has no torch` };
    if (torchHost() === "wsl") {
      const venv = await probe("wsl.exe", ["-d", wslDistro(), "-e", "bash", "-lc", `test -x ${wslPython()} && ${wslPython()} -c "${TORCH_PROBE}"`], 90_000);
      this.capabilities.llm = venv.ok && venv.output.endsWith("cuda")
        ? { available: true, detail: `WSL ${wslDistro()} · torch ${venv.output}` }
        : { available: false, detail: `WSL ${wslDistro()} has no CUDA torch at ${wslPython()}` };
    } else {
      const native = await probe(python, ["-c", TORCH_PROBE], 60_000);
      this.capabilities.llm = native.ok && native.output.endsWith("cuda")
        ? { available: true, detail: `${python} · torch ${native.output}` }
        : { available: false, detail: `${python} has no CUDA torch` };
    }
  }

  private settleDetached() {
    const run = this.run;
    if (!run || run.state !== "detached") return;
    if (run.host === "native" && run.pid) {
      if (alive(run.pid)) return;
    } else {
      const pulse = this.wslPulse;
      if (run.pid && (!pulse || pulse.pid !== run.pid || Date.now() - pulse.at > 10_000)) {
        const pid = run.pid;
        this.wslPulse = { pid, at: Date.now(), alive: pulse?.pid === pid ? pulse.alive : null };
        void probe("wsl.exe", ["-d", wslDistro(), "-e", "kill", "-0", String(pid)], 30_000).then(({ ok }) => {
          this.wslPulse = { pid, at: Date.now(), alive: ok };
        });
      }
      const known = this.wslPulse && this.wslPulse.pid === run.pid ? this.wslPulse.alive : null;
      if (known === true) return;
      if (known === null && this.liveRunning()) return;
      if (known === null && !this.liveState() && Date.now() / 1000 - run.startedAt < 300) return;
    }
    run.state = "exited";
    run.endedAt = run.endedAt ?? Date.now() / 1000;
    this.note("sweep finished while the dashboard was away");
    this.settleLiveState();
    fs.rmSync(this.file(STOP_FILE), { force: true });
    if (!run.analyzed && fs.existsSync(this.file("comparison.json"))) this.analyze(true);
    else this.persist();
  }

  status(): LauncherStatus {
    this.settleDetached();
    const run = this.run;
    return {
      state: run?.state ?? "idle",
      engine: run?.engine ?? null,
      config: run?.config ?? null,
      pid: run?.pid ?? null,
      startedAt: run?.startedAt ?? null,
      endedAt: run?.endedAt ?? null,
      exitCode: run?.exitCode ?? null,
      archivedTo: run?.archivedTo ?? null,
      error: run?.error ?? null,
      log: this.readLog(),
      model: engineLabel(),
      dataDir: this.dataDir,
      capabilities: this.capabilities,
    };
  }

  private busy(): boolean {
    const state = this.status().state;
    if (["launching", "running", "stopping", "analyzing", "detached"].includes(state)) return true;
    return this.liveRunning();
  }

  private archive(): string | null {
    const entries = fs.readdirSync(this.dataDir).filter((name) => name !== RECORD_FILE && name !== STOP_FILE);
    if (!entries.length) return null;
    const target = path.join(path.dirname(this.dataDir), `${path.basename(this.dataDir)}_archive`, stamp());
    fs.mkdirSync(target, { recursive: true });
    for (const name of entries) fs.renameSync(this.file(name), path.join(target, name));
    return target;
  }

  private experimentArgs(config: LaunchConfig, outputDir: string, stopFile: string): string[] {
    const args = [
      "-u",
      "experiment.py",
      "--output-dir",
      outputDir,
      "--agents",
      String(config.members),
      "--epochs",
      String(config.epochs),
      "--seeds",
      ...config.seeds.map(String),
      "--stop-file",
      stopFile,
    ];
    if (config.engine === "rehearsal") args.push("--device", "cpu", "--pace", String(config.pace));
    else args.push("--device", "cuda");
    return args;
  }

  private spawnLogged(command: string, args: string[], cwd?: string): ChildProcess {
    const fd = fs.openSync(this.file(LOG_FILE), "a");
    try {
      const child = spawn(command, args, {
        cwd,
        detached: true,
        windowsHide: true,
        stdio: ["ignore", fd, fd],
        env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
      });
      child.unref();
      return child;
    } finally {
      fs.closeSync(fd);
    }
  }

  launch(input: unknown): { code: number; body: unknown } {
    const { config, errors } = sanitizeConfig(input);
    if (!config) return { code: 400, body: { error: errors.join(" ") } };
    if (this.busy()) return { code: 409, body: { error: "A sweep is already running. Stop it before starting another." } };
    const capability = this.capabilities[config.engine];
    if (capability.available === false) return { code: 412, body: { error: `${config.engine.toUpperCase()} engine unavailable: ${capability.detail}` } };
    let archivedTo: string | null = null;
    try {
      fs.mkdirSync(this.dataDir, { recursive: true });
      fs.rmSync(this.file(STOP_FILE), { force: true });
      archivedTo = this.archive();
    } catch (error) {
      return { code: 500, body: { error: `Could not archive the previous run: ${String(error)}` } };
    }
    const host = config.engine === "llm" ? torchHost() : "native";
    this.run = {
      state: "launching",
      engine: config.engine,
      config,
      host,
      pid: null,
      startedAt: Date.now() / 1000,
      endedAt: null,
      exitCode: null,
      archivedTo,
      error: null,
      logOffset: 0,
      analyzed: false,
    };
    if (archivedTo) this.note(`previous run archived to ${archivedTo}`);
    let child: ChildProcess;
    try {
      if (host === "wsl") {
        const args = this.experimentArgs(config, toWslPath(this.dataDir), toWslPath(this.file(STOP_FILE)));
        const script = `exec >>"$1" 2>&1 && cd "$2" && shift 2 && echo ELFAROL_PID=$$ && export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 && exec ${wslPython()} "$@"`;
        this.note(`wsl ${wslDistro()} :: ${wslPython()} ${args.join(" ")}`);
        child = spawn("wsl.exe", ["-d", wslDistro(), "-e", "bash", "-lc", script, "elfarol", toWslPath(this.file(LOG_FILE)), toWslPath(this.repoRoot), ...args], {
          detached: true,
          windowsHide: true,
          stdio: "ignore",
        });
        child.unref();
      } else {
        const args = this.experimentArgs(config, this.dataDir, this.file(STOP_FILE));
        this.note(`${hostPython()} ${args.join(" ")}`);
        child = this.spawnLogged(hostPython(), args, this.repoRoot);
        this.run.pid = child.pid ?? null;
      }
    } catch (error) {
      this.run.state = "failed";
      this.run.error = String(error);
      this.persist();
      return { code: 500, body: { error: this.run.error } };
    }
    this.child = child;
    this.run.state = "running";
    this.persist();
    child.on("error", (error) => {
      if (!this.run) return;
      this.run.error = error.message;
      this.note(`process error: ${error.message}`);
      this.persist();
    });
    child.on("exit", (code, signal) => this.onExit(child, code, signal));
    return { code: 202, body: this.status() };
  }

  private onExit(child: ChildProcess, code: number | null, signal: NodeJS.Signals | null) {
    if (this.child !== child || !this.run) return;
    this.child = null;
    if (this.forceTimer) clearTimeout(this.forceTimer);
    this.forceTimer = null;
    this.run.exitCode = code;
    this.run.endedAt = Date.now() / 1000;
    this.note(`experiment exited with ${code ?? signal}`);
    fs.rmSync(this.file(STOP_FILE), { force: true });
    this.settleLiveState();
    const ok = code === 0 || code === 2 || code === 130;
    if (!ok && code !== null && !this.run.error) this.run.error = `experiment.py exited with code ${code}`;
    if (fs.existsSync(this.file("comparison.json"))) this.analyze(ok);
    else {
      this.run.state = ok ? "exited" : "failed";
      this.persist();
    }
  }

  private settleLiveState() {
    const live = this.liveState() as Record<string, unknown> | null;
    if (!live || typeof live.status !== "string" || !RUNNING_LIVE.has(live.status)) return;
    live.status = "interrupted";
    live.updated_at = Date.now() / 1000;
    try {
      fs.writeFileSync(this.file("live_state.json"), JSON.stringify(live, null, 2));
    } catch {
      return;
    }
  }

  private analyze(ok: boolean) {
    const run = this.run;
    if (!run) return;
    run.state = "analyzing";
    run.analyzed = true;
    this.persist();
    this.note("running analyzer.py on the sweep");
    const done = (code: number | null) => {
      if (this.run !== run || run.state !== "analyzing") return;
      this.note(code === 0 ? "analytics report ready" : `analyzer exited with ${code}`);
      run.state = ok ? "exited" : "failed";
      this.persist();
    };
    try {
      const analyzer = this.spawnLogged(hostPython(), ["-u", "analyzer.py", "--experiment-dir", this.dataDir, "--quiet"], this.repoRoot);
      analyzer.on("error", (error) => {
        this.note(`analyzer could not start: ${error.message}`);
        done(null);
      });
      analyzer.on("exit", done);
    } catch (error) {
      this.note(`analyzer could not start: ${String(error)}`);
      done(null);
    }
  }

  private signal(kind: "INT" | "KILL") {
    const run = this.run;
    if (!run?.pid) return;
    if (run.host === "wsl") {
      spawn("wsl.exe", ["-d", wslDistro(), "-e", "kill", `-${kind}`, String(run.pid)], { windowsHide: true }).on("error", () => undefined);
    } else if (process.platform !== "win32") {
      try {
        process.kill(run.pid, kind === "INT" ? "SIGINT" : "SIGKILL");
      } catch {
        return;
      }
    } else if (kind === "KILL") {
      try {
        process.kill(run.pid);
      } catch {
        return;
      }
    }
  }

  stop(): { code: number; body: unknown } {
    const run = this.run;
    const state = this.status().state;
    const active = !!run && ["launching", "running", "stopping", "detached"].includes(state);
    if (!active && !this.liveRunning()) return { code: 409, body: { error: "Nothing is running." } };
    try {
      fs.writeFileSync(this.file(STOP_FILE), String(Date.now()));
    } catch (error) {
      return { code: 500, body: { error: `Could not write the stop file: ${String(error)}` } };
    }
    if (!run || !active) return { code: 202, body: this.status() };
    if (run.state !== "stopping") this.note("stop requested: finishing the current night, then tearing down");
    if (run.state !== "detached") run.state = "stopping";
    this.persist();
    this.signal("INT");
    if (!this.forceTimer) {
      this.forceTimer = setTimeout(() => {
        this.forceTimer = null;
        const current = this.run;
        if (current !== run || !["stopping", "detached"].includes(current.state)) return;
        if (current.host === "native" && current.pid && !alive(current.pid)) return;
        this.note("graceful stop timed out; forcing the process down");
        this.signal("KILL");
        this.child?.kill();
      }, run.host === "wsl" ? 120_000 : 20_000);
    }
    return { code: 202, body: this.status() };
  }
}

function readBody(req: Connect.IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > BODY_LIMIT) {
        reject(new Error("payload too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : {});
      } catch {
        reject(new Error("invalid JSON"));
      }
    });
    req.on("error", reject);
  });
}

function sameOrigin(req: Connect.IncomingMessage): boolean {
  const origin = req.headers.origin;
  if (!origin) return true;
  try {
    return new URL(origin).host === req.headers.host;
  } catch {
    return false;
  }
}

export function simLauncher(options: LauncherOptions): Plugin {
  const launcher = new Launcher(options);
  const handler: Connect.NextHandleFunction = (req, res) => {
    const send = (code: number, body: unknown) => {
      res.statusCode = code;
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.setHeader("Cache-Control", "no-store");
      res.end(JSON.stringify(body));
    };
    const route = (req.url ?? "/").split("?")[0];
    if (req.method === "GET" && route === "/status") {
      send(200, launcher.status());
      return;
    }
    if (req.method !== "POST" || !["/launch", "/stop", "/probe"].includes(route)) {
      send(404, { error: "unknown launcher route" });
      return;
    }
    if (!sameOrigin(req) || !(req.headers["content-type"] ?? "").includes("application/json")) {
      send(403, { error: "launcher accepts same-origin JSON requests only" });
      return;
    }
    readBody(req)
      .then(async (body) => {
        if (route === "/probe") {
          await launcher.refreshCapabilities(true);
          send(200, launcher.status());
          return;
        }
        if (route === "/launch" && launcher.unverified((body as { engine?: Engine } | null)?.engine)) await launcher.refreshCapabilities(true);
        const result = route === "/launch" ? launcher.launch(body) : launcher.stop();
        send(result.code, result.body);
      })
      .catch((error: Error) => send(400, { error: error.message }));
  };
  return {
    name: "elfarol-sim-launcher",
    configureServer(server) {
      server.middlewares.use("/api/sim", handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/api/sim", handler);
    },
  };
}
