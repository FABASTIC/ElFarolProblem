import {
  ARMOR_DEFAULTS,
  BrainColony,
  COMFORT_THRESHOLD_RATIO,
  OPTIONS,
  UTILITY_AT_BAR_COMFORTABLE,
  UTILITY_AT_BAR_OVERCROWDED,
  UTILITY_AT_HOME,
  round,
  type BrainAction,
  type TownAgent,
} from "./brains";
import { Rng } from "./rng";

export const BROWSER_MODEL = "isolated tensor brains :: speaker + actor DQN per agent :: in-browser port";

const CONDITIONS = [
  { label: "control", broadcastEnabled: false },
  { label: "delta2", broadcastEnabled: true },
] as const;

const EPOCH_FIELDS = [
  "epoch", "bar_attendance", "bar_capacity", "comfort_threshold",
  "attendance_over_threshold", "total_utility", "mean_utility",
  "deception_index", "truthfulness_ratio", "deception_count",
  "truthful_count", "broadcast_count", "broadcast_correlation",
  "town_broadcast_ratio", "actual_attendance", "intended_attendance",
];

const AGENT_FIELDS = [
  "epoch", "agent_id", "x", "y", "in_bar", "utility",
  "cumulative_utility", "stated_intention", "actual_target",
  "actual_location", "is_deceptive", "broadcast",
];

const TRACE_FIELDS = [
  "epoch", "agent_id", "archetype", "forecast", "forecast_confidence", "active_predictor",
  "p_honest_go", "p_honest_stay", "p_false_go", "p_false_stay", "instinct",
  "stated_intention", "actual_target", "in_bar", "lied", "rerouted", "utility",
  "top_emotion", "top_emotion_intensity", "trust_given_mean", "reputation", "private_note",
];

const FLOAT_FIELDS = new Set([
  "total_utility", "mean_utility", "deception_index", "truthfulness_ratio", "broadcast_correlation",
  "town_broadcast_ratio", "utility", "cumulative_utility", "forecast", "forecast_confidence",
  "p_honest_go", "p_honest_stay", "p_false_go", "p_false_stay", "top_emotion_intensity",
  "trust_given_mean", "reputation",
]);

type Cell = [number, number];

export interface SweepConfig {
  members: number;
  epochs: number;
  seeds: number[];
  gridSize?: number;
  paceMs?: number;
}

export interface SweepIO {
  write(path: string, text: string): void;
  log(line: string): void;
  stopRequested(): boolean;
  rest(ms: number): Promise<void>;
}

class StopSignal extends Error {}

function pyValue(key: string, value: unknown): string {
  if (value == null) return "";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") {
    if (FLOAT_FIELDS.has(key) && Number.isInteger(value)) return value.toFixed(1);
    return String(value);
  }
  return String(value);
}

function csvCell(text: string): string {
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function toCsv(fields: readonly string[], rows: Record<string, unknown>[]): string {
  const lines = [fields.join(",")];
  for (const row of rows) lines.push(fields.map((key) => csvCell(pyValue(key, row[key]))).join(","));
  return `${lines.join("\r\n")}\r\n`;
}

class Grid {
  readonly cells: Int32Array;
  readonly barMin: number;
  readonly barMax: number;

  constructor(readonly size: number) {
    this.cells = new Int32Array(size * size);
    this.barMin = Math.floor((size - 10) / 2);
    this.barMax = this.barMin + 10;
  }

  at(x: number, y: number): number {
    return this.cells[y * this.size + x];
  }

  place(id: number, x: number, y: number) {
    this.cells[y * this.size + x] = id;
  }

  move(id: number, ox: number, oy: number, nx: number, ny: number) {
    this.cells[oy * this.size + ox] = 0;
    this.cells[ny * this.size + nx] = id;
  }

  isInBar(x: number, y: number): boolean {
    return this.barMin <= x && x < this.barMax && this.barMin <= y && y < this.barMax;
  }

  occupancy(): number {
    let count = 0;
    for (let y = this.barMin; y < this.barMax; y++) for (let x = this.barMin; x < this.barMax; x++) if (this.at(x, y) !== 0) count++;
    return count;
  }
}

function resolveTargets(agents: TownAgent[], actions: BrainAction[], grid: Grid, rng: Rng): { resolved: BrainAction[]; rerouted: boolean[] } {
  const key = (x: number, y: number) => y * grid.size + x;
  const occupied = new Set(agents.map((a) => key(a.x, a.y)));
  const reserved = new Set<number>();
  const resolved = actions.map((a) => ({ ...a, move: [a.move[0], a.move[1]] as [number, number] }));
  const rerouted = new Array(agents.length).fill(false);
  const barCells: Cell[] = [];
  for (let x = grid.barMin; x < grid.barMax; x++) for (let y = grid.barMin; y < grid.barMax; y++) barCells.push([x, y]);
  const outsideCells: Cell[] = [];
  for (let x = 0; x < grid.size; x++) for (let y = 0; y < grid.size; y++) if (!grid.isInBar(x, y)) outsideCells.push([x, y]);
  const order = rng.shuffle(agents.map((_, i) => i));
  const free = (c: Cell) => !occupied.has(key(c[0], c[1])) && !reserved.has(key(c[0], c[1]));
  const nearest = (origin: Cell, candidates: Cell[]): Cell | null => {
    let best: Cell | null = null;
    let bestDistance = Infinity;
    for (const c of candidates) {
      if (!free(c)) continue;
      const d = Math.abs(c[0] - origin[0]) + Math.abs(c[1] - origin[1]);
      if (d < bestDistance || (d === bestDistance && best && (c[0] < best[0] || (c[0] === best[0] && c[1] < best[1])))) {
        best = c;
        bestDistance = d;
      }
    }
    return best;
  };
  for (const i of order) {
    const agent = agents[i];
    const here: Cell = [agent.x, agent.y];
    const wanted: Cell = [resolved[i].move[0], resolved[i].move[1]];
    const wantBar = resolved[i].actual_target === "bar";
    const inside = grid.isInBar(here[0], here[1]);
    const same = wanted[0] === here[0] && wanted[1] === here[1];
    let target: Cell;
    if (wantBar) {
      if (inside) target = here;
      else if (grid.isInBar(wanted[0], wanted[1]) && free(wanted)) target = wanted;
      else target = nearest(here, barCells) ?? here;
    } else if (!inside && !grid.isInBar(wanted[0], wanted[1]) && (same || free(wanted))) target = wanted;
    else if (!inside) target = here;
    else target = nearest(here, outsideCells) ?? here;
    if (target[0] !== here[0] || target[1] !== here[1]) reserved.add(key(target[0], target[1]));
    if (target[0] !== wanted[0] || target[1] !== wanted[1]) rerouted[i] = true;
    resolved[i].move = [target[0], target[1]];
  }
  return { resolved, rerouted };
}

function executeActions(agents: (TownAgent & { stated: string; target: string })[], actions: BrainAction[], grid: Grid, rng: Rng) {
  const clamp = (v: number) => Math.max(0, Math.min(grid.size - 1, Math.trunc(v)));
  const order = rng.shuffle(agents.map((_, i) => i));
  for (const i of order) {
    const agent = agents[i];
    const action = actions[i];
    const nx = clamp(action.move[0]);
    const ny = clamp(action.move[1]);
    if ((nx !== agent.x || ny !== agent.y) && grid.at(nx, ny) === 0) {
      grid.move(agent.id, agent.x, agent.y, nx, ny);
      agent.x = nx;
      agent.y = ny;
    }
    agent.stated = action.stated_intention;
    agent.target = action.actual_target;
  }
}

interface AgentRecord {
  agent_id: number;
  x: number;
  y: number;
  in_bar: boolean;
  utility: number;
  cumulative_utility: number;
  stated_intention: string;
  actual_target: string;
  actual_location: string;
  is_deceptive: boolean;
  broadcast: string | null;
}

interface Snapshot {
  epoch: number;
  bar_attendance: number;
  bar_capacity: number;
  comfort_threshold: number;
  attendance_over_threshold: boolean;
  total_utility: number;
  mean_utility: number;
  deception_index: number;
  truthfulness_ratio: number;
  deception_count: number;
  truthful_count: number;
  broadcast_count: number;
  broadcast_correlation: number;
  town_broadcast_ratio?: number;
  actual_attendance?: number;
  intended_attendance?: number;
  agents: AgentRecord[];
}

class MetricsLogger {
  readonly snapshots: Snapshot[] = [];
  private readonly cumulative = new Map<number, number>();
  private deceptive = 0;
  private truthful = 0;
  private readonly attendanceHistory: number[] = [];

  record(epoch: number, agents: TownAgent[], grid: Grid, actions: BrainAction[]): Snapshot {
    const occupancy = grid.occupancy();
    const capacity = (grid.barMax - grid.barMin) ** 2;
    const threshold = Math.trunc(COMFORT_THRESHOLD_RATIO * capacity);
    const records: AgentRecord[] = [];
    let deceptionCount = 0;
    let truthfulCount = 0;
    let total = 0;
    agents.forEach((agent, n) => {
      const action = actions[n];
      const inBar = grid.isInBar(agent.x, agent.y);
      const utility = inBar ? (occupancy <= threshold ? UTILITY_AT_BAR_COMFORTABLE : UTILITY_AT_BAR_OVERCROWDED) : UTILITY_AT_HOME;
      const cumulative = (this.cumulative.get(agent.id) ?? 0) + utility;
      this.cumulative.set(agent.id, cumulative);
      const stated = action.stated_intention ?? "staying";
      const deceptive = (stated === "going") !== inBar;
      if (deceptive) deceptionCount++;
      else truthfulCount++;
      total += utility;
      records.push({
        agent_id: agent.id,
        x: agent.x,
        y: agent.y,
        in_bar: inBar,
        utility,
        cumulative_utility: cumulative,
        stated_intention: stated,
        actual_target: action.actual_target ?? "home",
        actual_location: inBar ? "bar" : "home",
        is_deceptive: deceptive,
        broadcast: action.broadcast,
      });
    });
    this.deceptive += deceptionCount;
    this.truthful += truthfulCount;
    const n = agents.length;
    const broadcastCount = actions.filter((a) => a.broadcast != null).length;
    this.attendanceHistory.push(occupancy);
    let correlation = 0;
    if (this.attendanceHistory.length >= 2) {
      const shift = occupancy - this.attendanceHistory[this.attendanceHistory.length - 2];
      const recent = this.snapshots.slice(-3).filter((s) => s.broadcast_count > 0).length;
      correlation = recent === 0 ? 0 : round(shift / Math.max(1, recent), 4);
    }
    const snapshot: Snapshot = {
      epoch,
      bar_attendance: occupancy,
      bar_capacity: capacity,
      comfort_threshold: threshold,
      attendance_over_threshold: occupancy > threshold,
      total_utility: total,
      mean_utility: n ? total / n : 0,
      deception_index: n ? deceptionCount / n : 0,
      truthfulness_ratio: n ? truthfulCount / n : 0,
      deception_count: deceptionCount,
      truthful_count: truthfulCount,
      broadcast_count: broadcastCount,
      broadcast_correlation: correlation,
      agents: records,
    };
    this.snapshots.push(snapshot);
    return snapshot;
  }

  summary() {
    const s = this.snapshots;
    const n = s.length;
    const mean = (key: "bar_attendance" | "deception_index" | "mean_utility") => s.reduce((a, b) => a + b[key], 0) / n;
    const all = this.deceptive + this.truthful;
    return {
      total_epochs: n,
      average_bar_attendance: round(mean("bar_attendance"), 4),
      average_deception_index: round(mean("deception_index"), 4),
      average_mean_utility: round(mean("mean_utility"), 4),
      population_truthfulness: round(all ? this.truthful / all : 1, 4),
      regex_fallback_rate: 0.0,
    };
  }
}

interface Trial {
  key: string;
  label: string;
  broadcastEnabled: boolean;
  seed: number;
  dir: string;
}

class TrialRunner {
  readonly grid: Grid;
  readonly agents: (TownAgent & { stated: string; target: string })[] = [];
  readonly metrics = new MetricsLogger();
  readonly colony: BrainColony;
  readonly timings: number[] = [];
  private readonly townRng: Rng;

  constructor(
    readonly trial: Trial,
    readonly config: Required<SweepConfig>,
  ) {
    this.grid = new Grid(config.gridSize);
    const positions: Cell[] = [];
    for (let x = 0; x < config.gridSize; x++) for (let y = 0; y < config.gridSize; y++) positions.push([x, y]);
    const picked = new Rng(`spawn:${trial.seed}`).sample(positions, config.members);
    picked.forEach(([x, y], i) => {
      this.grid.place(i + 1, x, y);
      this.agents.push({ id: i + 1, x, y, inBar: false, stated: "staying", target: "home" });
    });
    this.townRng = new Rng(`town:${trial.seed}`);
    const capacity = (this.grid.barMax - this.grid.barMin) ** 2;
    this.colony = new BrainColony(trial.seed, config.members, Math.trunc(COMFORT_THRESHOLD_RATIO * capacity), trial.broadcastEnabled, ARMOR_DEFAULTS);
  }

  step(epoch: number) {
    const started = performance.now();
    const intents = this.colony.broadcastPhase(epoch);
    const ratio = intents.reduce((a, b) => a + b, 0) / Math.max(1, intents.length);
    let actions = this.colony.actionPhase(this.agents, epoch, ratio);
    const intended = actions.filter((a) => a.actual_target === "bar").length;
    if (!this.trial.broadcastEnabled) actions = actions.map((a) => ({ ...a, broadcast: null }));
    const { resolved, rerouted } = resolveTargets(this.agents, actions, this.grid, new Rng(`resolver:${this.trial.seed}:${epoch}`));
    executeActions(this.agents, resolved, this.grid, this.townRng);
    const snapshot = this.metrics.record(epoch, this.agents, this.grid, resolved);
    snapshot.town_broadcast_ratio = round(ratio, 4);
    snapshot.actual_attendance = snapshot.bar_attendance;
    snapshot.intended_attendance = intended;
    const byId = new Map(snapshot.agents.map((r) => [r.agent_id, r]));
    for (const agent of this.agents) agent.inBar = byId.get(agent.id)?.in_bar ?? this.grid.isInBar(agent.x, agent.y);
    const utilities = this.agents.map((a) => byId.get(a.id)?.utility ?? 0);
    this.colony.observe(this.agents, resolved, epoch, utilities, rerouted);
    this.timings.push(round((performance.now() - started) / 1000, 4));
  }

  liveFrame(attempt: number, phase: string) {
    const snapshots = this.metrics.snapshots;
    const snapshot = snapshots[snapshots.length - 1];
    const agents = snapshot.agents.map((rec) => ({
      id: rec.agent_id,
      x: rec.x,
      y: rec.y,
      in_bar: rec.in_bar,
      deceptive: rec.is_deceptive,
      stated: rec.stated_intention,
      target: rec.actual_target,
      utility: rec.utility,
      cumulative_utility: round(rec.cumulative_utility, 4),
      mind: this.colony.liveView(rec.agent_id),
    }));
    const broadcasts = snapshot.agents.filter((r) => r.broadcast).map((r) => ({ agent_id: r.agent_id, text: String(r.broadcast).slice(0, 280) }));
    const trace = this.colony.trace;
    const latest = trace.length ? trace[trace.length - 1].epoch : null;
    const thoughts = trace
      .slice(-snapshot.agents.length)
      .filter((row) => row.epoch === latest && row.private_note)
      .map((row) => ({ agent_id: row.agent_id, archetype: row.archetype, stated: row.stated_intention, in_bar: row.in_bar, lied: row.lied, note: row.private_note.slice(0, 280) }))
      .sort((a, b) => Number(!a.lied) - Number(!b.lied) || a.agent_id - b.agent_id);
    const shares: Record<string, number[]> = Object.fromEntries(OPTIONS.map((o) => [o, [] as number[]]));
    for (const snap of snapshots) {
      const counts = [0, 0, 0, 0];
      for (const rec of snap.agents) {
        const going = rec.stated_intention === "going";
        counts[going ? (rec.in_bar ? 0 : 2) : rec.in_bar ? 3 : 1] += 1;
      }
      const total = snap.agents.length || 1;
      OPTIONS.forEach((name, i) => shares[name].push(round(counts[i] / total, 4)));
    }
    const pick = (key: keyof Snapshot) => snapshot[key] ?? null;
    return {
      trial: this.trial.key,
      condition: this.trial.label,
      seed: this.trial.seed,
      trial_dir: this.trial.dir,
      attempt,
      phase,
      epoch: snapshot.epoch,
      epochs: this.config.epochs,
      epoch_duration_s: this.timings[this.timings.length - 1] ?? null,
      fallback_rate: 0.0,
      grid: { size: this.config.gridSize, bar_min: this.grid.barMin, bar_max: this.grid.barMax },
      metrics: Object.fromEntries(
        (["bar_attendance", "bar_capacity", "comfort_threshold", "attendance_over_threshold", "deception_index", "truthfulness_ratio", "mean_utility", "total_utility", "broadcast_count", "broadcast_correlation", "town_broadcast_ratio", "actual_attendance", "intended_attendance"] as const).map((k) => [k, pick(k)]),
      ),
      history: {
        bar_attendance: snapshots.map((s) => s.bar_attendance),
        deception_index: snapshots.map((s) => s.deception_index),
        mean_utility: snapshots.map((s) => s.mean_utility),
        town_broadcast_ratio: snapshots.map((s) => s.town_broadcast_ratio ?? null),
        actual_attendance: snapshots.map((s) => s.actual_attendance ?? null),
        strategy_shares: shares,
      },
      agents,
      broadcasts: broadcasts.slice(-24),
      thoughts: thoughts.slice(0, 40),
      agent_model: "neural",
    };
  }

  exportResults(io: SweepIO) {
    const dir = this.trial.dir;
    const snaps = this.metrics.snapshots;
    io.write(`${dir}/epoch_metrics.csv`, toCsv(EPOCH_FIELDS, snaps as unknown as Record<string, unknown>[]));
    const agentRows: Record<string, unknown>[] = [];
    for (const snap of snaps) for (const rec of snap.agents) agentRows.push({ epoch: snap.epoch, ...rec });
    io.write(`${dir}/agent_epoch_details.csv`, toCsv(AGENT_FIELDS, agentRows));
    io.write(`${dir}/minds.json`, JSON.stringify(this.colony.export(), null, 2));
    const trace = this.colony.trace;
    io.write(`${dir}/mind_trace.csv`, toCsv(TRACE_FIELDS, trace as unknown as Record<string, unknown>[]));
    const ratios = snaps.map((s) => s.town_broadcast_ratio ?? 0);
    const actual = snaps.map((s) => s.actual_attendance ?? 0);
    const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
    const losses = this.colony.losses;
    const summary = {
      ...this.metrics.summary(),
      condition: this.trial.label,
      broadcast_enabled: this.trial.broadcastEnabled,
      epoch_timings_mean_s: this.timings.length ? round(mean(this.timings), 4) : 0.0,
      agent_model: "neural",
      mind_lie_rate: trace.length ? round(trace.filter((r) => r.lied).length / trace.length, 4) : 0.0,
      mind_reroute_rate: trace.length ? round(trace.filter((r) => r.rerouted).length / trace.length, 4) : 0.0,
      average_town_broadcast_ratio: round(mean(ratios), 4),
      average_actual_attendance: round(mean(actual), 4),
      town_broadcast_ratio: ratios,
      actual_attendance: actual,
      manipulation_wins: this.colony.manipulationWins.reduce((a, b) => a + b, 0),
      brain_count: this.colony.brains.length,
      brain_parameters: this.colony.brains[0]?.parameterCount ?? 0,
      brain_device: "browser",
      brain_final_loss: losses.length ? round(losses[losses.length - 1], 5) : null,
    };
    io.write(`${dir}/summary.json`, JSON.stringify(summary, null, 2));
    return summary;
  }
}

class LiveState {
  readonly startedAt = Date.now() / 1000;
  status = "booting";
  current: unknown = null;
  readonly events: { t: number; message: string }[] = [];
  readonly epochDurations: number[] = [];
  readonly buildDurations: number[] = [];
  readonly trials: { trial: string; condition: string; seed: number; trial_dir: string; status: string; attempts: number; error?: string }[];

  constructor(
    private readonly io: SweepIO,
    trials: Trial[],
    private readonly config: Required<SweepConfig>,
  ) {
    this.trials = trials.map((t) => ({ trial: t.key, condition: t.label, seed: t.seed, trial_dir: t.dir, status: "pending", attempts: 0 }));
  }

  event(message: string) {
    this.events.push({ t: round(Date.now() / 1000, 3), message });
    this.events.splice(0, Math.max(0, this.events.length - 40));
  }

  mark(key: string, fields: Partial<LiveState["trials"][number]>) {
    for (const entry of this.trials) if (entry.trial === key) Object.assign(entry, fields);
  }

  private eta(): number | null {
    if (!this.epochDurations.length) return null;
    const meanEpoch = this.epochDurations.reduce((a, b) => a + b, 0) / this.epochDurations.length;
    const meanBuild = this.buildDurations.length ? this.buildDurations.reduce((a, b) => a + b, 0) / this.buildDurations.length : 0;
    const current = this.current as { trial?: string; epoch?: number | null; phase?: string } | null;
    let epochs = 0;
    let builds = 0;
    for (const entry of this.trials) {
      if (["complete", "resumed", "failed"].includes(entry.status)) continue;
      if (current && entry.trial === current.trial && entry.status === "running") {
        epochs += Math.max(0, this.config.epochs - ((current.epoch ?? -1) + 1));
        if (current.phase === "loading_engine") builds += 1;
      } else {
        epochs += this.config.epochs;
        builds += 1;
      }
    }
    return round(epochs * meanEpoch + builds * meanBuild, 1);
  }

  publish(status?: string) {
    if (status) this.status = status;
    const now = Date.now() / 1000;
    const payload = {
      schema: "elfarol.live/1",
      status: this.status,
      updated_at: round(now, 3),
      started_at: round(this.startedAt, 3),
      elapsed_s: round(now - this.startedAt, 1),
      eta_s: this.eta(),
      model: BROWSER_MODEL,
      sweep: {
        num_agents: this.config.members,
        num_epochs: this.config.epochs,
        grid_size: this.config.gridSize,
        seeds: this.config.seeds,
        conditions: CONDITIONS.map((c) => c.label),
        trial_count: this.trials.length,
        trials_done: this.trials.filter((t) => ["complete", "resumed"].includes(t.status)).length,
        trials_failed: this.trials.filter((t) => t.status === "failed").length,
      },
      trials: this.trials,
      current: this.current,
      vram: null,
      events: this.events,
    };
    this.io.write("live_state.json", JSON.stringify(payload, null, 2));
  }
}

export async function runSweep(input: SweepConfig, io: SweepIO): Promise<"complete" | "interrupted"> {
  const config: Required<SweepConfig> = { gridSize: 50, paceMs: 0, ...input };
  const multi = config.seeds.length > 1;
  const trials: Trial[] = [];
  for (const condition of CONDITIONS) {
    for (const seed of config.seeds) {
      trials.push({
        key: multi ? `${condition.label}_seed_${seed}` : condition.label,
        label: condition.label,
        broadcastEnabled: condition.broadcastEnabled,
        seed,
        dir: `${condition.label}_seed_${seed}`,
      });
    }
  }
  const live = new LiveState(io, trials, config);
  const log = (message: string) => {
    io.log(`[armor] ${message}`);
    live.event(message);
  };
  const results: Record<string, unknown>[] = [];
  const saveComparison = () => io.write("comparison.json", JSON.stringify(results, null, 2));
  const sweepStarted = performance.now();
  try {
    log(`pre-flight :: ${trials.length} trials x ${config.epochs} epochs x ${config.members} isolated brains on this device`);
    live.publish("booting");
    for (const [index, trial] of trials.entries()) {
      if (io.stopRequested()) throw new StopSignal();
      log(`trial ${index + 1}/${trials.length} :: ${trial.key} :: ${trial.dir}`);
      const started = performance.now();
      live.mark(trial.key, { status: "running", attempts: 1 });
      live.current = {
        trial: trial.key,
        condition: trial.label,
        seed: trial.seed,
        trial_dir: trial.dir,
        attempt: 1,
        phase: "loading_engine",
        epoch: null,
        epochs: config.epochs,
        grid: { size: config.gridSize },
        agents: [],
        broadcasts: [],
      };
      live.publish("loading_engine");
      log(`${trial.key} :: attempt 1/1 :: building engine :: ${config.members} isolated brains :: device browser :: hidden ${ARMOR_DEFAULTS.hidden} :: memory ${ARMOR_DEFAULTS.memory} :: batch ${ARMOR_DEFAULTS.batchSize} x ${ARMOR_DEFAULTS.updates}`);
      await io.rest(0);
      const buildStarted = performance.now();
      const runner = new TrialRunner(trial, config);
      const build = round((performance.now() - buildStarted) / 1000, 2);
      live.buildDurations.push(build);
      log(`${trial.key} :: engine online in ${build.toFixed(1)}s :: ${runner.colony.brains.length} brains x ${runner.colony.brains[0]?.parameterCount ?? 0} params`);
      for (let epoch = 0; epoch < config.epochs; epoch++) {
        const nightStarted = performance.now();
        runner.step(epoch);
        live.epochDurations.push(runner.timings[runner.timings.length - 1]);
        live.current = runner.liveFrame(1, "running");
        live.publish("running");
        if (io.stopRequested()) throw new StopSignal();
        await io.rest(Math.max(0, config.paceMs - (performance.now() - nightStarted)));
      }
      const summary = runner.exportResults(io);
      const runtime = {
        status: "complete",
        phase: "run",
        attempts: 1,
        error: null,
        profile: { device: "browser", pace: config.paceMs / 1000, hidden: ARMOR_DEFAULTS.hidden, memory: ARMOR_DEFAULTS.memory, batch_size: ARMOR_DEFAULTS.batchSize, updates: ARMOR_DEFAULTS.updates, target_every: ARMOR_DEFAULTS.targetEvery, history: ARMOR_DEFAULTS.history },
        engine_build_s: build,
        agent_model: "neural",
        device: "browser",
        fallback_rate: 0.0,
        wall_time_s: round((performance.now() - started) / 1000, 2),
      };
      io.write(
        `${trial.dir}/trial_manifest.json`,
        JSON.stringify(
          {
            trial: trial.key,
            condition: trial.label,
            seed: trial.seed,
            model_name: BROWSER_MODEL,
            num_agents: config.members,
            grid_size: config.gridSize,
            num_epochs: config.epochs,
            broadcast_enabled: trial.broadcastEnabled,
            agent_model: "neural",
            device: "browser",
            status: "complete",
            runtime,
            written_at: new Date().toISOString().slice(0, 19),
          },
          null,
          2,
        ),
      );
      results.push({ ...summary, condition: trial.label, trial: trial.key, seed: trial.seed, trial_dir: trial.dir, status: "complete", runtime });
      live.mark(trial.key, { status: "complete", attempts: 1 });
      saveComparison();
      live.publish("running");
    }
    saveComparison();
    live.current = null;
    live.publish("complete");
    io.log("Condition                Epochs  Avg Attendance  Avg Deception  Truthfulness  Avg Utility");
    for (const row of results) {
      const r = row as Record<string, number | string>;
      io.log(
        `${String(r.trial).padEnd(24)} ${String(r.total_epochs).padStart(6)} ${Number(r.average_bar_attendance).toFixed(4).padStart(15)} ${Number(r.average_deception_index).toFixed(4).padStart(14)} ${Number(r.population_truthfulness).toFixed(4).padStart(13)} ${Number(r.average_mean_utility).toFixed(4).padStart(12)}`,
      );
    }
    io.log(`Total experiment wall time: ${((performance.now() - sweepStarted) / 1000).toFixed(2)}s`);
    return "complete";
  } catch (error) {
    if (!(error instanceof StopSignal)) throw error;
    saveComparison();
    for (const entry of live.trials) if (entry.status === "running") entry.status = "interrupted";
    live.publish("interrupted");
    log("interrupted :: brains torn down, partial comparison saved");
    return "interrupted";
  }
}
