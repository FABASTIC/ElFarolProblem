import type {
  AnalyticsReport,
  ComparisonRow,
  LiveFrame,
  LiveState,
  Payoffs,
  StrategyKey,
} from "./types";

export const STRATEGIES: StrategyKey[] = ["honest_go", "honest_stay", "false_go", "false_stay"];

export const STRATEGY_LABEL: Record<StrategyKey, string> = {
  honest_go: "HONEST · AT BAR",
  honest_stay: "HONEST · HOME",
  false_go: "BLUFF · SAID GO, STAYED",
  false_stay: "COVERT · SAID STAY, WENT",
};

export const STRATEGY_SHORT: Record<StrategyKey, string> = {
  honest_go: "H·BAR",
  honest_stay: "H·HOME",
  false_go: "BLUFF",
  false_stay: "COVERT",
};

export const STRATEGY_COLOR: Record<StrategyKey, string> = {
  honest_go: "#3987e5",
  honest_stay: "#9085e9",
  false_go: "#c98500",
  false_stay: "#d55181",
};

export const STRATEGY_STORY: Record<StrategyKey, string> = {
  honest_go: "Said going, and went",
  honest_stay: "Said staying, and stayed home",
  false_go: "Bluffed: said going, stayed home",
  false_stay: "Covert: said staying, went anyway",
};

export const CONDITION_COLOR: Record<string, string> = {
  control: "#00e5ff",
  delta2: "#ff0055",
};

export const CONDITION_LABEL: Record<string, string> = {
  control: "CONTROL",
  delta2: "DELTA 2",
};

export const CONDITION_DETAIL: Record<string, string> = {
  control: "NO BROADCAST",
  delta2: "BROADCAST + DECEPTION",
};

export const DEFAULT_PAYOFFS: Payoffs = { bar_comfortable: 1, bar_overcrowded: -1, home: 0.3 };

export const RUNNING_STATES = new Set(["booting", "loading_engine", "running"]);

export function conditionColor(condition: string | null | undefined): string {
  return (condition && CONDITION_COLOR[condition]) || "#ebebeb";
}

export function conditionLabel(condition: string | null | undefined): string {
  return (condition && CONDITION_LABEL[condition]) || (condition ?? "—").toUpperCase();
}

export interface MindView {
  archetype: string;
  traits: Record<string, number> | null;
  forecast: number | null;
  forecastConfidence: number | null;
  predictor: string | null;
  instinct: Partial<Record<StrategyKey, number>>;
  mood: string;
  moodIntensity: number;
  reputation: number | null;
  lies?: number | null;
  note: string | null;
}

export interface Thought {
  agentId: number;
  archetype: string;
  stated: string;
  inBar: boolean;
  lied: boolean;
  note: string;
}

export interface AgentState {
  id: number;
  x: number;
  y: number;
  inBar: boolean;
  strategy: StrategyKey;
  utility: number;
  cumulative: number;
  stated: string;
  target: string;
  broadcast: string | null;
  mind?: MindView | null;
}

export interface Frame {
  epoch: number;
  agents: AgentState[];
  broadcasts: { agentId: number; text: string }[];
  thoughts?: Thought[];
  preview?: boolean;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function previewFrame(members: number, seed: number, gridSize: number, barMin: number, barMax: number): Frame {
  const random = mulberry32(seed * 2654435761 + members);
  const cells: [number, number][] = [];
  for (let x = 0; x < gridSize; x += 1) {
    for (let y = 0; y < gridSize; y += 1) {
      if (x >= barMin && x < barMax && y >= barMin && y < barMax) continue;
      cells.push([x, y]);
    }
  }
  for (let i = cells.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  const count = Math.min(members, cells.length);
  const agents: AgentState[] = Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    x: cells[i][0],
    y: cells[i][1],
    inBar: false,
    strategy: "honest_stay",
    utility: 0,
    cumulative: 0,
    stated: "",
    target: "",
    broadcast: null,
    mind: null,
  }));
  return { epoch: -1, agents, broadcasts: [], thoughts: [], preview: true };
}

export function appendLiveFrame(frames: Frame[], frame: Frame): Frame[] {
  if (!frames.length) return [frame];
  const last = frames[frames.length - 1];
  if (frame.epoch > last.epoch) return [...frames, frame];
  if (frame.epoch === last.epoch) return [...frames.slice(0, -1), frame];
  return [frame];
}

export function attachMinds(
  frames: Frame[],
  traceRows: Record<string, string>[],
  minds: Record<string, { archetype?: string; traits?: Record<string, number> }> | null,
): Frame[] {
  if (!traceRows.length) return frames;
  const byKey = new Map<string, Record<string, string>>();
  for (const row of traceRows) byKey.set(`${row.epoch}:${row.agent_id}`, row);
  const num = (value: string | undefined) => (value != null && value !== "" && Number.isFinite(Number(value)) ? Number(value) : null);
  return frames.map((frame) => {
    const thoughts: Thought[] = [];
    const agents = frame.agents.map((agent) => {
      const row = byKey.get(`${frame.epoch}:${agent.id}`);
      if (!row) return agent;
      const lied = truthy(row.lied);
      const note = row.private_note || null;
      if (note) {
        thoughts.push({ agentId: agent.id, archetype: row.archetype, stated: row.stated_intention, inBar: agent.inBar, lied, note });
      }
      const mind: MindView = {
        archetype: row.archetype,
        traits: minds?.[String(agent.id)]?.traits ?? null,
        forecast: num(row.forecast),
        forecastConfidence: num(row.forecast_confidence),
        predictor: row.active_predictor ? row.active_predictor.replace(/_/g, " ") : null,
        instinct: {
          honest_go: num(row.p_honest_go) ?? 0,
          honest_stay: num(row.p_honest_stay) ?? 0,
          false_go: num(row.p_false_go) ?? 0,
          false_stay: num(row.p_false_stay) ?? 0,
        },
        mood: row.top_emotion || "calm",
        moodIntensity: num(row.top_emotion_intensity) ?? 0,
        reputation: num(row.reputation),
        note,
      };
      return { ...agent, mind };
    });
    thoughts.sort((a, b) => Number(b.lied) - Number(a.lied) || a.agentId - b.agentId);
    return { ...frame, agents, thoughts };
  });
}

export function strategyOf(stated: string | null | undefined, inBar: boolean): StrategyKey {
  const going = (stated ?? "").trim().toLowerCase() === "going";
  if (going) return inBar ? "honest_go" : "false_go";
  return inBar ? "false_stay" : "honest_stay";
}

function truthy(value: string | undefined): boolean {
  return ["true", "1", "1.0", "yes"].includes((value ?? "").trim().toLowerCase());
}

export function framesFromRows(rows: Record<string, string>[]): Frame[] {
  const byEpoch = new Map<number, AgentState[]>();
  for (const row of rows) {
    const epoch = Number(row.epoch);
    const id = Number(row.agent_id);
    if (!Number.isFinite(epoch) || !Number.isFinite(id)) continue;
    const inBar = "in_bar" in row ? truthy(row.in_bar) : (row.actual_location ?? "").toLowerCase() === "bar";
    const agent: AgentState = {
      id,
      x: Number(row.x),
      y: Number(row.y),
      inBar,
      strategy: strategyOf(row.stated_intention, inBar),
      utility: Number(row.utility),
      cumulative: Number(row.cumulative_utility),
      stated: row.stated_intention ?? "",
      target: row.actual_target ?? "",
      broadcast: row.broadcast ? row.broadcast : null,
    };
    const list = byEpoch.get(epoch);
    if (list) list.push(agent);
    else byEpoch.set(epoch, [agent]);
  }
  return [...byEpoch.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([epoch, agents]) => {
      agents.sort((a, b) => a.id - b.id);
      return {
        epoch,
        agents,
        broadcasts: agents.filter((a) => a.broadcast).map((a) => ({ agentId: a.id, text: a.broadcast as string })),
      };
    });
}

export function frameFromLive(frame: LiveFrame | null | undefined): Frame | null {
  if (!frame || frame.epoch == null || !frame.agents?.length) return null;
  const agents = frame.agents
    .map<AgentState>((a) => ({
      id: a.id,
      x: a.x,
      y: a.y,
      inBar: a.in_bar,
      strategy: strategyOf(a.stated, a.in_bar),
      utility: a.utility ?? 0,
      cumulative: a.cumulative_utility ?? 0,
      stated: a.stated ?? "",
      target: a.target ?? "",
      broadcast: null,
      mind: a.mind
        ? {
            archetype: a.mind.archetype,
            traits: a.mind.traits ?? null,
            forecast: a.mind.forecast,
            forecastConfidence: a.mind.forecast_confidence,
            predictor: a.mind.predictor,
            instinct: a.mind.instinct ?? {},
            mood: a.mind.mood,
            moodIntensity: a.mind.mood_intensity,
            reputation: a.mind.reputation,
            lies: a.mind.lies ?? null,
            note: a.mind.note,
          }
        : null,
    }))
    .sort((a, b) => a.id - b.id);
  const spoken = new Map(frame.broadcasts.map((b) => [b.agent_id, b.text]));
  agents.forEach((a) => {
    a.broadcast = spoken.get(a.id) ?? null;
  });
  return {
    epoch: frame.epoch,
    agents,
    broadcasts: frame.broadcasts.map((b) => ({ agentId: b.agent_id, text: b.text })),
    thoughts: (frame.thoughts ?? []).map((t) => ({
      agentId: t.agent_id,
      archetype: t.archetype,
      stated: t.stated,
      inBar: t.in_bar,
      lied: t.lied,
      note: t.note,
    })),
  };
}

export interface GameView {
  n: number;
  threshold: number;
  capacity: number;
  payoffs: Payoffs;
  gridSize: number;
  barMin: number;
  barMax: number;
  pureAttendance: number[];
  primaryAttendance: number;
  mixedProbability: number;
  mixedAttendance: number;
  dominant: string | null;
  congestion: boolean;
  source: "analyzer" | "derived";
}

function payoffGo(attendance: number, threshold: number, payoffs: Payoffs): number {
  return attendance <= threshold ? payoffs.bar_comfortable : payoffs.bar_overcrowded;
}

function binomialCdf(k: number, n: number, p: number): number {
  if (k < 0) return 0;
  if (k >= n) return 1;
  if (p <= 0) return 1;
  if (p >= 1) return 0;
  let logCoeff = 0;
  let peak = -Infinity;
  const terms: number[] = [];
  for (let j = 0; j <= k; j += 1) {
    if (j > 0) logCoeff += Math.log(n - j + 1) - Math.log(j);
    const term = logCoeff + j * Math.log(p) + (n - j) * Math.log1p(-p);
    terms.push(term);
    peak = Math.max(peak, term);
  }
  return Math.min(1, Math.exp(peak) * terms.reduce((acc, t) => acc + Math.exp(t - peak), 0));
}

export function solveNash(n: number, threshold: number, payoffs: Payoffs) {
  const pure: number[] = [];
  for (let a = 0; a <= n; a += 1) {
    const goersHold = a === 0 || payoffGo(a, threshold, payoffs) >= payoffs.home;
    const stayersHold = a === n || payoffGo(a + 1, threshold, payoffs) <= payoffs.home;
    if (goersHold && stayersHold) pure.push(a);
  }
  const expectedGo = (p: number) => {
    const f = binomialCdf(threshold - 1, n - 1, p);
    return payoffs.bar_comfortable * f + payoffs.bar_overcrowded * (1 - f);
  };
  let probability: number;
  if (expectedGo(1) >= payoffs.home) probability = 1;
  else if (expectedGo(0) <= payoffs.home) probability = 0;
  else {
    let lo = 0;
    let hi = 1;
    for (let i = 0; i < 100; i += 1) {
      const mid = (lo + hi) / 2;
      if (expectedGo(mid) > payoffs.home) lo = mid;
      else hi = mid;
    }
    probability = (lo + hi) / 2;
  }
  const primary = pure.length
    ? pure.reduce((best, a) => (Math.abs(a - n * probability) < Math.abs(best - n * probability) ? a : best), pure[0])
    : Math.round(n * probability);
  let dominant: string | null = null;
  if (payoffGo(n, threshold, payoffs) > payoffs.home) dominant = "go";
  else if (payoffGo(1, threshold, payoffs) < payoffs.home) dominant = "stay";
  return { pure, primary, probability, mixedAttendance: n * probability, dominant, congestion: n > threshold };
}

export function gameView(report: AnalyticsReport | null, live: LiveState | null, gridHint?: number | null): GameView {
  const game = report?.game ?? null;
  const metrics = live?.current?.metrics ?? {};
  const n = game?.num_agents ?? live?.sweep.num_agents ?? 50;
  const threshold = game?.comfort_threshold ?? (typeof metrics.comfort_threshold === "number" ? metrics.comfort_threshold : 60);
  const capacity = game?.bar_capacity ?? (typeof metrics.bar_capacity === "number" ? metrics.bar_capacity : 100);
  const payoffs = game?.payoffs ?? DEFAULT_PAYOFFS;
  const gridSize = live?.sweep.grid_size ?? gridHint ?? report?.trials.find((t) => t.grid_size)?.grid_size ?? 50;
  const side = Math.max(1, Math.round(Math.sqrt(capacity)));
  const barMin = live?.current?.grid?.bar_min ?? Math.floor((gridSize - side) / 2);
  const barMax = live?.current?.grid?.bar_max ?? barMin + side;
  if (game?.nash) {
    return {
      n,
      threshold,
      capacity,
      payoffs,
      gridSize,
      barMin,
      barMax,
      pureAttendance: game.nash.pure_attendance,
      primaryAttendance: game.nash.primary_attendance,
      mixedProbability: game.nash.mixed_go_probability,
      mixedAttendance: game.nash.mixed_expected_attendance,
      dominant: game.nash.dominant_strategy,
      congestion: game.nash.congestion_possible,
      source: "analyzer",
    };
  }
  const solved = solveNash(n, threshold, payoffs);
  return {
    n,
    threshold,
    capacity,
    payoffs,
    gridSize,
    barMin,
    barMax,
    pureAttendance: solved.pure,
    primaryAttendance: solved.primary,
    mixedProbability: solved.probability,
    mixedAttendance: solved.mixedAttendance,
    dominant: solved.dominant,
    congestion: solved.congestion,
    source: "derived",
  };
}

export interface TrialTrace {
  epochs: number[];
  attendance: number[];
  deception: number[];
  utility: number[];
  regret: number[];
  klStep: (number | null)[];
  shares: Record<StrategyKey, number[]>;
}

function smooth(counts: number[], alpha = 0.5): number[] {
  const total = counts.reduce((acc, c) => acc + c, 0) + alpha * counts.length;
  return counts.map((c) => (c + alpha) / total);
}

function kl(p: number[], q: number[]): number {
  return p.reduce((acc, pi, i) => acc + pi * (Math.log(pi) - Math.log(q[i])), 0);
}

function regretAt(attendance: number, n: number, game: GameView): number {
  if (n <= 0) return 0;
  const barGain = Math.max(0, game.payoffs.home - payoffGo(attendance, game.threshold, game.payoffs));
  const homeGain = Math.max(0, payoffGo(attendance + 1, game.threshold, game.payoffs) - game.payoffs.home);
  return (attendance * barGain + (n - attendance) * homeGain) / n;
}

function klSeries(countsByEpoch: number[][]): (number | null)[] {
  return countsByEpoch.map((counts, i) => (i === 0 ? null : kl(smooth(counts), smooth(countsByEpoch[i - 1]))));
}

export function traceFromFrames(frames: Frame[], game: GameView): TrialTrace {
  const shares = Object.fromEntries(STRATEGIES.map((s) => [s, [] as number[]])) as Record<StrategyKey, number[]>;
  const counts: number[][] = [];
  const attendance: number[] = [];
  const deception: number[] = [];
  const utility: number[] = [];
  const regret: number[] = [];
  for (const frame of frames) {
    const n = frame.agents.length || 1;
    const tally = STRATEGIES.map((s) => frame.agents.filter((a) => a.strategy === s).length);
    counts.push(tally);
    STRATEGIES.forEach((s, i) => shares[s].push(tally[i] / n));
    const present = frame.agents.filter((a) => a.inBar).length;
    attendance.push(present);
    deception.push((tally[2] + tally[3]) / n);
    utility.push(frame.agents.reduce((acc, a) => acc + (Number.isFinite(a.utility) ? a.utility : 0), 0) / n);
    regret.push(regretAt(present, frame.agents.length, game));
  }
  return { epochs: frames.map((f) => f.epoch), attendance, deception, utility, regret, klStep: klSeries(counts), shares };
}

export function traceFromLive(live: LiveFrame | null | undefined, game: GameView): TrialTrace | null {
  const history = live?.history;
  if (!live || !history) return null;
  const length = history.bar_attendance.length;
  const n = live.agents.length || game.n;
  const epochs = Array.from({ length }, (_, i) => i);
  const attendance = history.bar_attendance.map((v) => v ?? 0);
  const shares = Object.fromEntries(
    STRATEGIES.map((s) => [s, (history.strategy_shares?.[s] ?? new Array(length).fill(0)).slice(0, length)]),
  ) as Record<StrategyKey, number[]>;
  const counts = epochs.map((i) => STRATEGIES.map((s) => Math.round((shares[s][i] ?? 0) * n)));
  return {
    epochs,
    attendance,
    deception: history.deception_index.map((v) => v ?? 0),
    utility: history.mean_utility.map((v) => v ?? 0),
    regret: attendance.map((a) => regretAt(a, n, game)),
    klStep: history.strategy_shares ? klSeries(counts) : epochs.map(() => null),
    shares,
  };
}

export interface TrialEntry {
  key: string;
  label: string;
  condition: string;
  seed: string;
  status: string;
  attempts: number | null;
  fallback: number | null;
  wallTime: number | null;
  buildTime: number | null;
  peakVram: number | null;
  encoding: string | null;
  error: string | null;
  epochs: number | null;
  isLive: boolean;
  revision: string;
}

function seedSortKey(seed: string): [number, number, string] {
  return /^-?\d+$/.test(seed) ? [0, Number(seed), seed] : [1, 0, seed];
}

export function mergeTrials(
  comparison: ComparisonRow[] | null,
  live: LiveState | null,
  report: AnalyticsReport | null,
  liveKey: string | null,
): TrialEntry[] {
  const entries = new Map<string, TrialEntry>();
  const blank = (key: string, condition: string, seed: string): TrialEntry => ({
    key,
    label: key,
    condition,
    seed,
    status: "pending",
    attempts: null,
    fallback: null,
    wallTime: null,
    buildTime: null,
    peakVram: null,
    encoding: null,
    error: null,
    epochs: null,
    isLive: false,
    revision: "",
  });
  for (const trial of report?.trials ?? []) {
    const entry = blank(trial.trial, trial.condition, String(trial.seed));
    entry.status = trial.status ?? "complete";
    entry.epochs = trial.epochs;
    entry.fallback = trial.fallback_rate ?? null;
    entries.set(trial.trial, entry);
  }
  for (const row of comparison ?? []) {
    if (!row.trial_dir) continue;
    const entry = entries.get(row.trial_dir) ?? blank(row.trial_dir, row.condition, String(row.seed ?? ""));
    const runtime = row.runtime ?? {};
    entry.condition = row.condition ?? entry.condition;
    entry.seed = row.seed != null ? String(row.seed) : entry.seed;
    entry.label = row.trial ?? entry.label;
    entry.status = row.status ?? runtime.status ?? entry.status;
    entry.attempts = runtime.attempts ?? entry.attempts;
    entry.fallback = row.regex_fallback_rate ?? runtime.fallback_rate ?? entry.fallback;
    entry.wallTime = runtime.wall_time_s ?? null;
    entry.buildTime = runtime.engine_build_s ?? null;
    entry.peakVram = runtime.vram_peak_mib ?? null;
    entry.encoding = runtime.prompt_encoding ?? null;
    entry.error = runtime.error ?? null;
    entry.epochs = row.total_epochs ?? entry.epochs;
    entry.revision = `${entry.status}:${runtime.wall_time_s ?? ""}:${row.total_epochs ?? ""}`;
    entries.set(row.trial_dir, entry);
  }
  for (const trial of live?.trials ?? []) {
    const entry = entries.get(trial.trial_dir) ?? blank(trial.trial_dir, trial.condition, String(trial.seed));
    entry.label = trial.trial || entry.label;
    if (trial.status !== "complete" || entry.status === "pending") entry.status = trial.status;
    entry.attempts = trial.attempts || entry.attempts;
    entry.error = trial.error ?? entry.error;
    entries.set(trial.trial_dir, entry);
  }
  const order: Record<string, number> = { control: 0, delta2: 1 };
  return [...entries.values()]
    .map((entry) => ({ ...entry, isLive: entry.key === liveKey }))
    .sort((a, b) => {
      const byCondition = (order[a.condition] ?? 2) - (order[b.condition] ?? 2) || a.condition.localeCompare(b.condition);
      if (byCondition !== 0) return byCondition;
      const ka = seedSortKey(a.seed);
      const kb = seedSortKey(b.seed);
      return ka[0] - kb[0] || ka[1] - kb[1] || ka[2].localeCompare(kb[2]);
    });
}

export function cumulativeRange(frames: Frame[]): [number, number] {
  let lo = 0;
  let hi = 1;
  for (const frame of frames) {
    for (const agent of frame.agents) {
      if (Number.isFinite(agent.cumulative)) {
        lo = Math.min(lo, agent.cumulative);
        hi = Math.max(hi, agent.cumulative);
      }
    }
  }
  return [lo, hi];
}
