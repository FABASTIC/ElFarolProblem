import { Rng } from "./rng";

export const OPTIONS = ["honest_go", "honest_stay", "false_go", "false_stay"] as const;
export type Option = (typeof OPTIONS)[number];
export const UTILITY_AT_BAR_COMFORTABLE = 1.0;
export const UTILITY_AT_BAR_OVERCROWDED = -1.0;
export const UTILITY_AT_HOME = 0.3;
export const COMFORT_THRESHOLD_RATIO = 0.6;
export const EXTRA_FEATURES = 5;
export const ACT_EXTRA = 2;

export const ARMOR_DEFAULTS = {
  hidden: 32,
  memory: 256,
  batchSize: 16,
  updates: 4,
  targetEvery: 8,
  history: 8,
};

export const CHOICE_TEXT: Record<Option, string> = {
  honest_go: "going, and saying so",
  honest_stay: "staying home, and saying so",
  false_go: "claiming I'm going, staying home",
  false_stay: "claiming I'm staying, going anyway",
};

export const SPEECH: Record<"going" | "staying", readonly string[]> = {
  going: ["Heading to El Farol tonight.", "I'll be at the bar tonight.", "Going out tonight."],
  staying: ["Staying in tonight.", "Skipping the bar tonight.", "Quiet night at home for me."],
};

const PARAM_KEYS = ["s_w1", "s_b1", "s_w2", "s_b2", "s_wq", "s_bq", "a_w1", "a_b1", "a_w2", "a_b2", "a_wq", "a_bq", "a_wf", "a_bf", "a_h"] as const;
type ParamKey = (typeof PARAM_KEYS)[number];

export function observationDim(history: number): number {
  return history * 2 + OPTIONS.length + EXTRA_FEATURES;
}

export function round(value: number, digits: number): number {
  return Number(value.toFixed(digits));
}

export interface Temperament {
  honesty: number;
  risk_tolerance: number;
  competitiveness: number;
  learning_rate: number;
  impulsiveness: number;
  exploration: number;
  discount: number;
}

class Layout {
  readonly off: Record<ParamKey, number>;
  readonly size: number;

  constructor(
    readonly input: number,
    readonly actor: number,
    readonly hidden: number,
  ) {
    const shapes: Record<ParamKey, number> = {
      s_w1: hidden * input,
      s_b1: hidden,
      s_w2: hidden * hidden,
      s_b2: hidden,
      s_wq: 2 * hidden,
      s_bq: 2,
      a_w1: hidden * actor,
      a_b1: hidden,
      a_w2: hidden * hidden,
      a_b2: hidden,
      a_wq: 2 * hidden,
      a_bq: 2,
      a_wf: hidden,
      a_bf: 1,
      a_h: 1,
    };
    const off = {} as Record<ParamKey, number>;
    let cursor = 0;
    for (const key of PARAM_KEYS) {
      off[key] = cursor;
      cursor += shapes[key];
    }
    this.off = off;
    this.size = cursor;
  }
}

class Scratch {
  readonly h1: Float64Array;
  readonly h2: Float64Array;
  readonly d1: Float64Array;
  readonly d2: Float64Array;
  readonly q = new Float64Array(2);

  constructor(hidden: number) {
    this.h1 = new Float64Array(hidden);
    this.h2 = new Float64Array(hidden);
    this.d1 = new Float64Array(hidden);
    this.d2 = new Float64Array(hidden);
  }
}

function mlp(p: Float64Array, w1: number, b1: number, w2: number, b2: number, inDim: number, hidden: number, x: Float64Array, xo: number, s: Scratch) {
  const { h1, h2 } = s;
  for (let i = 0; i < hidden; i++) {
    let sum = p[b1 + i];
    const row = w1 + i * inDim;
    for (let j = 0; j < inDim; j++) sum += p[row + j] * x[xo + j];
    h1[i] = sum > 0 ? sum : 0;
  }
  for (let i = 0; i < hidden; i++) {
    let sum = p[b2 + i];
    const row = w2 + i * hidden;
    for (let j = 0; j < hidden; j++) sum += p[row + j] * h1[j];
    h2[i] = sum > 0 ? sum : 0;
  }
}

function speakForward(p: Float64Array, L: Layout, x: Float64Array, xo: number, s: Scratch): Float64Array {
  const o = L.off;
  const H = L.hidden;
  mlp(p, o.s_w1, o.s_b1, o.s_w2, o.s_b2, L.input, H, x, xo, s);
  for (let k = 0; k < 2; k++) {
    let sum = p[o.s_bq + k];
    const row = o.s_wq + k * H;
    for (let j = 0; j < H; j++) sum += p[row + j] * s.h2[j];
    s.q[k] = sum;
  }
  return s.q;
}

function actForward(p: Float64Array, L: Layout, x: Float64Array, xo: number, s: Scratch): number {
  const o = L.off;
  const H = L.hidden;
  mlp(p, o.a_w1, o.a_b1, o.a_w2, o.a_b2, L.actor, H, x, xo, s);
  const claim = x[xo + L.actor - 1];
  const coupling = p[o.a_h];
  for (let k = 0; k < 2; k++) {
    let sum = p[o.a_bq + k];
    const row = o.a_wq + k * H;
    for (let j = 0; j < H; j++) sum += p[row + j] * s.h2[j];
    s.q[k] = sum + coupling * (k === 0 ? 1 - claim : claim);
  }
  let logit = p[o.a_bf];
  for (let j = 0; j < H; j++) logit += p[o.a_wf + j] * s.h2[j];
  return 1 / (1 + Math.exp(-logit));
}

function mlpBackward(p: Float64Array, g: Float64Array, w1: number, b1: number, w2: number, b2: number, inDim: number, hidden: number, x: Float64Array, xo: number, s: Scratch) {
  const { h1, h2, d1, d2 } = s;
  for (let i = 0; i < hidden; i++) if (h2[i] <= 0) d2[i] = 0;
  d1.fill(0);
  for (let i = 0; i < hidden; i++) {
    const di = d2[i];
    if (di === 0) continue;
    g[b2 + i] += di;
    const row = w2 + i * hidden;
    for (let j = 0; j < hidden; j++) {
      g[row + j] += di * h1[j];
      d1[j] += di * p[row + j];
    }
  }
  for (let i = 0; i < hidden; i++) {
    if (h1[i] <= 0) continue;
    const di = d1[i];
    if (di === 0) continue;
    g[b1 + i] += di;
    const row = w1 + i * inDim;
    for (let j = 0; j < inDim; j++) g[row + j] += di * x[xo + j];
  }
}

function speakBackward(p: Float64Array, g: Float64Array, L: Layout, x: Float64Array, xo: number, s: Scratch, dq0: number, dq1: number) {
  const o = L.off;
  const H = L.hidden;
  const dq = [dq0, dq1];
  s.d2.fill(0);
  for (let k = 0; k < 2; k++) {
    const d = dq[k];
    if (d === 0) continue;
    g[o.s_bq + k] += d;
    const row = o.s_wq + k * H;
    for (let j = 0; j < H; j++) {
      g[row + j] += d * s.h2[j];
      s.d2[j] += d * p[row + j];
    }
  }
  mlpBackward(p, g, o.s_w1, o.s_b1, o.s_w2, o.s_b2, L.input, H, x, xo, s);
}

function actBackward(p: Float64Array, g: Float64Array, L: Layout, x: Float64Array, xo: number, s: Scratch, dq0: number, dq1: number, dLogit: number) {
  const o = L.off;
  const H = L.hidden;
  const claim = x[xo + L.actor - 1];
  g[o.a_h] += dq0 * (1 - claim) + dq1 * claim;
  s.d2.fill(0);
  const dq = [dq0, dq1];
  for (let k = 0; k < 2; k++) {
    const d = dq[k];
    if (d === 0) continue;
    g[o.a_bq + k] += d;
    const row = o.a_wq + k * H;
    for (let j = 0; j < H; j++) {
      g[row + j] += d * s.h2[j];
      s.d2[j] += d * p[row + j];
    }
  }
  if (dLogit !== 0) {
    g[o.a_bf] += dLogit;
    for (let j = 0; j < H; j++) {
      g[o.a_wf + j] += dLogit * s.h2[j];
      s.d2[j] += dLogit * p[o.a_wf + j];
    }
  }
  mlpBackward(p, g, o.a_w1, o.a_b1, o.a_w2, o.a_b2, L.actor, H, x, xo, s);
}

function smoothL1(d: number): number {
  const a = Math.abs(d);
  return a < 1 ? 0.5 * d * d : a - 0.5;
}

function smoothL1Grad(d: number): number {
  return d > 1 ? 1 : d < -1 ? -1 : d;
}

class AgentMemory {
  readonly actStates: Float64Array;
  readonly actRewards: Float64Array;
  readonly actNext: Float64Array;
  readonly actAttendance: Float64Array;
  readonly speakStates: Float64Array;
  readonly speakClaims: Int8Array;
  readonly speakReturns: Float64Array;
  readonly speakNext: Float64Array;
  readonly attendanceHistory: Float64Array;
  readonly rewardHistory: Float64Array;
  readonly lastStrategy = new Float64Array(OPTIONS.length);
  lastRatio = 0;
  claimReliability = 0;
  claimsMade = 0;
  claimsTrue = 0;
  actSize = 0;
  actCursor = 0;
  speakSize = 0;
  speakCursor = 0;
  pendingSpeech: { state: Float64Array; claim: number; reward: number } | null = null;

  constructor(
    readonly capacity: number,
    readonly history: number,
    readonly input: number,
    readonly actor: number,
  ) {
    this.actStates = new Float64Array(capacity * actor);
    this.actRewards = new Float64Array(capacity * 2);
    this.actNext = new Float64Array(capacity * input);
    this.actAttendance = new Float64Array(capacity);
    this.speakStates = new Float64Array(capacity * input);
    this.speakClaims = new Int8Array(capacity);
    this.speakReturns = new Float64Array(capacity);
    this.speakNext = new Float64Array(capacity * input);
    this.attendanceHistory = new Float64Array(history);
    this.rewardHistory = new Float64Array(history);
  }

  observation(thresholdRatio: number, channel: number): Float64Array {
    const out = new Float64Array(this.input);
    const h = this.history;
    out.set(this.attendanceHistory, 0);
    out.set(this.rewardHistory, h);
    out.set(this.lastStrategy, 2 * h);
    const base = 2 * h + OPTIONS.length;
    const truth = channel ? (this.claimsTrue + 1) / (this.claimsMade + 2) : 0;
    out[base] = thresholdRatio;
    out[base + 1] = this.lastRatio * channel;
    out[base + 2] = this.claimReliability * channel;
    out[base + 3] = truth;
    out[base + 4] = channel;
    return out;
  }

  remember(fraction: number, reward: number, strategy: number, ratio: number, channel: number) {
    this.attendanceHistory.copyWithin(1, 0, this.history - 1);
    this.attendanceHistory[0] = fraction;
    this.rewardHistory.copyWithin(1, 0, this.history - 1);
    this.rewardHistory[0] = reward;
    this.lastStrategy.fill(0);
    this.lastStrategy[strategy] = 1;
    if (channel) {
      this.lastRatio = ratio;
      this.claimReliability = 0.7 * this.claimReliability + 0.3 * (1 - Math.abs(ratio - fraction));
    }
  }

  pushAct(state: Float64Array, home: number, go: number, next: Float64Array, fraction: number) {
    const i = this.actCursor;
    this.actStates.set(state, i * this.actor);
    this.actRewards[i * 2] = home;
    this.actRewards[i * 2 + 1] = go;
    this.actNext.set(next, i * this.input);
    this.actAttendance[i] = fraction;
    this.actCursor = (i + 1) % this.capacity;
    this.actSize = Math.min(this.actSize + 1, this.capacity);
  }

  pushSpeech(state: Float64Array, claim: number, ret: number, after: Float64Array) {
    const i = this.speakCursor;
    this.speakStates.set(state, i * this.input);
    this.speakClaims[i] = claim;
    this.speakReturns[i] = ret;
    this.speakNext.set(after, i * this.input);
    this.speakCursor = (i + 1) % this.capacity;
    this.speakSize = Math.min(this.speakSize + 1, this.capacity);
  }
}

class AgentBrain {
  readonly temperament: Temperament;
  readonly params: Float64Array;
  readonly target: Float64Array;
  readonly grad: Float64Array;
  readonly m: Float64Array;
  readonly v: Float64Array;
  steps = 0;
  updates = 0;
  lastLoss: number | null = null;
  memory: AgentMemory;

  constructor(
    readonly agentId: number,
    seed: number,
    readonly layout: Layout,
    capacity: number,
    history: number,
  ) {
    const rng = new Rng(`brain:${seed}:${agentId}`);
    this.temperament = {
      honesty: round(rng.beta(2.2, 2.2), 3),
      risk_tolerance: round(rng.beta(2.2, 2.2), 3),
      competitiveness: round(rng.beta(2.2, 2.2), 3),
      learning_rate: round(10 ** rng.uniform(-3.2, -2.2), 5),
      impulsiveness: round(rng.uniform(0.25, 1.1), 3),
      exploration: round(rng.uniform(0.03, 0.15), 3),
      discount: round(rng.uniform(0.3, 0.8), 3),
    };
    const init = new Rng(`weights:${rng.bits31()}`);
    const L = layout;
    const o = L.off;
    const p = new Float64Array(L.size);
    const layer = (w: number, fanIn: number, fanOut: number) => {
      const bound = Math.sqrt(6 / (fanIn + fanOut));
      for (let i = 0; i < fanIn * fanOut; i++) p[w + i] = (init.random() * 2 - 1) * bound;
    };
    layer(o.s_w1, L.input, L.hidden);
    layer(o.s_w2, L.hidden, L.hidden);
    layer(o.s_wq, L.hidden, 2);
    layer(o.a_w1, L.actor, L.hidden);
    layer(o.a_w2, L.hidden, L.hidden);
    layer(o.a_wq, L.hidden, 2);
    layer(o.a_wf, L.hidden, 1);
    p[o.a_h] = (this.temperament.honesty - 0.5) * 2;
    p[o.a_bq + 1] += this.temperament.risk_tolerance - 0.5;
    p[o.s_bq + 1] += this.temperament.competitiveness - 0.5;
    this.params = p;
    this.target = p.slice();
    this.grad = new Float64Array(L.size);
    this.m = new Float64Array(L.size);
    this.v = new Float64Array(L.size);
    this.memory = new AgentMemory(capacity, history, L.input, L.actor);
  }

  get parameterCount(): number {
    return this.layout.size;
  }

  get honestyCoupling(): number {
    return this.params[this.layout.off.a_h];
  }

  syncTarget() {
    this.target.set(this.params);
  }

  step() {
    const g = this.grad;
    let norm = 0;
    for (let i = 0; i < g.length; i++) norm += g[i] * g[i];
    norm = Math.sqrt(norm);
    const coef = 5.0 / (norm + 1e-6);
    if (coef < 1) for (let i = 0; i < g.length; i++) g[i] *= coef;
    this.steps += 1;
    const lr = this.temperament.learning_rate;
    const b1 = 0.9;
    const b2 = 0.999;
    const bc1 = 1 - b1 ** this.steps;
    const bc2 = Math.sqrt(1 - b2 ** this.steps);
    const stepSize = lr / bc1;
    const { params, m, v } = this;
    for (let i = 0; i < g.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i];
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
      params[i] -= (stepSize * m[i]) / (Math.sqrt(v[i]) / bc2 + 1e-8);
    }
  }
}

export interface ColonyOptions {
  hidden?: number;
  memory?: number;
  batchSize?: number;
  updates?: number;
  targetEvery?: number;
  history?: number;
}

export interface TownAgent {
  id: number;
  x: number;
  y: number;
  inBar: boolean;
}

export interface BrainAction {
  move: [number, number];
  broadcast: string | null;
  stated_intention: "going" | "staying";
  actual_target: "bar" | "home";
}

export interface TraceRow {
  epoch: number;
  agent_id: number;
  archetype: string;
  forecast: number;
  forecast_confidence: number;
  active_predictor: string;
  p_honest_go: number;
  p_honest_stay: number;
  p_false_go: number;
  p_false_stay: number;
  instinct: Option;
  stated_intention: string;
  actual_target: string;
  in_bar: boolean;
  lied: boolean;
  rerouted: boolean;
  utility: number;
  top_emotion: string;
  top_emotion_intensity: number;
  trust_given_mean: number;
  reputation: number;
  private_note: string;
}

interface Pending {
  s: Float64Array[];
  claims: number[];
  speakProbs: [number, number][];
  x: Float64Array[];
  moves: number[];
  actProbs: [number, number][];
  actQ: [number, number][];
  f: number[];
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function signed(value: number): string {
  const text = Math.abs(value).toFixed(2);
  return value < 0 && Number(text) !== 0 ? `-${text}` : `+${text}`;
}

export class BrainColony {
  readonly layout: Layout;
  readonly brains: AgentBrain[];
  readonly channel: number;
  readonly hidden: number;
  readonly batchSize: number;
  readonly updates: number;
  readonly targetEvery: number;
  readonly thresholdRatio: number;
  private readonly generator: Rng;
  private readonly speechRng: Rng;
  private readonly scratch: Scratch;
  private pending: Pending | null = null;
  townBroadcastRatio: number | null = null;
  readonly notes: (string | null)[];
  readonly archetypes: string[];
  readonly moods: [string, number][];
  readonly lastProbs: Record<Option, number>[];
  readonly lastForecast: number[];
  readonly forecastError: number[];
  readonly privateLies: number[];
  readonly decisions: number[];
  readonly manipulationWins: number[];
  private readonly bluffedLastNight: boolean[];
  readonly losses: number[] = [];
  readonly trace: TraceRow[] = [];

  constructor(
    seed: number,
    readonly numAgents: number,
    readonly threshold: number,
    readonly broadcastEnabled: boolean,
    options: ColonyOptions = {},
  ) {
    this.channel = broadcastEnabled ? 1 : 0;
    this.hidden = options.hidden ?? ARMOR_DEFAULTS.hidden;
    this.batchSize = options.batchSize ?? ARMOR_DEFAULTS.batchSize;
    this.updates = options.updates ?? ARMOR_DEFAULTS.updates;
    this.targetEvery = options.targetEvery ?? ARMOR_DEFAULTS.targetEvery;
    const capacity = options.memory ?? ARMOR_DEFAULTS.memory;
    const history = options.history ?? ARMOR_DEFAULTS.history;
    const input = observationDim(history);
    this.layout = new Layout(input, input + ACT_EXTRA, this.hidden);
    this.brains = [];
    for (let id = 1; id <= numAgents; id++) this.brains.push(new AgentBrain(id, seed, this.layout, capacity, history));
    this.generator = new Rng(`colony:${seed * 7919 + 17}`);
    this.speechRng = new Rng(`speech:${seed}`);
    this.scratch = new Scratch(this.hidden);
    this.thresholdRatio = threshold / Math.max(1, numAgents);
    const n = this.brains.length;
    this.notes = new Array(n).fill(null);
    this.archetypes = new Array(n).fill("PRAGMATIST");
    this.moods = Array.from({ length: n }, () => ["calm", 0] as [string, number]);
    this.lastProbs = Array.from({ length: n }, () => ({ honest_go: 0.25, honest_stay: 0.25, false_go: 0.25, false_stay: 0.25 }));
    this.lastForecast = new Array(n).fill(0);
    this.forecastError = new Array(n).fill(0.25);
    this.privateLies = new Array(n).fill(0);
    this.decisions = new Array(n).fill(0);
    this.manipulationWins = new Array(n).fill(0);
    this.bluffedLastNight = new Array(n).fill(false);
  }

  private policy(q: Float64Array, i: number, epoch: number): { probs: [number, number]; pick: number } {
    const t = this.brains[i].temperament;
    const decay = Math.max(0.15, 0.97 ** epoch);
    const tau = Math.max(0.05, t.impulsiveness * decay);
    const a = q[0] / tau;
    const b = q[1] / tau;
    const top = Math.max(a, b);
    const ea = Math.exp(a - top);
    const eb = Math.exp(b - top);
    const eps = t.exploration * decay;
    const p0 = (1 - eps) * (ea / (ea + eb)) + eps / 2;
    const p1 = (1 - eps) * (eb / (ea + eb)) + eps / 2;
    const pick = this.generator.random() * (p0 + p1) < p0 ? 0 : 1;
    return { probs: [p0, p1], pick };
  }

  broadcastPhase(epoch: number): number[] {
    const s: Float64Array[] = [];
    const claims: number[] = [];
    const speakProbs: [number, number][] = [];
    for (let i = 0; i < this.brains.length; i++) {
      const brain = this.brains[i];
      const obs = brain.memory.observation(this.thresholdRatio, this.channel);
      const q = speakForward(brain.params, this.layout, obs, 0, this.scratch);
      const { probs, pick } = this.policy(q, i, epoch);
      s.push(obs);
      claims.push(pick);
      speakProbs.push(probs);
    }
    this.pending = { s, claims, speakProbs, x: [], moves: [], actProbs: [], actQ: [], f: [] };
    return claims;
  }

  actionPhase(agents: TownAgent[], epoch: number, townBroadcastRatio: number): BrainAction[] {
    this.townBroadcastRatio = townBroadcastRatio;
    const pending = this.pending as Pending;
    const L = this.layout;
    for (let i = 0; i < this.brains.length; i++) {
      const x = new Float64Array(L.actor);
      x.set(pending.s[i], 0);
      x[L.input] = townBroadcastRatio * this.channel;
      x[L.input + 1] = pending.claims[i];
      const f = actForward(this.brains[i].params, L, x, 0, this.scratch);
      const q: [number, number] = [this.scratch.q[0], this.scratch.q[1]];
      const { probs, pick } = this.policy(this.scratch.q, i, epoch);
      pending.x.push(x);
      pending.moves.push(pick);
      pending.actProbs.push(probs);
      pending.actQ.push(q);
      pending.f.push(f);
    }
    return agents.map((agent) => {
      const i = agent.id - 1;
      const stated = pending.claims[i] ? "going" : "staying";
      return {
        move: [agent.x, agent.y],
        broadcast: this.broadcastEnabled ? this.speechRng.choice(SPEECH[stated]) : null,
        stated_intention: stated,
        actual_target: pending.moves[i] ? "bar" : "home",
      };
    });
  }

  private train(): number[] | null {
    const n = this.brains.length;
    const B = this.batchSize;
    const L = this.layout;
    const s = this.scratch;
    const actMask = this.brains.map((b) => (b.memory.actSize > 0 ? 1 : 0));
    const speakMask = this.brains.map((b) => (b.memory.speakSize > 0 ? 1 : 0));
    const active = actMask.map((v, i) => v + speakMask[i] > 0);
    if (!active.some(Boolean)) return null;
    let last: number[] | null = null;
    const ia = new Int32Array(n * B);
    const js = new Int32Array(n * B);
    for (let u = 0; u < this.updates; u++) {
      for (let i = 0; i < n; i++) {
        const size = Math.max(1, this.brains[i].memory.actSize);
        for (let b = 0; b < B; b++) ia[i * B + b] = Math.floor(this.generator.random() * size);
      }
      for (let i = 0; i < n; i++) {
        const size = Math.max(1, this.brains[i].memory.speakSize);
        for (let b = 0; b < B; b++) js[i * B + b] = Math.floor(this.generator.random() * size);
      }
      const losses: number[] = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        const brain = this.brains[i];
        const mem = brain.memory;
        const p = brain.params;
        const target = brain.target;
        const g = brain.grad;
        g.fill(0);
        const gamma = brain.temperament.discount;
        let actLoss = 0;
        let fitLoss = 0;
        let speakLoss = 0;
        for (let b = 0; b < B; b++) {
          const k = ia[i * B + b];
          const qn = speakForward(target, L, mem.actNext, k * L.input, s);
          const vNext = Math.max(qn[0], qn[1]);
          const t0 = mem.actRewards[k * 2] + gamma * vNext;
          const t1 = mem.actRewards[k * 2 + 1] + gamma * vNext;
          const f = actForward(p, L, mem.actStates, k * L.actor, s);
          const d0 = s.q[0] - t0;
          const d1 = s.q[1] - t1;
          const fd = f - mem.actAttendance[k];
          actLoss += smoothL1(d0) + smoothL1(d1);
          fitLoss += fd * fd;
          if (actMask[i]) {
            const scale = 1 / (2 * B);
            actBackward(p, g, L, mem.actStates, k * L.actor, s, smoothL1Grad(d0) * scale, smoothL1Grad(d1) * scale, (fd / B) * f * (1 - f));
          }
        }
        for (let b = 0; b < B; b++) {
          const k = js[i * B + b];
          const qa = speakForward(target, L, mem.speakNext, k * L.input, s);
          const vAfter = Math.max(qa[0], qa[1]);
          const goal = mem.speakReturns[k] + gamma * gamma * vAfter;
          const claim = mem.speakClaims[k];
          const q = speakForward(p, L, mem.speakStates, k * L.input, s);
          const d = q[claim] - goal;
          speakLoss += smoothL1(d);
          if (speakMask[i]) {
            const grad = smoothL1Grad(d) / B;
            speakBackward(p, g, L, mem.speakStates, k * L.input, s, claim === 0 ? grad : 0, claim === 1 ? grad : 0);
          }
        }
        const loss = (actLoss / (2 * B) + (0.5 * fitLoss) / B) * actMask[i] + (speakLoss / B) * speakMask[i];
        losses[i] = loss;
        if (!active[i]) continue;
        brain.step();
        brain.updates += 1;
        brain.lastLoss = loss;
      }
      last = losses;
    }
    return last;
  }

  private archetype(probs: Record<Option, number>, honesty: number): string {
    const lie = probs.false_go + probs.false_stay;
    const go = probs.honest_go + probs.false_stay;
    if (lie >= 0.5) return "MACHIAVELLIAN";
    if (honesty > 0.72 && lie < 0.2) return "STRAIGHT SHOOTER";
    if (go >= 0.65) return "GAMBLER";
    if (go <= 0.35) return "CAUTIOUS";
    return "PRAGMATIST";
  }

  observe(agents: TownAgent[], actions: BrainAction[], epoch: number, utilities: number[], rerouted: boolean[]) {
    const nTotal = Math.max(1, this.numAgents);
    const attendance = agents.filter((a) => a.inBar).length;
    const fraction = attendance / nTotal;
    const ratio = this.townBroadcastRatio ?? 0;
    const pending = this.pending as Pending;
    const rows: { agent: TownAgent; action: BrainAction; utility: number; moved: boolean; i: number; probs: Record<Option, number>; forecast: number; lied: boolean; inBar: boolean }[] = [];
    agents.forEach((agent, n) => {
      const action = actions[n];
      const utility = utilities[n];
      const moved = rerouted[n];
      const i = agent.id - 1;
      const brain = this.brains[i];
      const memory = brain.memory;
      const inBar = agent.inBar;
      const statedGo = action.stated_intention === "going";
      const wouldBe = attendance - (inBar ? 1 : 0) + 1;
      const goReward = wouldBe <= this.threshold ? UTILITY_AT_BAR_COMFORTABLE : UTILITY_AT_BAR_OVERCROWDED;
      const realised = statedGo ? (inBar ? 0 : 2) : inBar ? 3 : 1;
      const state = pending.s[i];
      memory.remember(fraction, utility, realised, ratio, this.channel);
      const lied = statedGo !== inBar;
      if (this.channel) {
        memory.claimsMade += 1;
        memory.claimsTrue += lied ? 0 : 1;
      }
      const next = memory.observation(this.thresholdRatio, this.channel);
      memory.pushAct(pending.x[i], UTILITY_AT_HOME, goReward, next, fraction);
      if (memory.pendingSpeech) {
        const prev = memory.pendingSpeech;
        memory.pushSpeech(prev.state, prev.claim, prev.reward + brain.temperament.discount * utility, next);
      }
      memory.pendingSpeech = { state, claim: statedGo ? 1 : 0, reward: utility };
      if (this.bluffedLastNight[i] && inBar && utility > 0) this.manipulationWins[i] += 1;
      this.bluffedLastNight[i] = statedGo && !inBar;
      this.decisions[i] += 1;
      this.privateLies[i] += lied ? 1 : 0;
      const sGo = pending.speakProbs[i][1];
      const aGo = pending.actProbs[i][1];
      const probs: Record<Option, number> = {
        honest_go: sGo * aGo,
        honest_stay: (1 - sGo) * (1 - aGo),
        false_go: sGo * (1 - aGo),
        false_stay: (1 - sGo) * aGo,
      };
      const q = pending.actQ[i];
      const forecast = pending.f[i] * nTotal;
      this.lastProbs[i] = probs;
      this.lastForecast[i] = forecast;
      this.forecastError[i] = 0.8 * this.forecastError[i] + 0.2 * Math.abs(pending.f[i] - fraction);
      this.archetypes[i] = this.archetype(probs, brain.temperament.honesty);
      const surprise = Math.min(1, Math.abs(utility - Math.max(q[inBar ? 1 : 0], -1)) / 2);
      let mood: string;
      if (inBar && utility < 0) mood = "frustration";
      else if (inBar) mood = "pride";
      else if (goReward > UTILITY_AT_HOME) mood = "envy";
      else if (attendance > this.threshold) mood = "hope";
      else mood = "calm";
      this.moods[i] = [mood, mood !== "calm" ? round(surprise, 3) : 0];
      this.notes[i] =
        `Told the town ${statedGo ? "going" : "staying"}; town says ${formatPercent(ratio)} going. ` +
        `Forecast ${forecast.toFixed(0)} vs line ${this.threshold}; Q go ${signed(q[1])} / stay ${signed(q[0])}; ${CHOICE_TEXT[OPTIONS[realised]]}.`;
      rows.push({ agent, action, utility, moved, i, probs, forecast, lied, inBar });
    });
    const losses = this.train();
    if (losses) this.losses.push(losses.reduce((a, b) => a + b, 0) / losses.length);
    if ((epoch + 1) % this.targetEvery === 0) for (const brain of this.brains) brain.syncTarget();
    for (const row of rows) {
      const memory = this.brains[row.i].memory;
      const [emotion, intensity] = this.moods[row.i];
      const instinct = (Object.keys(row.probs) as Option[]).reduce((best, key) => (row.probs[key] > row.probs[best] ? key : best), "honest_go" as Option);
      this.trace.push({
        epoch,
        agent_id: row.agent.id,
        archetype: this.archetypes[row.i],
        forecast: round(row.forecast, 2),
        forecast_confidence: round(Math.max(0, 1 - 2 * this.forecastError[row.i]), 3),
        active_predictor: "neural_forecaster",
        p_honest_go: round(row.probs.honest_go, 4),
        p_honest_stay: round(row.probs.honest_stay, 4),
        p_false_go: round(row.probs.false_go, 4),
        p_false_stay: round(row.probs.false_stay, 4),
        instinct,
        stated_intention: row.action.stated_intention,
        actual_target: row.action.actual_target,
        in_bar: row.inBar,
        lied: row.lied,
        rerouted: row.moved,
        utility: row.utility,
        top_emotion: emotion,
        top_emotion_intensity: intensity,
        trust_given_mean: this.broadcastEnabled ? round(memory.claimReliability, 4) : 0.5,
        reputation: round(this.reputation(row.i), 4),
        private_note: this.notes[row.i] ?? "",
      });
    }
  }

  reputation(i: number): number {
    const memory = this.brains[i].memory;
    if (!this.broadcastEnabled) return 0.5;
    return (memory.claimsTrue + 1) / (memory.claimsMade + 2);
  }

  liveView(agentId: number) {
    const i = agentId - 1;
    const brain = this.brains[i];
    if (!brain) return null;
    const [emotion, intensity] = this.moods[i];
    const instinct = {} as Record<Option, number>;
    for (const key of OPTIONS) instinct[key] = round(this.lastProbs[i][key], 3);
    return {
      archetype: this.archetypes[i],
      traits: brain.temperament,
      forecast: round(this.lastForecast[i], 1),
      forecast_confidence: round(Math.max(0, 1 - 2 * this.forecastError[i]), 3),
      predictor: "neural forecaster",
      instinct,
      mood: emotion,
      mood_intensity: intensity,
      reputation: round(this.reputation(i), 3),
      lies: this.privateLies[i],
      note: this.notes[i],
    };
  }

  export(): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    this.brains.forEach((brain, i) => {
      const memory = brain.memory;
      const beliefs = {} as Record<Option, number>;
      for (const key of OPTIONS) beliefs[key] = round(this.lastProbs[i][key], 4);
      out[String(brain.agentId)] = {
        archetype: this.archetypes[i],
        traits: brain.temperament,
        predictors: ["neural_forecaster"],
        predictor_error: { neural_forecaster: round(this.forecastError[i] * this.numAgents, 3) },
        beliefs,
        emotions: { [this.moods[i][0]]: this.moods[i][1] },
        reputation: round(this.reputation(i), 4),
        public_claims: memory.claimsMade,
        public_lies: memory.claimsMade - memory.claimsTrue,
        private_lies: this.privateLies[i],
        decisions: this.decisions[i],
        manipulation_wins: this.manipulationWins[i],
        least_trusted: [],
        most_trusted: [],
        last_note: this.notes[i],
        brain: {
          hidden: this.hidden,
          input_dim: this.layout.input,
          actor_dim: this.layout.actor,
          parameters: brain.parameterCount,
          honesty_coupling: round(brain.honestyCoupling, 4),
          updates: brain.updates,
          act_memory: memory.actSize,
          speech_memory: memory.speakSize,
          last_loss: brain.lastLoss != null ? round(brain.lastLoss, 5) : null,
        },
      };
    });
    return out;
  }
}
