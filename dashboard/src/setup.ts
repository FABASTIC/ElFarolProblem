export type Engine = "llm" | "rehearsal";

export interface LaunchConfig {
  members: number;
  epochs: number;
  seeds: number[];
  engine: Engine;
  pace: number;
}

export type LauncherState = "idle" | "launching" | "running" | "stopping" | "analyzing" | "exited" | "failed" | "detached";

export interface EngineCapability {
  available: boolean | null;
  detail: string;
}

export interface LauncherStatus {
  state: LauncherState;
  engine: Engine | null;
  config: LaunchConfig | null;
  pid: number | null;
  startedAt: number | null;
  endedAt: number | null;
  exitCode: number | null;
  archivedTo: string | null;
  error: string | null;
  log: string[];
  model: string;
  dataDir: string;
  capabilities: Record<Engine, EngineCapability>;
}

export const BAR_CAPACITY = 100;
export const COMFORT_THRESHOLD = 60;

export const LIMITS = {
  members: { min: 4, max: 200 },
  epochs: { min: 1, max: 500 },
  seeds: { min: 1, max: 12 },
  pace: { min: 0, max: 10 },
} as const;

export const RECOMMENDED: LaunchConfig = {
  members: 100,
  epochs: 50,
  seeds: [42, 100, 2026],
  engine: "llm",
  pace: 1.5,
};

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  members: number;
  epochs: number;
  seeds: number[];
  recommended?: boolean;
}

export const PRESETS: Preset[] = [
  { id: "scout", label: "SCOUT", blurb: "Fast sanity pass, one seed", members: 80, epochs: 12, seeds: [42] },
  {
    id: "classic",
    label: "CLASSIC",
    blurb: "Arthur's El Farol: 100 patrons, comfort line 60",
    members: 100,
    epochs: 50,
    seeds: [42, 100, 2026],
    recommended: true,
  },
  { id: "deep", label: "DEEP STUDY", blurb: "Six paired seeds, p-floor 0.031", members: 120, epochs: 100, seeds: [42, 100, 2026, 7, 13, 99] },
];

export const SECONDS_PER_DECISION = 0.003;
export const ENGINE_BUILD_S = 2;

function asInt(value: unknown): number | null {
  const n = typeof value === "string" && value.trim() !== "" ? Number(value) : value;
  return typeof n === "number" && Number.isFinite(n) && Number.isInteger(n) ? n : null;
}

export function sanitizeConfig(input: unknown): { config: LaunchConfig | null; errors: string[] } {
  const errors: string[] = [];
  const raw = (input && typeof input === "object" ? input : {}) as Record<string, unknown>;
  const members = asInt(raw.members);
  const epochs = asInt(raw.epochs);
  const engine = raw.engine === "rehearsal" || raw.engine === "llm" ? raw.engine : null;
  const pace = typeof raw.pace === "number" && Number.isFinite(raw.pace) ? raw.pace : RECOMMENDED.pace;
  const seedList = Array.isArray(raw.seeds) ? raw.seeds.map(asInt) : [];
  if (members == null || members < LIMITS.members.min || members > LIMITS.members.max) {
    errors.push(`Members must be a whole number from ${LIMITS.members.min} to ${LIMITS.members.max}.`);
  }
  if (epochs == null || epochs < LIMITS.epochs.min || epochs > LIMITS.epochs.max) {
    errors.push(`Nights must be a whole number from ${LIMITS.epochs.min} to ${LIMITS.epochs.max}.`);
  }
  if (!engine) errors.push("Pick an engine: Isolated PyTorch Tensors or rehearsal.");
  if (seedList.some((s) => s == null || s < 0 || s > 2 ** 31 - 1)) errors.push("Seeds must be non-negative whole numbers.");
  const seeds = [...new Set(seedList.filter((s): s is number => s != null && s >= 0))];
  if (seeds.length < LIMITS.seeds.min || seeds.length > LIMITS.seeds.max) {
    errors.push(`Use ${LIMITS.seeds.min} to ${LIMITS.seeds.max} distinct seeds.`);
  }
  if (pace < LIMITS.pace.min || pace > LIMITS.pace.max) errors.push(`Pace must be ${LIMITS.pace.min} to ${LIMITS.pace.max} seconds per night.`);
  if (errors.length) return { config: null, errors };
  return { config: { members: members as number, epochs: epochs as number, seeds, engine: engine as Engine, pace }, errors };
}

export interface RunEstimate {
  trials: number;
  decisions: number;
  seconds: number;
  calibrated: boolean;
}

export function estimateRun(config: LaunchConfig, secondsPerDecision: number | null): RunEstimate {
  const trials = config.seeds.length * 2;
  const decisions = trials * config.members * config.epochs;
  if (config.engine === "rehearsal") {
    return { trials, decisions, seconds: trials * config.epochs * (config.pace + 0.02), calibrated: true };
  }
  const perDecision = secondsPerDecision ?? SECONDS_PER_DECISION;
  return { trials, decisions, seconds: decisions * perDecision + trials * ENGINE_BUILD_S, calibrated: secondsPerDecision != null };
}

export interface SetupAdvice {
  tone: "good" | "warn" | "fail";
  text: string;
}

export function adviseConfig(config: LaunchConfig): SetupAdvice[] {
  const advice: SetupAdvice[] = [];
  if (config.members <= COMFORT_THRESHOLD) {
    advice.push({
      tone: "warn",
      text: `${config.members} members can never overcrowd a ${COMFORT_THRESHOLD}-seat comfort line: going is strictly dominant, so there is no El Farol dilemma. Use more than ${COMFORT_THRESHOLD}.`,
    });
  } else {
    advice.push({ tone: "good", text: `${config.members} members against a comfort line of ${COMFORT_THRESHOLD}: the bar can overcrowd, so the dilemma is live.` });
  }
  if (config.epochs < 20) {
    advice.push({ tone: "warn", text: "Under 20 nights, forecasters barely re-rank and trust has little time to form." });
  }
  if (config.seeds.length < 3) {
    advice.push({ tone: "warn", text: "With fewer than 3 paired seeds the control vs broadcast contrast cannot reach significance." });
  }
  if (config.engine === "rehearsal") {
    advice.push({ tone: "warn", text: "Rehearsal is a slowed-down preview engine and is not available in this build." });
  }
  return advice;
}
