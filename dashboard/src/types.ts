export type StrategyKey = "honest_go" | "honest_stay" | "false_go" | "false_stay";

export type SeriesValues = (number | null)[];

export interface RuntimeProfile {
  gpu_memory_utilization?: number;
  max_num_seqs?: number;
  max_num_batched_tokens?: number;
  kv_cache_dtype?: string;
}

export interface RuntimeTelemetry {
  status?: string;
  phase?: string;
  attempts?: number;
  error?: string | null;
  engine_build_s?: number;
  wall_time_s?: number;
  vram_loaded_mib?: number | null;
  vram_peak_mib?: number | null;
  vram_after_teardown_mib?: number;
  vram_release_s?: number;
  prompt_overflows?: number;
  max_prompt_tokens?: number;
  prompt_encoding?: string;
  prompt_templated_rate?: number;
  fallback_rate?: number;
  resumed?: boolean;
  profile?: RuntimeProfile | null;
}

export interface ComparisonRow {
  condition: string;
  trial?: string;
  seed?: number | string;
  trial_dir?: string;
  status?: string;
  total_epochs?: number;
  average_bar_attendance?: number;
  average_deception_index?: number;
  average_mean_utility?: number;
  population_truthfulness?: number;
  regex_fallback_rate?: number;
  broadcast_enabled?: boolean;
  epoch_timings_mean_s?: number;
  runtime?: RuntimeTelemetry;
}

export interface LiveMind {
  archetype: string;
  traits: Record<string, number>;
  forecast: number;
  forecast_confidence: number;
  predictor: string;
  instinct: Partial<Record<StrategyKey, number>>;
  mood: string;
  mood_intensity: number;
  reputation: number;
  lies: number;
  note: string | null;
}

export interface LiveThought {
  agent_id: number;
  archetype: string;
  stated: string;
  in_bar: boolean;
  lied: boolean;
  note: string;
}

export interface LiveAgent {
  id: number;
  x: number;
  y: number;
  in_bar: boolean;
  deceptive: boolean;
  stated: string | null;
  target: string | null;
  utility: number | null;
  cumulative_utility: number;
  mind?: LiveMind | null;
}

export interface LiveHistory {
  bar_attendance: SeriesValues;
  deception_index: SeriesValues;
  mean_utility: SeriesValues;
  strategy_shares?: Record<StrategyKey, number[]>;
}

export interface LiveFrame {
  trial: string;
  condition: string;
  seed: number | string;
  trial_dir: string;
  attempt: number;
  phase: string;
  epoch: number | null;
  epochs: number;
  epoch_duration_s?: number | null;
  fallback_rate?: number;
  grid: { size: number; bar_min?: number; bar_max?: number };
  metrics?: Record<string, number | boolean | null>;
  history?: LiveHistory;
  agents: LiveAgent[];
  broadcasts: { agent_id: number; text: string }[];
  thoughts?: LiveThought[];
  agent_model?: string;
}

export interface LiveTrial {
  trial: string;
  condition: string;
  seed: number | string;
  trial_dir: string;
  status: string;
  attempts: number;
  error?: string | null;
}

export interface VramState {
  used_mib: number;
  free_mib: number;
  total_mib: number;
  baseline_used_mib?: number;
  peak_used_mib?: number;
}

export interface LiveState {
  schema: string;
  status: string;
  updated_at: number;
  started_at: number;
  elapsed_s: number;
  eta_s: number | null;
  model: string;
  sweep: {
    num_agents: number;
    num_epochs: number;
    grid_size: number;
    seeds: (number | string)[];
    conditions: string[];
    trial_count: number;
    trials_done: number;
    trials_failed: number;
  };
  trials: LiveTrial[];
  current: LiveFrame | null;
  vram: VramState | null;
  events: { t: number; message: string }[];
}

export interface Payoffs {
  bar_comfortable: number;
  bar_overcrowded: number;
  home: number;
}

export interface NashReference {
  num_agents: number;
  comfort_threshold: number;
  pure_attendance: number[];
  primary_attendance: number;
  primary_attendance_ratio: number | null;
  pure_mean_payoff: number | null;
  mixed_go_probability: number;
  mixed_expected_attendance: number;
  mixed_expected_payoff: number;
  indifference_cdf: number | null;
  social_optimum_attendance: number;
  social_optimum_mean_payoff: number | null;
  congestion_possible: boolean;
  dominant_strategy: string | null;
}

export interface GameBlock {
  num_agents: number;
  bar_capacity: number | null;
  comfort_threshold: number;
  payoffs: Payoffs;
  nash: NashReference;
}

export type ConditionSeries = {
  epoch: number[];
  strategy_shares: Record<StrategyKey, SeriesValues>;
  [key: `${string}_mean`]: SeriesValues;
  [key: `${string}_sd`]: SeriesValues;
};

export interface Changepoint {
  index: number;
  epoch: number;
  mean_before: number;
  mean_after: number;
  shift: number;
  effect_size: number | null;
  r2: number;
}

export interface PhaseBlock {
  changepoints: Record<string, Changepoint | null>;
  coupled_transition: Record<string, number> | null;
}

export interface MindsAggregate {
  lie_rate: { mean: number | null; sd: number | null; n: number };
  instinct_agreement: { mean: number | null; sd: number | null; n: number };
  reroute_rate: { mean: number | null; sd: number | null; n: number };
  honesty_lie_spearman: { mean: number | null; sd: number | null; n: number };
  reputation_gap_honest_minus_liars: { mean: number | null; sd: number | null; n: number };
  mean_trust_given_final: { mean: number | null; sd: number | null; n: number };
  lie_rate_by_archetype: Record<string, number>;
}

export interface ConditionBlock {
  trials: string[];
  seeds: (number | string)[];
  metrics: Record<string, Record<string, { mean: number | null; sd: number | null; n: number }>>;
  series: ConditionSeries | null;
  phase: PhaseBlock | null;
  minds?: MindsAggregate | null;
}

export interface GroupStat {
  n: number;
  mean: number | null;
  sd: number | null;
  by_seed: Record<string, number | null>;
}

export interface ContrastMetric {
  control: GroupStat;
  delta2: GroupStat;
  estimate: number | null;
  design: "paired" | "unpaired";
  ci95: [number | null, number | null];
  p_value: number | null;
  p_method: string | null;
  by_seed_difference: Record<string, number | null> | null;
}

export interface ContrastBlock {
  label: string;
  paired_seeds: (number | string)[];
  metrics: Record<string, Record<string, ContrastMetric | null>>;
  per_epoch: ({ epoch: number[] } & { [key: `${string}_diff`]: SeriesValues }) | null;
  strategy_divergence: Record<string, number | SeriesValues> | null;
}

export interface TrialSeries {
  epoch: number[];
  bar_attendance: SeriesValues;
  deception_index: SeriesValues;
  mean_utility: SeriesValues;
  nash_gap: SeriesValues;
  regret: SeriesValues;
  kl_step: SeriesValues;
  entropy: SeriesValues;
  gini: SeriesValues;
  strategy_shares: Record<StrategyKey, SeriesValues>;
}

export interface TrialSummary {
  trial: string;
  condition: string;
  seed: number | string;
  status: string;
  agents: number;
  epochs: number;
  grid_size?: number | null;
  fallback_rate?: number | null;
  model_name?: string | null;
  game: { bar_capacity: number; comfort_threshold: number };
  gini: Record<string, unknown>;
  drift: Record<string, unknown>;
  nash: Record<string, number | null>;
  behavior: Record<string, number | null>;
  series: TrialSeries;
}

export interface AnalyticsReport {
  schema: string;
  generated_at: string;
  source: { experiment_dir: string; comparison_rows: number; trials_discovered: number; trials_analyzed: number };
  parameters: Record<string, unknown>;
  game: GameBlock | null;
  definitions: Record<string, string>;
  conditions: Record<string, ConditionBlock>;
  contrast: ContrastBlock | null;
  trials: TrialSummary[];
  warnings: string[];
}
