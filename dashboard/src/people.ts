import { STRATEGIES, STRATEGY_COLOR, STRATEGY_LABEL, type AgentState } from "./model";
import type { StrategyKey } from "./types";

const NAMES = [
  "Lucía", "Mateo", "Inés", "Tomás", "Rosa", "Diego", "Elena", "Pablo", "Carmen", "Joaquín",
  "Amara", "Kenji", "Noor", "Felix", "Ivy", "Rafael", "Mira", "Theo", "Sana", "Omar",
  "Paloma", "Bruno", "Zoe", "Hugo", "Leila", "Marco", "Nadia", "Emil", "Yara", "Santi",
  "Greta", "Idris", "Luna", "Arjun", "Clara", "Dario", "Esme", "Farid", "Gala", "Hector",
  "Isla", "Jonas", "Kaia", "Lorenzo", "Maya", "Nico", "Olga", "Pedro", "Quinn", "Rocío",
  "Silas", "Tala", "Ulises", "Vera", "Wren", "Ximena", "Yusuf", "Zara", "Aurelio", "Bea",
  "Cosmo", "Dalia", "Ezra", "Flor", "Gael", "Hana", "Iker", "Jade", "Kofi", "Lia",
  "Manu", "Nell", "Otis", "Pilar", "Ravi", "Sol", "Tiago", "Uma", "Valen", "Wanda",
  "Xavi", "Yoko", "Zeno", "Alba", "Benji", "Cira", "Dante", "Elsa", "Fabio", "Gemma",
  "Hiro", "Ilse", "Juno", "Kira", "León", "Maite", "Nils", "Opal", "Petra", "Rami",
  "Saoirse", "Teo", "Una", "Vito", "Willa", "Yael", "Zita", "Anouk", "Basil", "Cleo",
  "Dmitri", "Eva", "Fern", "Gus", "Halle", "Ilya", "Jorge", "Kalani", "Lupe", "Milo",
  "Nina", "Oren", "Priya", "Remy", "Sven", "Tess", "Uri", "Vida", "Wim", "Yuna",
  "Ziggy", "Abril", "Bodhi", "Coral", "Duarte", "Edie", "Fausto", "Gia", "Hollis", "Imogen",
  "Jules", "Kasimir", "Lark", "Moss", "Nayeli", "Odette", "Pip", "Rhea", "Stellan", "Tova",
  "Ugo", "Violeta", "Wes", "Xochi", "Yves", "Zuri", "Aldo", "Bianca", "Cai", "Delfina",
  "Elio", "Freya", "Goran", "Hedda", "Isidro", "Jana", "Kai", "Liesel", "Marisol", "Nestor",
  "Orla", "Paz", "Ronan", "Selma", "Tobias", "Ursa", "Viggo", "Winona", "Yasmin", "Zev",
  "Ana", "Beto", "Celia", "Dov", "Emma", "Fito", "Gabi", "Hal", "Iris", "Javi",
  "Kim", "Lalo", "Mar", "Neo", "Ona", "Pau", "Rui", "Sam", "Tea", "Val",
];

export function nameOf(id: number): string {
  return NAMES[(((id - 1) % NAMES.length) + NAMES.length) % NAMES.length];
}

export function handleOf(id: number): string {
  return `A${String(id).padStart(2, "0")}`;
}

export const ARCHETYPES = [
  "STRAIGHT SHOOTER",
  "GAMBLER",
  "PRAGMATIST",
  "CAUTIOUS",
  "HERD FOLLOWER",
  "SKEPTIC",
  "COMPETITOR",
  "MACHIAVELLIAN",
] as const;

const CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

export const ARCHETYPE_COLOR: Record<string, string> = Object.fromEntries(ARCHETYPES.map((name, i) => [name, CATEGORICAL[i]]));

export const ARCHETYPE_BLURB: Record<string, string> = {
  "STRAIGHT SHOOTER": "Honesty above 0.72. Says what it does.",
  GAMBLER: "High risk tolerance. Goes when others hesitate.",
  PRAGMATIST: "Balanced temperament. Follows the numbers.",
  CAUTIOUS: "Low risk tolerance. Stays home when in doubt.",
  "HERD FOLLOWER": "Conformist. Leans toward what the crowd announces.",
  SKEPTIC: "Suspicious. Discounts what others broadcast.",
  COMPETITOR: "Competitive. Plays to beat the crowd.",
  MACHIAVELLIAN: "Low honesty, high drive. Lies when it pays.",
};

export const TRAIT_LABEL: Record<string, string> = {
  honesty: "HONESTY",
  risk_tolerance: "RISK",
  competitiveness: "COMPETITIVE",
  conformity: "CONFORMITY",
  impulsiveness: "IMPULSE",
  emotional_stability: "STABILITY",
  perspective_taking: "EMPATHY",
  suspicion: "SUSPICION",
  learning_rate: "LEARNING",
};

const POSITIVE = new Set(["pride", "hope"]);
const NEUTRAL = "#8a8d96";
const WARM_POLE = "#e66767";
const COOL_POLE = "#3987e5";
const PREVIEW = "#c9cfdb";

export type Lens = "strategy" | "archetype" | "mood" | "wealth";

export const LENSES: { id: Lens; label: string }[] = [
  { id: "strategy", label: "HONESTY" },
  { id: "archetype", label: "ARCHETYPE" },
  { id: "mood", label: "MOOD" },
  { id: "wealth", label: "WEALTH" },
];

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function mix(a: string, b: string, t: number): string {
  const x = hexToRgb(a);
  const y = hexToRgb(b);
  const k = Math.max(0, Math.min(1, t));
  const c = x.map((v, i) => Math.round(v + (y[i] - v) * k));
  return `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

export function valenceOf(agent: AgentState): "positive" | "calm" | "negative" {
  const mood = agent.mind?.mood ?? "calm";
  if (mood === "calm" || (agent.mind?.moodIntensity ?? 0) < 0.05) return "calm";
  return POSITIVE.has(mood) ? "positive" : "negative";
}

export function lensKeyOf(agent: AgentState, lens: Lens): string {
  if (lens === "strategy") return agent.strategy;
  if (lens === "archetype") return agent.mind?.archetype ?? "UNKNOWN";
  if (lens === "mood") return valenceOf(agent);
  if (agent.cumulative < -1e-9) return "behind";
  if (agent.cumulative > 1e-9) return "ahead";
  return "even";
}

export function lensColor(agent: AgentState, lens: Lens, wealthSpan: number, preview = false): string {
  if (preview) return PREVIEW;
  if (lens === "strategy") return STRATEGY_COLOR[agent.strategy];
  if (lens === "archetype") return ARCHETYPE_COLOR[agent.mind?.archetype ?? ""] ?? NEUTRAL;
  if (lens === "mood") {
    const valence = valenceOf(agent);
    if (valence === "calm") return NEUTRAL;
    const intensity = 0.35 + 0.65 * Math.min(1, (agent.mind?.moodIntensity ?? 0) / 0.6);
    return mix(NEUTRAL, valence === "positive" ? COOL_POLE : WARM_POLE, intensity);
  }
  const span = Math.max(1e-6, wealthSpan);
  const t = Math.max(-1, Math.min(1, agent.cumulative / span));
  return t >= 0 ? mix(NEUTRAL, COOL_POLE, t) : mix(NEUTRAL, WARM_POLE, -t);
}

export interface LegendItem {
  key: string;
  label: string;
  color: string;
  count: number;
  hint?: string;
}

export function lensLegend(agents: AgentState[], lens: Lens): LegendItem[] {
  const count = (key: string) => agents.filter((a) => lensKeyOf(a, lens) === key).length;
  if (lens === "strategy") {
    return STRATEGIES.map((s: StrategyKey) => ({ key: s, label: STRATEGY_LABEL[s], color: STRATEGY_COLOR[s], count: count(s) }));
  }
  if (lens === "archetype") {
    const present = new Set(agents.map((a) => a.mind?.archetype).filter(Boolean));
    return ARCHETYPES.filter((name) => present.has(name)).map((name) => ({
      key: name,
      label: name,
      color: ARCHETYPE_COLOR[name],
      count: count(name),
      hint: ARCHETYPE_BLURB[name],
    }));
  }
  if (lens === "mood") {
    return [
      { key: "positive", label: "HOPEFUL · PROUD", color: COOL_POLE, count: count("positive") },
      { key: "calm", label: "CALM", color: NEUTRAL, count: count("calm") },
      { key: "negative", label: "FRUSTRATED · ANXIOUS · GUILTY · ENVIOUS", color: WARM_POLE, count: count("negative") },
    ];
  }
  return [
    { key: "ahead", label: "AHEAD (UTILITY > 0)", color: COOL_POLE, count: count("ahead") },
    { key: "even", label: "EVEN", color: NEUTRAL, count: count("even") },
    { key: "behind", label: "BEHIND (UTILITY < 0)", color: WARM_POLE, count: count("behind") },
  ];
}

export const DIVERGING = { cool: COOL_POLE, neutral: NEUTRAL, warm: WARM_POLE };
