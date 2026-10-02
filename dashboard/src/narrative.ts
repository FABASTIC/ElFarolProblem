import type { AgentState, Frame } from "./model";

export interface NoteFacts {
  townPct: number | null;
  forecast: number | null;
  qGo: number | null;
  qStay: number | null;
}

export interface Reasoning {
  headline: string;
  context: string | null;
  outcome: string | null;
  lied: boolean;
}

export interface NightSummary {
  night: number;
  members: number;
  attendance: number;
  threshold: number;
  crowded: boolean;
  control: boolean;
  honest: number;
  dishonest: number;
  honestGo: number;
  honestStay: number;
  bluff: number;
  covert: number;
  lieRate: number;
  lieDelta: number | null;
  utilHonest: number | null;
  utilLiar: number | null;
  verdict: "honesty" | "lying" | "even" | null;
  claimedPct: number | null;
  actualPct: number;
  headline: string;
  lines: string[];
}

const LIES = new Set(["false_go", "false_stay"]);

function pick<T>(id: number, options: T[]): T {
  return options[Math.abs(Math.floor(Math.sin(id * 91.7) * 1000)) % options.length];
}

function count(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`;
}

function signed(value: number): string {
  const rounded = Math.round(value * 100) / 100;
  return `${rounded > 0 ? "+" : rounded < 0 ? "−" : ""}${Math.abs(rounded).toFixed(2)}`;
}

function mean(values: number[]): number | null {
  const clean = values.filter((v) => Number.isFinite(v));
  return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : null;
}

export function parseNote(note: string | null | undefined): NoteFacts {
  const text = note ?? "";
  const town = /town says (\d+(?:\.\d+)?)% going/i.exec(text);
  const forecast = /forecast (\d+(?:\.\d+)?)/i.exec(text);
  const q = /Q go ([+-]?\d+(?:\.\d+)?) \/ stay ([+-]?\d+(?:\.\d+)?)/i.exec(text);
  return {
    townPct: town ? Number(town[1]) : null,
    forecast: forecast ? Number(forecast[1]) : null,
    qGo: q ? Number(q[1]) : null,
    qStay: q ? Number(q[2]) : null,
  };
}

export function narrate(agent: AgentState, threshold: number, crowded: boolean | null): Reasoning {
  const facts = parseNote(agent.mind?.note);
  const forecast = agent.mind?.forecast ?? facts.forecast;
  const expectCrowd = forecast != null ? forecast > threshold : null;
  const spoke = !!agent.broadcast;
  const lied = spoke && LIES.has(agent.strategy);
  let headline: string;
  if (!spoke) {
    if (agent.inBar) headline = expectCrowd ? pick(agent.id, ["Expecting a crowd, but I'm going anyway.", "Probably packed. Going regardless."]) : pick(agent.id, ["Should be quiet tonight. Heading over.", "My guess is there's room. I'm going."]);
    else headline = expectCrowd ? pick(agent.id, ["Too many will go tonight. Staying home.", "Feels like a packed night. I'll sit this out."]) : pick(agent.id, ["Not worth the risk tonight. Staying in.", "I'll stay home and see how it goes."]);
  } else if (agent.strategy === "false_go") {
    headline = pick(agent.id, ["Told them I'm going so they'd stay home. I'm not going.", "Said I'd be there to scare the crowd off. Staying in myself."]);
  } else if (agent.strategy === "false_stay") {
    headline = pick(agent.id, ["Said I'd stay home so fewer would show. Then I went.", "Told them I'm staying in to thin the crowd. Heading over anyway."]);
  } else if (agent.strategy === "honest_go") {
    headline = expectCrowd ? "Expecting a crowd, but I'm going, and I said so." : pick(agent.id, ["Looks like a quiet night. Going, and I told them the truth.", "Said I'm going, and I am. Should be room."]);
  } else {
    headline = expectCrowd ? "It'll be packed. Staying in, and I said as much." : pick(agent.id, ["Said I'd stay home, and I meant it.", "Told them I'm staying in. I am."]);
  }
  const context: string[] = [];
  if (spoke && facts.townPct != null) context.push(`Town says ${Math.round(facts.townPct)}% going`);
  if (forecast != null) context.push(`I expect ~${Math.round(forecast)} for ${threshold} seats`);
  let outcome: string | null = null;
  if (crowded != null) {
    if (agent.inBar) outcome = crowded ? "Backfired: the bar was packed." : "Paid off: there was room.";
    else outcome = crowded ? "Good call: the bar was packed." : "Missed out: the bar had room.";
  }
  return { headline, context: context.length ? context.join(" · ") : null, outcome, lied };
}

function tally(frame: Frame) {
  const agents = frame.agents;
  const n = agents.length;
  const control = !agents.some((a) => a.broadcast);
  const by = (key: string) => agents.filter((a) => a.strategy === key).length;
  const honestGo = by("honest_go");
  const honestStay = by("honest_stay");
  const bluff = control ? 0 : by("false_go");
  const covert = control ? 0 : by("false_stay");
  const dishonest = bluff + covert;
  return { n, control, honestGo, honestStay, bluff, covert, dishonest, lieRate: n ? dishonest / n : 0 };
}

export function summarizeNight(frame: Frame, previous: Frame | null, threshold: number): NightSummary {
  const t = tally(frame);
  const attendance = frame.agents.filter((a) => a.inBar).length;
  const crowded = attendance > threshold;
  const honestAgents = frame.agents.filter((a) => !LIES.has(a.strategy) || t.control);
  const liarAgents = t.control ? [] : frame.agents.filter((a) => LIES.has(a.strategy));
  const utilHonest = mean(honestAgents.map((a) => a.utility));
  const utilLiar = mean(liarAgents.map((a) => a.utility));
  let verdict: NightSummary["verdict"] = null;
  if (!t.control && utilHonest != null && utilLiar != null) verdict = utilLiar > utilHonest + 0.05 ? "lying" : utilHonest > utilLiar + 0.05 ? "honesty" : "even";
  const prev = previous && previous.agents.length ? tally(previous) : null;
  const lieDelta = prev && !t.control ? Math.round((t.lieRate - prev.lieRate) * 100) : null;
  const claimed = t.control ? null : frame.agents.filter((a) => (a.stated ?? "").toLowerCase() === "going").length;
  const claimedPct = claimed != null && t.n ? Math.round((claimed / t.n) * 100) : null;
  const actualPct = t.n ? Math.round((attendance / t.n) * 100) : 0;
  const honest = t.n - t.dishonest;

  const headline = crowded
    ? `The bar overflowed: ${attendance} people for ${threshold} seats.`
    : attendance === threshold
      ? `A full house: exactly ${threshold} people, not one too many.`
      : `A comfortable night: ${attendance} people, ${count(threshold - attendance, "seat", "seats")} to spare.`;

  const lines: string[] = [];
  if (t.control) {
    lines.push("This is a control town. Nobody can talk, so nobody can lie. Everyone guessed alone.");
    lines.push(`${count(attendance, "person", "people")} went out; ${count(t.n - attendance, "person", "people")} stayed home.`);
  } else {
    lines.push(`${honest} of ${t.n} kept their word. ${count(t.dishonest, "person", "people")} did not.`);
    if (t.bluff) lines.push(`${count(t.bluff, "person", "people")} bluffed: they announced they were going, then stayed home, hoping to scare others off.`);
    if (t.covert) lines.push(`${count(t.covert, "person", "people")} slipped in: they said they'd stay home, then showed up anyway.`);
    if (claimedPct != null) lines.push(`The town heard ${claimedPct}% would go. In the end ${actualPct}% did.`);
    if (verdict === "lying" && utilHonest != null && utilLiar != null) lines.push(`Lying paid tonight: liars averaged ${signed(utilLiar)} against ${signed(utilHonest)} for honest folk.`);
    else if (verdict === "honesty" && utilHonest != null && utilLiar != null) lines.push(`Honesty paid tonight: honest folk averaged ${signed(utilHonest)} against ${signed(utilLiar)} for liars.`);
    else if (verdict === "even") lines.push("Honest folk and liars came out about even.");
    if (lieDelta != null) lines.push(lieDelta >= 3 ? `Lying is up ${lieDelta} points on last night.` : lieDelta <= -3 ? `Lying is down ${Math.abs(lieDelta)} points on last night.` : "Lying held steady on last night.");
  }

  return {
    night: frame.epoch + 1,
    members: t.n,
    attendance,
    threshold,
    crowded,
    control: t.control,
    honest,
    dishonest: t.dishonest,
    honestGo: t.honestGo,
    honestStay: t.honestStay,
    bluff: t.bluff,
    covert: t.covert,
    lieRate: t.lieRate,
    lieDelta,
    utilHonest,
    utilLiar,
    verdict,
    claimedPct,
    actualPct,
    headline,
    lines,
  };
}
