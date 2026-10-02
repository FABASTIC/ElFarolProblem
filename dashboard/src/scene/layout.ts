import type { AgentState, Frame } from "../model";

export type Zone = "home" | "bar" | "spill";

export interface Placement {
  x: number;
  z: number;
  zone: Zone;
}

export type Placements = Map<number, Placement>;

const SPACING = 0.62;
const homeCache = new Map<string, Map<number, { x: number; z: number }>>();

function hash(n: number, salt: number): number {
  const v = Math.sin(n * 127.1 + salt * 311.7) * 43758.5453;
  return v - Math.floor(v);
}

function homeSlots(ids: number[], gridSize: number, barMin: number, barMax: number): Map<number, { x: number; z: number }> {
  const key = `${gridSize}:${barMin}:${barMax}:${ids.join(",")}`;
  const cached = homeCache.get(key);
  if (cached) return cached;
  const half = gridSize / 2;
  const center = (barMin + barMax) / 2;
  const roads = new Set([Math.floor(center) - 1, Math.floor(center)]);
  const candidates: { x: number; y: number; r: number }[] = [];
  for (let x = 1; x < gridSize - 1; x += 1) {
    for (let y = 1; y < gridSize - 1; y += 1) {
      if (x >= barMin - 3 && x < barMax + 3 && y >= barMin - 3 && y < barMax + 3) continue;
      if (roads.has(x) || roads.has(y)) continue;
      candidates.push({ x, y, r: hash(x * 977 + y, gridSize) });
    }
  }
  candidates.sort((a, b) => a.r - b.r);
  const picks: { x: number; y: number }[] = [];
  let spacing = Math.max(1, Math.sqrt(Math.max(1, candidates.length) / Math.max(1, ids.length)) * 0.92);
  while (picks.length < ids.length && spacing >= 0.5) {
    for (const cell of candidates) {
      if (picks.length >= ids.length) break;
      if (picks.some((p) => p.x === cell.x && p.y === cell.y)) continue;
      if (picks.every((p) => Math.hypot(p.x - cell.x, p.y - cell.y) >= spacing)) picks.push(cell);
    }
    spacing *= 0.8;
  }
  let fill = 0;
  while (picks.length < ids.length) {
    picks.push(candidates[fill % Math.max(1, candidates.length)] ?? { x: fill % gridSize, y: Math.floor(fill / gridSize) % gridSize });
    fill += 1;
  }
  const sorted = [...ids].sort((a, b) => a - b);
  const result = new Map<number, { x: number; z: number }>();
  sorted.forEach((id, i) => {
    const cell = picks[i];
    result.set(id, {
      x: cell.x - half + 0.5 + (hash(id, 3) - 0.5) * 0.5,
      z: cell.y - half + 0.5 + (hash(id, 5) - 0.5) * 0.5,
    });
  });
  if (homeCache.size > 24) homeCache.clear();
  homeCache.set(key, result);
  return result;
}

function barSlots(x0: number, x1: number): { x: number; z: number }[] {
  const inner = 0.42;
  const span = x1 - x0 - inner * 2;
  const count = Math.max(1, Math.floor(span / SPACING) + 1);
  const step = count > 1 ? span / (count - 1) : 0;
  const slots: { x: number; z: number }[] = [];
  for (let c = 0; c < count; c += 1) {
    for (let r = 0; r < count; r += 1) slots.push({ x: x0 + inner + c * step, z: x0 + inner + r * step });
  }
  return slots;
}

function spillSlot(k: number, x0: number, x1: number): { x: number; z: number } {
  const ring = Math.floor(k / 28);
  const pad = 0.8 + ring * 0.6;
  const lo = x0 - pad;
  const hi = x1 + pad;
  const side = hi - lo;
  const t = ((k % 28) / 28 + ring * 0.137) % 1;
  const d = t * side * 4;
  if (d < side) return { x: lo + d, z: lo };
  if (d < side * 2) return { x: hi, z: lo + (d - side) };
  if (d < side * 3) return { x: hi - (d - side * 2), z: hi };
  return { x: lo, z: hi - (d - side * 3) };
}

export function layoutTown(frame: Frame | null, gridSize: number, barMin: number, barMax: number): Placements {
  const placements: Placements = new Map();
  const agents = frame?.agents ?? [];
  if (!agents.length) return placements;
  const homes = homeSlots(
    agents.map((a) => a.id),
    gridSize,
    barMin,
    barMax,
  );
  const half = gridSize / 2;
  const x0 = barMin - half;
  const x1 = barMax - half;
  const slots = barSlots(x0, x1);
  const inside = frame?.preview ? [] : agents.filter((a) => a.inBar);
  const honest = inside.filter((a) => a.strategy !== "false_stay").sort((a, b) => a.id - b.id);
  const covert = inside.filter((a) => a.strategy === "false_stay").sort((a, b) => a.id - b.id);
  const seated = new Map<number, { x: number; z: number }>();
  const overflow: AgentState[] = [];
  const free = new Set(slots.map((_, i) => i));
  const span = x1 - x0;
  const groups = [
    { members: honest, cx: x0 + span * (covert.length ? 0.34 : 0.5), cz: x0 + span * 0.5 },
    { members: covert, cx: x0 + span * (honest.length ? 0.7 : 0.5), cz: x0 + span * 0.5 },
  ];
  const order = groups.map((g) =>
    slots
      .map((slot, i) => ({ i, d: Math.hypot(slot.x - g.cx, slot.z - g.cz) + hash(i, g.cx) * 0.25 }))
      .sort((a, b) => a.d - b.d)
      .map((entry) => entry.i),
  );
  const cursor = [0, 0];
  const longest = Math.max(honest.length, covert.length);
  for (let k = 0; k < longest; k += 1) {
    groups.forEach((group, g) => {
      const agent = group.members[k];
      if (!agent) return;
      while (cursor[g] < order[g].length && !free.has(order[g][cursor[g]])) cursor[g] += 1;
      const index = order[g][cursor[g]];
      if (index == null) {
        overflow.push(agent);
        return;
      }
      free.delete(index);
      const slot = slots[index];
      seated.set(agent.id, { x: slot.x + (hash(agent.id, 11) - 0.5) * 0.18, z: slot.z + (hash(agent.id, 13) - 0.5) * 0.18 });
    });
  }
  overflow.forEach((agent, k) => {
    const spot = spillSlot(k, x0, x1);
    placements.set(agent.id, { ...spot, zone: "spill" });
  });
  for (const agent of agents) {
    if (placements.has(agent.id)) continue;
    const seat = seated.get(agent.id);
    if (seat) {
      placements.set(agent.id, { ...seat, zone: "bar" });
      continue;
    }
    const home = homes.get(agent.id) ?? { x: agent.x - half + 0.5, z: agent.y - half + 0.5 };
    placements.set(agent.id, { ...home, zone: "home" });
  }
  return placements;
}
