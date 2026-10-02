export function isNum(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function fmt(value: number | null | undefined, digits = 3): string {
  return isNum(value) ? value.toFixed(digits) : "—";
}

export function fmtSigned(value: number | null | undefined, digits = 3): string {
  if (!isNum(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "±";
  return `${sign}${Math.abs(value).toFixed(digits)}`;
}

export function fmtPct(value: number | null | undefined, digits = 1): string {
  return isNum(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

export function fmtP(value: number | null | undefined): string {
  if (!isNum(value)) return "—";
  return value < 0.001 ? "<0.001" : value.toFixed(3);
}

export function fmtDuration(seconds: number | null | undefined): string {
  if (!isNum(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

export function fmtGiB(mib: number | null | undefined): string {
  return isNum(mib) ? `${(mib / 1024).toFixed(2)}` : "—";
}

export function fmtClock(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000);
  return date.toLocaleTimeString([], { hour12: false });
}

export function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

export function niceTicks(min: number, max: number, count = 4): number[] {
  if (!isNum(min) || !isNum(max) || max <= min) return [min];
  const raw = (max - min) / Math.max(1, count);
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? raw;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-6; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

export function tickLabel(value: number, span: number): string {
  if (span >= 20) return value.toFixed(0);
  if (span >= 2) return value.toFixed(1);
  if (span >= 0.2) return value.toFixed(2);
  return value.toFixed(3);
}
