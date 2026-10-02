import { useMemo, useState, type KeyboardEvent, type PointerEvent } from "react";
import { fmt, isNum, niceTicks, tickLabel } from "../format";
import { useElementSize } from "./useElementSize";

export interface ChartSeries {
  id: string;
  label: string;
  color: string;
  values: (number | null)[];
  band?: { lo: (number | null)[]; hi: (number | null)[] };
  width?: number;
  opacity?: number;
}

export interface ChartReference {
  y: number;
  label: string;
  color: string;
  dashed?: boolean;
}

interface LineChartProps {
  index: string;
  title: string;
  unit?: string;
  x: number[];
  series: ChartSeries[];
  references?: ChartReference[];
  cursor?: number | null;
  yMin?: number;
  yMax?: number;
  digits?: number;
  height?: number;
  onScrub?: (x: number) => void;
}

const PAD = { top: 10, right: 10, bottom: 18, left: 34 };

function pathFor(xs: number[], ys: (number | null)[], sx: (v: number) => number, sy: (v: number) => number): string {
  let d = "";
  let pen = false;
  ys.forEach((y, i) => {
    if (!isNum(y) || !isNum(xs[i])) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${sx(xs[i]).toFixed(2)},${sy(y).toFixed(2)}`;
    pen = true;
  });
  return d;
}

function bandFor(xs: number[], lo: (number | null)[], hi: (number | null)[], sx: (v: number) => number, sy: (v: number) => number): string {
  const runs: string[] = [];
  let upper: string[] = [];
  let lower: string[] = [];
  const flush = () => {
    if (upper.length > 1) runs.push(`M${upper.join("L")}L${lower.reverse().join("L")}Z`);
    upper = [];
    lower = [];
  };
  xs.forEach((x, i) => {
    const a = lo[i];
    const b = hi[i];
    if (!isNum(a) || !isNum(b) || !isNum(x)) {
      flush();
      return;
    }
    upper.push(`${sx(x).toFixed(2)},${sy(b).toFixed(2)}`);
    lower.push(`${sx(x).toFixed(2)},${sy(a).toFixed(2)}`);
  });
  flush();
  return runs.join("");
}

export default function LineChart({
  index,
  title,
  unit,
  x,
  series,
  references = [],
  cursor = null,
  yMin,
  yMax,
  digits = 3,
  height = 132,
  onScrub,
}: LineChartProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const width = Math.max(120, size.width);
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;

  const domain = useMemo(() => {
    const values: number[] = [];
    series.forEach((s) => {
      s.values.forEach((v) => isNum(v) && values.push(v));
      s.band?.lo.forEach((v) => isNum(v) && values.push(v));
      s.band?.hi.forEach((v) => isNum(v) && values.push(v));
    });
    references.forEach((r) => values.push(r.y));
    const lo = yMin ?? (values.length ? Math.min(...values) : 0);
    let hi = yMax ?? (values.length ? Math.max(...values) : 1);
    if (hi <= lo) hi = lo + 1;
    const pad = yMax == null ? (hi - lo) * 0.08 : 0;
    return [lo, hi + pad] as const;
  }, [series, references, yMin, yMax]);

  const xLo = x.length ? x[0] : 0;
  const xHi = x.length > 1 ? x[x.length - 1] : xLo + 1;
  const sx = (v: number) => PAD.left + ((v - xLo) / (xHi - xLo || 1)) * plotW;
  const sy = (v: number) => PAD.top + plotH - ((v - domain[0]) / (domain[1] - domain[0])) * plotH;
  const ticks = niceTicks(domain[0], domain[1], 3);
  const integerTicks = niceTicks(xLo, xHi, 4).filter((t) => t >= xLo && t <= xHi && Number.isInteger(t));
  const xTicks = integerTicks.length ? integerTicks : [xLo, xHi].filter((t, i, all) => all.indexOf(t) === i);
  const span = domain[1] - domain[0];

  const nearest = (clientX: number, rect: DOMRect) => {
    if (!x.length) return null;
    const value = xLo + ((clientX - rect.left - PAD.left) / plotW) * (xHi - xLo);
    let best = 0;
    x.forEach((xv, i) => {
      if (Math.abs(xv - value) < Math.abs(x[best] - value)) best = i;
    });
    return best;
  };

  const onMove = (event: PointerEvent<SVGRectElement>) => {
    const rect = (event.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    setHover(nearest(event.clientX, rect));
  };

  const onClick = (event: PointerEvent<SVGRectElement>) => {
    const rect = (event.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const i = nearest(event.clientX, rect);
    if (i != null && onScrub) onScrub(x[i]);
  };

  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!x.length) return;
    const current = hover ?? (cursor != null ? Math.max(0, x.indexOf(cursor)) : 0);
    if (event.key === "ArrowRight") setHover(Math.min(x.length - 1, current + 1));
    else if (event.key === "ArrowLeft") setHover(Math.max(0, current - 1));
    else if (event.key === "Enter" && hover != null && onScrub) onScrub(x[hover]);
    else return;
    event.preventDefault();
  };

  const hoverX = hover != null ? x[hover] : null;
  const tooltipLeft = hoverX != null ? sx(hoverX) : 0;
  const flip = tooltipLeft > width * 0.62;
  const hasData = x.length > 0 && series.some((s) => s.values.some(isNum));

  if (!hasData) {
    return (
      <figure className="chart" ref={ref} aria-label={`${title} chart`}>
        <figcaption className="chart__head">
          <span className="chart__index">{index}</span>
          <span className="chart__title">{title}</span>
          {unit && <span className="chart__unit">{unit}</span>}
        </figcaption>
        <p className="empty chart__empty" style={{ height }}>
          NO TRACE
        </p>
      </figure>
    );
  }

  return (
    <figure className="chart" ref={ref} tabIndex={0} onKeyDown={onKey} onBlur={() => setHover(null)} aria-label={`${title} chart`}>
      <figcaption className="chart__head">
        <span className="chart__index">{index}</span>
        <span className="chart__title">{title}</span>
        {unit && <span className="chart__unit">{unit}</span>}
      </figcaption>
      <svg width={width} height={height} role="img" aria-label={title}>
        {ticks.map((t) => (
          <g key={`y${t}`}>
            <line x1={PAD.left} x2={width - PAD.right} y1={sy(t)} y2={sy(t)} className="chart__grid" />
            <text x={PAD.left - 6} y={sy(t)} className="chart__tick" textAnchor="end" dominantBaseline="middle">
              {tickLabel(t, span)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={sx(t)} y={height - 4} className="chart__tick" textAnchor="middle">
            {t.toFixed(0)}
          </text>
        ))}
        <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} className="chart__frame" />
        {references.map((r) => (
          <g key={`r${r.label}`}>
            <line
              x1={PAD.left}
              x2={width - PAD.right}
              y1={sy(r.y)}
              y2={sy(r.y)}
              stroke={r.color}
              strokeWidth={1}
              strokeDasharray={r.dashed ? "4 3" : undefined}
              opacity={0.8}
            />
            <text x={width - PAD.right - 4} y={sy(r.y) - 3} className="chart__ref" textAnchor="end">
              {r.label}
            </text>
          </g>
        ))}
        {series.map((s) =>
          s.band ? <path key={`b${s.id}`} d={bandFor(x, s.band.lo, s.band.hi, sx, sy)} fill={s.color} opacity={0.1} /> : null,
        )}
        {series.map((s) => (
          <path
            key={`l${s.id}`}
            d={pathFor(x, s.values, sx, sy)}
            fill="none"
            stroke={s.color}
            strokeWidth={s.width ?? 2}
            strokeLinejoin="round"
            strokeLinecap="round"
            opacity={s.opacity ?? 1}
          />
        ))}
        {cursor != null && cursor >= xLo && cursor <= xHi && (
          <line x1={sx(cursor)} x2={sx(cursor)} y1={PAD.top} y2={PAD.top + plotH} className="chart__cursor" />
        )}
        {hoverX != null && (
          <g>
            <line x1={sx(hoverX)} x2={sx(hoverX)} y1={PAD.top} y2={PAD.top + plotH} className="chart__crosshair" />
            {series.map((s) =>
              isNum(s.values[hover as number]) ? (
                <circle
                  key={`d${s.id}`}
                  cx={sx(hoverX)}
                  cy={sy(s.values[hover as number] as number)}
                  r={4}
                  fill={s.color}
                  className="chart__dot"
                />
              ) : null,
            )}
          </g>
        )}
        <rect
          x={PAD.left}
          y={0}
          width={plotW}
          height={height}
          fill="transparent"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          onClick={onClick}
          style={{ cursor: onScrub ? "pointer" : "crosshair" }}
        />
      </svg>
      {hover != null && hoverX != null && (
        <div className={`tooltip${flip ? " tooltip--flip" : ""}`} style={{ left: tooltipLeft, top: PAD.top + 24 }}>
          <div className="tooltip__head">EPOCH {hoverX}</div>
          {series.map((s) => (
            <div className="tooltip__row" key={`t${s.id}`}>
              <span className="tooltip__key" style={{ background: s.color }} />
              <span className="tooltip__value">{fmt(s.values[hover] ?? null, digits)}</span>
              <span className="tooltip__label">{s.label}</span>
            </div>
          ))}
        </div>
      )}
    </figure>
  );
}
