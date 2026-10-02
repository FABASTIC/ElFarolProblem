import { useState } from "react";
import { fmtPct } from "../format";
import { STRATEGIES, STRATEGY_COLOR, STRATEGY_SHORT, type TrialTrace } from "../model";
import { useElementSize } from "./useElementSize";

interface StrategyStripProps {
  trace: TrialTrace | null;
  cursor: number | null;
  condition: string | null;
  onScrub?: (epoch: number) => void;
}

const LABEL_W = 58;
const ROW_H = 24;
const GAP = 2;

function ramp(share: number): string {
  const t = Math.max(0, Math.min(1, share));
  const lo = [24, 24, 28];
  const hi = [236, 236, 240];
  const c = lo.map((l, i) => Math.round(l + (hi[i] - l) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export default function StrategyStrip({ trace, cursor, condition, onScrub }: StrategyStripProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);
  const epochs = trace?.epochs ?? [];
  const width = Math.max(160, size.width);
  const cellW = epochs.length ? (width - LABEL_W) / epochs.length : 0;
  const height = STRATEGIES.length * (ROW_H + GAP);

  return (
    <figure className="chart strip" ref={ref} data-condition={condition ?? undefined} aria-label="Strategy share strip">
      <figcaption className="chart__head">
        <span className="chart__index">E</span>
        <span className="chart__title">STRATEGY SHARE // THIS TRIAL</span>
        <span className="chart__unit">0 → 1</span>
      </figcaption>
      {epochs.length === 0 ? (
        <p className="empty">NO STRATEGY TRACE</p>
      ) : (
        <svg width={width} height={height} role="img" aria-label="Share of each strategy per epoch">
          {STRATEGIES.map((s, row) => (
            <g key={s} transform={`translate(0, ${row * (ROW_H + GAP)})`}>
              <rect x={0} y={ROW_H / 2 - 4} width={8} height={8} fill={STRATEGY_COLOR[s]} />
              <text x={14} y={ROW_H / 2} className="chart__tick" dominantBaseline="middle">
                {STRATEGY_SHORT[s]}
              </text>
              {trace!.shares[s].map((share, col) => (
                <rect
                  key={col}
                  x={LABEL_W + col * cellW}
                  y={0}
                  width={Math.max(0.5, cellW - GAP)}
                  height={ROW_H}
                  fill={ramp(share)}
                  className="strip__cell"
                  onPointerEnter={() => setHover({ row, col })}
                  onPointerLeave={() => setHover(null)}
                  onClick={() => onScrub?.(epochs[col])}
                />
              ))}
            </g>
          ))}
          {cursor != null && epochs.includes(cursor) && (
            <rect
              x={LABEL_W + epochs.indexOf(cursor) * cellW - 1}
              y={-1}
              width={cellW + 0}
              height={height}
              className="strip__cursor"
            />
          )}
        </svg>
      )}
      {hover && trace && (
        <div
          className="tooltip"
          style={{ left: LABEL_W + hover.col * cellW + cellW / 2, top: hover.row * (ROW_H + GAP) + 30 }}
        >
          <div className="tooltip__head">EPOCH {epochs[hover.col]}</div>
          <div className="tooltip__row">
            <span className="tooltip__key" style={{ background: STRATEGY_COLOR[STRATEGIES[hover.row]] }} />
            <span className="tooltip__value">{fmtPct(trace.shares[STRATEGIES[hover.row]][hover.col], 1)}</span>
            <span className="tooltip__label">{STRATEGY_SHORT[STRATEGIES[hover.row]]}</span>
          </div>
        </div>
      )}
    </figure>
  );
}
