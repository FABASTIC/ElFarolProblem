import { useState, type MouseEvent } from "react";
import { Segmented } from "./Kinetic";
import type { WorldMode } from "./TopBar";
import { useElementSize } from "./useElementSize";

interface NightTimelineProps {
  total: number;
  attendance: (number | null)[];
  available: Set<number>;
  cursorEpoch: number | null;
  threshold: number;
  members: number;
  mode: WorldMode;
  playing: boolean;
  onPlaying: (value: boolean) => void;
  speed: number;
  onSpeed: (value: number) => void;
  onSeek: (epoch: number) => void;
  onStep: (delta: number) => void;
  atEdge: boolean;
  onEdge: () => void;
}

const SPEEDS = [1, 2, 4, 8];
const H = 46;
const PAD = 4;

export default function NightTimeline({
  total,
  attendance,
  available,
  cursorEpoch,
  threshold,
  members,
  mode,
  playing,
  onPlaying,
  speed,
  onSpeed,
  onSeek,
  onStep,
  atEdge,
  onEdge,
}: NightTimelineProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const nights = Math.max(1, total);
  const width = Math.max(120, size.width);
  const slot = width / nights;
  const gap = slot > 5 ? 1.5 : slot > 2.5 ? 0.6 : 0;
  const scale = Math.max(members, threshold + 10, 1);
  const y = (v: number) => H - PAD - (v / scale) * (H - PAD * 2);
  const playable = mode === "replay" || (mode === "live" && available.size > 1);
  const staged = mode === "setup" || mode === "boot";

  const nightAt = (event: MouseEvent<SVGRectElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return Math.max(0, Math.min(nights - 1, Math.floor(((event.clientX - rect.left) / rect.width) * nights)));
  };

  const hovered = hover != null ? attendance[hover] : null;

  return (
    <section className="hud-card nights" aria-label="Nights timeline">
      <div className="nights__controls">
        <button type="button" className="btn" onClick={() => onStep(-1)} disabled={!playable} aria-label="Previous night">
          ◀
        </button>
        <button
          type="button"
          className="btn btn--play"
          onClick={() => onPlaying(!playing)}
          disabled={mode !== "replay"}
          aria-pressed={playing}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? "❚❚" : "▶"}
        </button>
        <button type="button" className="btn" onClick={() => onStep(1)} disabled={!playable} aria-label="Next night">
          ▶
        </button>
        {mode === "live" ? (
          <button type="button" className="btn btn--live" aria-pressed={atEdge} onClick={onEdge}>
            ● LIVE
          </button>
        ) : (
          <Segmented label="Playback speed" className="speeds" options={SPEEDS.map((s) => ({ id: s, label: `${s}×` }))} value={speed} onChange={onSpeed} disabled={mode !== "replay"} />
        )}
      </div>
      <div className="nights__strip" ref={ref}>
        <svg width={width} height={H} role="img" aria-label="Bar attendance per night against the comfort line">
          {Array.from({ length: nights }, (_, i) => {
            const v = attendance[i];
            const x = i * slot + gap / 2;
            const w = Math.max(0.6, slot - gap);
            if (v == null) {
              return <rect key={i} x={x} y={H - PAD - 2} width={w} height={2} className="nights__empty" />;
            }
            return (
              <rect
                key={i}
                x={x}
                y={y(v)}
                width={w}
                height={Math.max(1, H - PAD - y(v))}
                rx={Math.min(2, w / 3)}
                className="nights__bar"
                data-tone={v > threshold ? "fail" : "good"}
                data-seekable={available.has(i)}
                opacity={cursorEpoch != null && i > cursorEpoch ? 0.35 : 1}
              />
            );
          })}
          <line x1={0} x2={width} y1={y(threshold)} y2={y(threshold)} className="nights__threshold" />
          <text x={width - 4} y={y(threshold) - 3} textAnchor="end" className="nights__label">
            COMFORT LINE {threshold}
          </text>
          {cursorEpoch != null && cursorEpoch >= 0 && (
            <rect x={cursorEpoch * slot - 1} y={0} width={Math.max(2, slot) + 2} height={H} className="nights__cursor" />
          )}
          <rect
            x={0}
            y={0}
            width={width}
            height={H}
            fill="transparent"
            style={{ cursor: playable ? "pointer" : "default" }}
            onPointerMove={(event) => setHover(nightAt(event))}
            onPointerLeave={() => setHover(null)}
            onClick={(event) => {
              const night = nightAt(event);
              if (available.has(night)) onSeek(night);
            }}
          />
        </svg>
        {hover != null && (
          <div className="tooltip nights__tip" style={{ left: Math.min(width - 160, Math.max(0, hover * slot)) }}>
            <div className="tooltip__head">NIGHT {hover + 1}</div>
            <div className="tooltip__row">
              <span className="tooltip__key" style={{ background: hovered == null ? "var(--muted)" : hovered > threshold ? "var(--critical)" : "var(--good)" }} />
              <span className="tooltip__value">{hovered ?? "—"}</span>
              <span className="tooltip__label">
                {hovered == null ? "not yet played" : hovered > threshold ? "in the bar · overcrowded" : "in the bar · comfortable"}
              </span>
            </div>
            {hovered != null && !available.has(hover) && <div className="tooltip__head">positions not loaded for this night</div>}
          </div>
        )}
      </div>
      <div className="nights__legend">
        {staged ? (
          <span>{nights} nights staged</span>
        ) : (
          <>
            <span>
              <i data-tone="good" aria-hidden="true" />✓ COMFORTABLE
            </span>
            <span>
              <i data-tone="fail" aria-hidden="true" />▲ OVERCROWDED
            </span>
          </>
        )}
      </div>
    </section>
  );
}
