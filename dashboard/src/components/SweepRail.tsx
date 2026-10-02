import { useEffect, useRef, useState } from "react";
import { clamp, fmt, fmtClock, fmtDuration, fmtGiB, fmtPct, isNum } from "../format";
import { conditionColor, conditionLabel, type Frame, type TrialEntry } from "../model";
import type { LiveState } from "../types";

interface SweepRailProps {
  live: LiveState | null;
  liveRunning: boolean;
  trials: TrialEntry[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
  follow: boolean;
  onFollow: (value: boolean) => void;
  frame: Frame | null;
}

const STATUS_TONE: Record<string, string> = {
  complete: "good",
  resumed: "good",
  running: "live",
  failed: "fail",
  pending: "idle",
};

function VramMeter({ live }: { live: LiveState | null }) {
  const vram = live?.vram;
  if (!vram || !isNum(vram.total_mib) || vram.total_mib <= 0) {
    return <p className="empty">NO VRAM TELEMETRY</p>;
  }
  const used = clamp(vram.used_mib / vram.total_mib, 0, 1);
  const peak = isNum(vram.peak_used_mib) ? clamp(vram.peak_used_mib / vram.total_mib, 0, 1) : null;
  const base = isNum(vram.baseline_used_mib) ? clamp(vram.baseline_used_mib / vram.total_mib, 0, 1) : null;
  const tone = used > 0.97 ? "fail" : used > 0.93 ? "warn" : "ok";
  return (
    <div className="vram">
      <div className="vram__figures">
        <span className="vram__value">{fmtGiB(vram.used_mib)}</span>
        <span className="vram__total">/ {fmtGiB(vram.total_mib)} GiB</span>
        <span className="vram__peak">PEAK {fmtGiB(vram.peak_used_mib)}</span>
      </div>
      <div
        className="meter"
        data-tone={tone}
        role="meter"
        aria-label="GPU memory in use"
        aria-valuemin={0}
        aria-valuemax={vram.total_mib}
        aria-valuenow={vram.used_mib}
      >
        <div className="meter__fill" style={{ width: `${used * 100}%` }} />
        {base != null && <div className="meter__mark meter__mark--base" style={{ left: `${base * 100}%` }} title="Baseline before any engine" />}
        {peak != null && <div className="meter__mark meter__mark--peak" style={{ left: `${peak * 100}%` }} title="Peak observed" />}
      </div>
      <div className="vram__legend">
        <span>
          <i className="swatch swatch--base" /> BASELINE {fmtGiB(vram.baseline_used_mib)}
        </span>
        <span>
          <i className="swatch swatch--peak" /> PEAK
        </span>
        <span>FREE {fmtGiB(vram.free_mib)}</span>
      </div>
    </div>
  );
}

export default function SweepRail({ live, liveRunning, trials, selectedKey, onSelect, follow, onFollow, frame }: SweepRailProps) {
  const [tab, setTab] = useState<"minds" | "signals" | "armor">("minds");
  const listSection = useRef<HTMLElement>(null);

  useEffect(() => {
    const container = listSection.current;
    const item = container?.querySelector<HTMLElement>('[aria-selected="true"]');
    if (!container || !item) return;
    const top = item.offsetTop - container.offsetTop;
    const bottom = top + item.offsetHeight;
    const behavior: ScrollBehavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
    if (top < container.scrollTop) container.scrollTo({ top: Math.max(0, top - 36), behavior });
    else if (bottom > container.scrollTop + container.clientHeight) container.scrollTo({ top: bottom - container.clientHeight + 8, behavior });
  }, [selectedKey, trials.length]);

  const done = trials.filter((t) => t.status === "complete" || t.status === "resumed").length;
  const total = live?.sweep.trial_count ?? trials.length;
  const current = live?.current;
  const epochProgress = current && isNum(current.epoch) && current.epochs ? (current.epoch + 1) / current.epochs : null;
  const selected = trials.find((t) => t.key === selectedKey) ?? null;
  const events = [...(live?.events ?? [])].reverse();
  const signals = frame?.broadcasts ?? [];
  const thoughts = frame?.thoughts ?? [];
  const liars = thoughts.filter((t) => t.lied).length;

  return (
    <aside className="panel rail rail--left" style={{ ["--stagger" as string]: 1 }} aria-label="Sweep control">
      <section className="section">
        <h2 className="section__title">
          <strong>01 // SWEEP</strong>
          <span>
            {done}/{total || "—"} TRIALS
          </span>
        </h2>
        <div className="progress" aria-hidden="true">
          {Array.from({ length: Math.max(total, trials.length) }, (_, i) => {
            const trial = trials[i];
            return (
              <span
                key={i}
                className="progress__cell"
                data-tone={trial ? STATUS_TONE[trial.status] ?? "idle" : "idle"}
                style={{ ["--key" as string]: trial ? conditionColor(trial.condition) : "var(--hairline-strong)" }}
              />
            );
          })}
        </div>
        {liveRunning && current ? (
          <div className="current">
            <div className="kv">
              <span>ACTIVE</span>
              <strong>{current.trial}</strong>
            </div>
            <div className="kv">
              <span>PHASE</span>
              <strong>{current.phase.replace("_", " ").toUpperCase()}</strong>
            </div>
            <div className="kv">
              <span>EPOCH</span>
              <strong>{isNum(current.epoch) ? `${current.epoch + 1} / ${current.epochs}` : "—"}</strong>
            </div>
            <div className="bar" aria-hidden="true">
              <div className="bar__fill" style={{ width: `${(epochProgress ?? 0) * 100}%` }} />
            </div>
          </div>
        ) : (
          <p className="empty">{live ? `SWEEP ${live.status.toUpperCase()}` : "NO LIVE SWEEP // python3 experiment.py"}</p>
        )}
      </section>

      <section className="section">
        <h2 className="section__title">
          <strong>02 // VRAM</strong>
          <span>RTX MEMORY</span>
        </h2>
        <VramMeter live={live} />
      </section>

      <section className="section section--grow" ref={listSection}>
        <h2 className="section__title">
          <strong>03 // TRIALS</strong>
          <button
            type="button"
            className="btn btn--tiny"
            aria-pressed={follow}
            onClick={() => onFollow(!follow)}
            disabled={!liveRunning}
            title="Lock the arena to the live trial"
          >
            FOLLOW LIVE
          </button>
        </h2>
        {trials.length === 0 ? (
          <p className="empty">NO TRIALS ON DISK</p>
        ) : (
          <ul className="trials" role="listbox" aria-label="Trials">
            {trials.map((trial) => (
              <li key={trial.key} role="presentation">
                <button
                  type="button"
                  role="option"
                  aria-selected={trial.key === selectedKey}
                  className="trial"
                  onClick={() => onSelect(trial.key)}
                  style={{ ["--key" as string]: conditionColor(trial.condition) }}
                >
                  <span className="trial__key" aria-hidden="true" />
                  <span className="trial__name">
                    <strong>{conditionLabel(trial.condition)}</strong>
                    <span>SEED {trial.seed || "—"}</span>
                  </span>
                  <span className="trial__meta">
                    <span className="status" data-tone={trial.isLive ? "live" : STATUS_TONE[trial.status] ?? "idle"}>
                      <i aria-hidden="true" />
                      {trial.isLive ? "LIVE" : trial.status.toUpperCase()}
                    </span>
                    <span className="trial__fallback">{trial.fallback != null ? `FB ${fmtPct(trial.fallback, 0)}` : ""}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
        {selected && (
          <dl className="runtime">
            <div>
              <dt>ATTEMPTS</dt>
              <dd>{selected.attempts ?? "—"}</dd>
            </div>
            <div>
              <dt>ENGINE</dt>
              <dd>{isNum(selected.buildTime) ? `${fmt(selected.buildTime, 1)}s` : "—"}</dd>
            </div>
            <div>
              <dt>WALL</dt>
              <dd>{fmtDuration(selected.wallTime)}</dd>
            </div>
            <div>
              <dt>PEAK</dt>
              <dd>{isNum(selected.peakVram) ? `${fmtGiB(selected.peakVram)}G` : "—"}</dd>
            </div>
            <div>
              <dt>PARSE FB</dt>
              <dd>{fmtPct(selected.fallback, 1)}</dd>
            </div>
            <div>
              <dt>PROMPT</dt>
              <dd>{selected.encoding ? selected.encoding.replace("_", " ").toUpperCase() : "—"}</dd>
            </div>
            {selected.error && (
              <div className="runtime__error">
                <dt>ERROR</dt>
                <dd>{selected.error}</dd>
              </div>
            )}
          </dl>
        )}
      </section>

      <section className="section section--feed">
        <div className="tabs" role="tablist" aria-label="Feeds">
          <button type="button" role="tab" aria-selected={tab === "minds"} className="tab" onClick={() => setTab("minds")}>
            MINDS <span className="tab__count">{liars ? `${liars} LIE` : thoughts.length}</span>
          </button>
          <button type="button" role="tab" aria-selected={tab === "signals"} className="tab" onClick={() => setTab("signals")}>
            SIGNALS <span className="tab__count">{signals.length}</span>
          </button>
          <button type="button" role="tab" aria-selected={tab === "armor"} className="tab" onClick={() => setTab("armor")}>
            ARMOR LOG <span className="tab__count">{events.length}</span>
          </button>
        </div>
        <ol className="feedlist" role="tabpanel" aria-live="polite">
          {tab === "minds" &&
            (thoughts.length ? (
              thoughts.map((t) => (
                <li key={`${t.agentId}-${t.note}`} className="thought" data-lie={t.lied}>
                  <span className="thought__head">
                    <strong>A{String(t.agentId).padStart(2, "0")}</strong>
                    <span>{t.archetype}</span>
                    <span>
                      SAID {t.stated.toUpperCase()} · {t.inBar ? "AT BAR" : "HOME"}
                    </span>
                    {t.lied && <span className="thought__tag">LIE</span>}
                  </span>
                  <span className="thought__note">“{t.note}”</span>
                </li>
              ))
            ) : (
              <li className="empty">NO PRIVATE THOUGHTS · CLONE AGENTS OR NO TRACE</li>
            ))}
          {tab === "signals" &&
            (signals.length ? (
              signals.map((signal) => (
                <li key={`${signal.agentId}-${signal.text}`}>
                  <span className="feedlist__who">A{String(signal.agentId).padStart(2, "0")}</span>
                  <span className="feedlist__text">{signal.text}</span>
                </li>
              ))
            ) : (
              <li className="empty">NO BROADCASTS THIS EPOCH</li>
            ))}
          {tab === "armor" &&
            (events.length ? (
              events.map((event, i) => (
                <li key={`${event.t}-${i}`}>
                  <span className="feedlist__who">{fmtClock(event.t)}</span>
                  <span className="feedlist__text">{event.message}</span>
                </li>
              ))
            ) : (
              <li className="empty">NO ARMOR EVENTS</li>
            ))}
        </ol>
      </section>
    </aside>
  );
}
