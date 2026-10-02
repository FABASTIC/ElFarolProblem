import { useEffect, useRef, useState } from "react";
import { fmt, fmtClock, fmtDuration, isNum } from "../format";
import { conditionColor, conditionLabel, type Frame, type TrialEntry } from "../model";
import type { LiveState } from "../types";
import ScrollProgress from "./ScrollProgress";

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

export default function SweepRail({ live, liveRunning, trials, selectedKey, onSelect, follow, onFollow, frame }: SweepRailProps) {
  const [tab, setTab] = useState<"minds" | "signals" | "armor">("minds");
  const listSection = useRef<HTMLElement>(null);
  const feed = useRef<HTMLOListElement>(null);

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
          <p className="empty">{live ? `SWEEP ${live.status.toUpperCase()}` : `${done} TOWNS RECORDED`}</p>
        )}
      </section>

      <section className="section section--grow" ref={listSection}>
        <h2 className="section__title">
          <strong>02 // TRIALS</strong>
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
          <p className="empty">NO TRIALS RECORDED</p>
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
              <dt>BRAINS UP</dt>
              <dd>{isNum(selected.buildTime) ? `${fmt(selected.buildTime, 1)}s` : "—"}</dd>
            </div>
            <div>
              <dt>WALL</dt>
              <dd>{fmtDuration(selected.wallTime)}</dd>
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
            RUN LOG <span className="tab__count">{events.length}</span>
          </button>
        </div>
        <div className="scrollwrap">
          <ol className="feedlist" role="tabpanel" aria-live="polite" ref={feed}>
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
                <li className="empty">NO PRIVATE THOUGHTS FOR THIS NIGHT</li>
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
                <li className="empty">NO RUN EVENTS</li>
              ))}
          </ol>
          <ScrollProgress target={feed} watch={`${tab}:${thoughts.length}:${signals.length}:${events.length}`} />
        </div>
      </section>
    </aside>
  );
}
