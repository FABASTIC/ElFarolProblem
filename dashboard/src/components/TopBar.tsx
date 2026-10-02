import { useEffect, useState } from "react";
import { fmtDuration } from "../format";
import { conditionColor, conditionLabel, type TrialEntry } from "../model";
import type { FeedState } from "../telemetry";

export interface FeedHealth {
  name: string;
  file: string;
  state: FeedState;
  receivedAt: number | null;
}

export type View = "world" | "lab";
export type WorldMode = "setup" | "boot" | "live" | "replay" | "empty";

interface TopBarProps {
  view: View;
  onView: (view: View) => void;
  model: string | null;
  statusLabel: string;
  statusTone: string;
  night: number | null;
  nights: number | null;
  trial: TrialEntry | null;
  mode: WorldMode;
  elapsed: number | null;
  eta: number | null;
  feeds: FeedHealth[];
  canStop: boolean;
  stopping: boolean;
  onStop: () => void;
  canNew: boolean;
  onNew: () => void;
}

const MODE_LABEL: Record<WorldMode, string> = {
  setup: "STAGED",
  boot: "BOOTING",
  live: "LIVE",
  replay: "REPLAY",
  empty: "IDLE",
};

function useNow(intervalMs: number) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}

function feedTone(feed: FeedHealth, now: number): string {
  if (feed.state === "error") return "fail";
  if (feed.state === "missing" || feed.state === "connecting") return "idle";
  const age = feed.receivedAt ? (now - feed.receivedAt) / 1000 : Infinity;
  return age < 30 ? "good" : "warn";
}

function Mark() {
  return (
    <svg className="topbar__mark" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.5 L14.2 6 L9.8 6 Z" fill="var(--lamp)" />
      <rect x="9.6" y="6.4" width="4.8" height="2.4" fill="var(--lamp)" />
      <path d="M10 9.2 H14 L15 21.5 H9 Z" fill="var(--text)" />
      <rect x="9.6" y="12.6" width="4.8" height="2.6" fill="var(--covert)" />
      <path d="M14.6 7.6 L23 4.8 L23 10.4 Z" fill="var(--lamp)" opacity="0.35" />
    </svg>
  );
}

export default function TopBar({
  view,
  onView,
  model,
  statusLabel,
  statusTone,
  night,
  nights,
  trial,
  mode,
  elapsed,
  eta,
  feeds,
  canStop,
  stopping,
  onStop,
  canNew,
  onNew,
}: TopBarProps) {
  const now = useNow(1000);
  const [arming, setArming] = useState(false);

  useEffect(() => {
    if (!arming) return;
    const id = window.setTimeout(() => setArming(false), 4000);
    return () => window.clearTimeout(id);
  }, [arming]);

  const progress = night != null && nights ? Math.min(1, night / nights) : 0;

  return (
    <header className="topbar">
      <div className="topbar__brand">
        <Mark />
        <div>
          <h1>EL FAROL</h1>
          <p title={model ?? undefined}>{model ?? "AGENT SOCIETY SIMULATOR"}</p>
        </div>
      </div>

      <div className="topbar__clock" aria-live="polite">
        <div className="topbar__night">
          <span>NIGHT</span>
          <strong>{night ?? "—"}</strong>
          <em>/ {nights ?? "—"}</em>
        </div>
        <div className="topbar__track" aria-hidden="true">
          <i style={{ width: `${progress * 100}%` }} />
        </div>
        <div className="topbar__meta">
          <span className="modechip" data-mode={mode}>
            {MODE_LABEL[mode]}
          </span>
          {trial && mode !== "setup" && (
            <span className="topbar__trial">
              <i style={{ background: conditionColor(trial.condition) }} aria-hidden="true" />
              {conditionLabel(trial.condition)} · SEED {trial.seed}
            </span>
          )}
        </div>
      </div>

      <div className="topbar__right">
        <div className="segmented topbar__views" role="tablist" aria-label="View">
          <button type="button" role="tab" className="btn btn--seg" aria-selected={view === "world"} aria-pressed={view === "world"} onClick={() => onView("world")}>
            WORLD
          </button>
          <button type="button" role="tab" className="btn btn--seg" aria-selected={view === "lab"} aria-pressed={view === "lab"} onClick={() => onView("lab")}>
            LAB
          </button>
        </div>
        <div className="topbar__status">
          <span className="pill" data-tone={statusTone}>
            <span className="pill__dot" aria-hidden="true" />
            {statusLabel}
          </span>
          <span className="topbar__times">
            {fmtDuration(elapsed)}
            {eta != null && <small> · ETA {fmtDuration(eta)}</small>}
          </span>
        </div>
        {canStop ? (
          <button
            type="button"
            className="btn btn--danger"
            disabled={stopping}
            onClick={() => {
              if (arming) {
                setArming(false);
                onStop();
              } else {
                setArming(true);
              }
            }}
          >
            {stopping ? "STOPPING…" : arming ? "CONFIRM STOP" : "■ STOP"}
          </button>
        ) : (
          <button type="button" className="btn btn--go" onClick={onNew} disabled={!canNew}>
            + NEW SIMULATION
          </button>
        )}
        <ul className="topbar__feeds" aria-label="Data feeds">
          {feeds.map((feed) => (
            <li key={feed.name} className="feed" data-tone={feedTone(feed, now)} title={`${feed.name} · /data/${feed.file}`}>
              <span className="feed__dot" aria-hidden="true" />
            </li>
          ))}
          <li className="feed feed--clock">{new Date(now).toLocaleTimeString([], { hour12: false })}</li>
        </ul>
      </div>
    </header>
  );
}
