import { useEffect, useState } from "react";
import DitheredLogo from "./DitheredLogo";
import { Segmented } from "./Kinetic";
import SplitFlap, { FLAP_DIGITS } from "./SplitFlap";
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
  onAbstract: () => void;
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

const LOGO = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><defs><linearGradient id="t" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ece9e2"/><stop offset="1" stop-color="#ece9e2" stop-opacity="0.42"/></linearGradient><linearGradient id="b" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#ffcf7a"/><stop offset="1" stop-color="#ffcf7a" stop-opacity="0"/></linearGradient></defs><path d="M15.6 36 L17.4 16.5 H22.6 L24.4 36 Z" fill="url(#t)"/><rect x="15" y="14.4" width="10" height="2.1" fill="#ece9e2"/><rect x="16.8" y="9.4" width="6.4" height="5" fill="#ffcf7a"/><path d="M16.2 9.4 L20 5 L23.8 9.4 Z" fill="#ece9e2"/><path d="M23.2 10.4 L39.5 5 L39.5 19 Z" fill="url(#b)"/><path d="M16.8 11.8 L0.5 6.4 L0.5 19.4 Z" fill="url(#b)" transform="translate(17.3 0) scale(-1 1) translate(-17.3 0)" opacity="0.6"/><rect x="8" y="36" width="24" height="2.4" fill="#ece9e2"/></svg>`;

const TONE_COLOR: Record<string, string> = {
  good: "#0ca30c",
  live: "#5fd35f",
  warn: "#fab219",
  fail: "#d03b3b",
  idle: "#85827c",
};

function Mark() {
  return (
    <span className="topbar__mark">
      <DitheredLogo svg={LOGO} size={38} grid={34} threshold={96} />
    </span>
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
  onAbstract,
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
          <strong>
            <SplitFlap text={night != null ? String(night).padStart(2, "0") : "--"} charset={night != null ? FLAP_DIGITS : " -"} size="md" align="right" columns={Math.max(2, String(nights ?? 0).length)} label={night != null ? `Night ${night}` : "No night"} />
          </strong>
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
        <button type="button" className="btn btn--tiny topbar__abstract" onClick={onAbstract}>
          <span className="topbar__abstract-glyph" aria-hidden="true">
            §
          </span>
          ABSTRACT
        </button>
        <Segmented
          kind="tab"
          label="View"
          className="topbar__views"
          value={view}
          onChange={onView}
          options={[
            { id: "world", label: "WORLD" },
            { id: "lab", label: "LAB" },
          ]}
        />
        <div className="topbar__status">
          <span className="pill pill--flap" data-tone={statusTone}>
            <SplitFlap text={statusLabel} size="sm" showIndicators accentColor={TONE_COLOR[statusTone] ?? TONE_COLOR.idle} label={statusLabel} />
          </span>
          {elapsed != null && (
            <span className="topbar__times">
              {fmtDuration(elapsed)}
              {eta != null && <small> · ETA {fmtDuration(eta)}</small>}
            </span>
          )}
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
