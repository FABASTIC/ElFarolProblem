import { useEffect, useState } from "react";
import { fmtDuration } from "../format";
import type { FeedState } from "../telemetry";

export interface FeedHealth {
  name: string;
  file: string;
  state: FeedState;
  receivedAt: number | null;
}

interface HeaderProps {
  model: string | null;
  status: string | null;
  elapsed: number | null;
  eta: number | null;
  feeds: FeedHealth[];
}

const STATUS: Record<string, { label: string; tone: string }> = {
  running: { label: "RUNNING", tone: "live" },
  loading_engine: { label: "LOADING ENGINE", tone: "warn" },
  booting: { label: "BOOTING", tone: "warn" },
  complete: { label: "COMPLETE", tone: "good" },
  failed: { label: "FAILED", tone: "fail" },
  interrupted: { label: "INTERRUPTED", tone: "warn" },
  stalled: { label: "STALLED", tone: "fail" },
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

export default function Header({ model, status, elapsed, eta, feeds }: HeaderProps) {
  const now = useNow(1000);
  const descriptor = (status && STATUS[status]) || { label: status ? status.toUpperCase() : "NO SIGNAL", tone: "idle" };

  return (
    <header className="panel panel--glass header" style={{ ["--stagger" as string]: 0 }}>
      <div className="header__brand">
        <span className="header__mark" aria-hidden="true">
          <i style={{ background: "var(--control)" }} />
          <i style={{ background: "var(--delta2)" }} />
        </span>
        <div>
          <h1 className="header__title">EL FAROL // CONTROL CENTER</h1>
          <p className="header__sub">{model ?? "MODEL —"}</p>
        </div>
      </div>
      <div className="header__status">
        <span className="pill" data-tone={descriptor.tone}>
          <span className="pill__dot" aria-hidden="true" />
          {descriptor.label}
        </span>
        <dl className="header__stats">
          <div>
            <dt>ELAPSED</dt>
            <dd>{fmtDuration(elapsed)}</dd>
          </div>
          <div>
            <dt>ETA</dt>
            <dd>{status === "running" || status === "loading_engine" ? fmtDuration(eta) : "—"}</dd>
          </div>
        </dl>
      </div>
      <ul className="header__feeds" aria-label="Data feeds">
        {feeds.map((feed) => {
          const tone = feedTone(feed, now);
          const age = feed.receivedAt ? Math.max(0, Math.round((now - feed.receivedAt) / 1000)) : null;
          return (
            <li key={feed.name} className="feed" data-tone={tone} title={`/data/${feed.file}`}>
              <span className="feed__dot" aria-hidden="true" />
              <span className="feed__name">{feed.name}</span>
              <span className="feed__age">{feed.state === "missing" ? "ABSENT" : age == null ? "…" : `${age}S`}</span>
            </li>
          );
        })}
        <li className="feed feed--clock">{new Date(now).toLocaleTimeString([], { hour12: false })}</li>
      </ul>
    </header>
  );
}
