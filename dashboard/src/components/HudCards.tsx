import { useState } from "react";
import { fmt, fmtClock, fmtPct } from "../format";
import { conditionColor, conditionLabel, type Frame, type GameView, type TrialEntry } from "../model";
import { LENSES, handleOf, lensLegend, nameOf, type Lens, DIVERGING } from "../people";
import type { WorldMode } from "./TopBar";

const STATUS_TONE: Record<string, string> = {
  complete: "good",
  resumed: "good",
  running: "live",
  failed: "fail",
  interrupted: "warn",
  pending: "idle",
};

interface TonightCardProps {
  frame: Frame | null;
  game: GameView;
  mode: WorldMode;
  members: number;
}

export function TonightCard({ frame, game, mode, members }: TonightCardProps) {
  const staged = mode === "setup" || mode === "boot" || !frame || frame.preview;
  const n = staged ? members : frame.agents.length;
  const attendance = staged ? null : frame.agents.filter((a) => a.inBar).length;
  const lying = staged ? null : frame.agents.filter((a) => a.strategy === "false_go" || a.strategy === "false_stay").length;
  const spoke = staged ? null : frame.agents.filter((a) => a.broadcast).length;
  const utility = staged || !n ? null : frame.agents.reduce((acc, a) => acc + (Number.isFinite(a.utility) ? a.utility : 0), 0) / n;
  const scale = Math.max(n, game.threshold + 10, 1);
  const crowded = attendance != null && attendance > game.threshold;
  const verdict = attendance == null ? "BAR CLOSED" : crowded ? "OVERCROWDED" : "COMFORTABLE";

  return (
    <section className="hud-card tonight" aria-label="Tonight at the bar">
      <p className="eyebrow">{staged ? "BEFORE THE FIRST NIGHT" : `NIGHT ${frame.epoch + 1}`}</p>
      <div className="tonight__hero">
        <strong>{attendance ?? n}</strong>
        <span>
          {attendance == null ? "members staged" : "in the bar"}
          <br />
          {attendance == null ? `${game.threshold} comfortable seats` : `of ${n} members`}
        </span>
        <span className="verdict" data-tone={attendance == null ? "idle" : crowded ? "fail" : "good"}>
          <i aria-hidden="true">{attendance == null ? "○" : crowded ? "▲" : "✓"}</i>
          {verdict}
        </span>
      </div>
      <div className="gauge" role="meter" aria-label="Bar attendance" aria-valuemin={0} aria-valuemax={scale} aria-valuenow={attendance ?? 0}>
        <div className="gauge__fill" data-tone={crowded ? "fail" : "good"} style={{ width: `${((attendance ?? 0) / scale) * 100}%` }} />
        <div className="gauge__line" style={{ left: `${(game.threshold / scale) * 100}%` }}>
          <em>T {game.threshold}</em>
        </div>
      </div>
      <dl className="tonight__stats">
        <div>
          <dt>AT HOME</dt>
          <dd>{attendance == null ? n : n - attendance}</dd>
        </div>
        <div>
          <dt>LYING</dt>
          <dd>
            {lying ?? "—"}
            {lying != null && n > 0 && <small> {fmtPct(lying / n, 0)}</small>}
          </dd>
        </div>
        <div>
          <dt>SPOKE</dt>
          <dd>{spoke ?? "—"}</dd>
        </div>
        <div>
          <dt>MEAN UTIL</dt>
          <dd>{fmt(utility, 2)}</dd>
        </div>
      </dl>
      {staged && (
        <p className="tonight__note">
          {members > game.threshold
            ? `${members} members, ${game.threshold} comfortable seats: someone will be disappointed.`
            : `${members} members can never fill ${game.threshold} seats. Add more to make it a dilemma.`}
        </p>
      )}
    </section>
  );
}

interface PopulationCardProps {
  frame: Frame | null;
  lens: Lens;
  onLens: (lens: Lens) => void;
  isolate: string | null;
  onIsolate: (key: string | null) => void;
  staged: boolean;
}

export function PopulationCard({ frame, lens, onLens, isolate, onIsolate, staged }: PopulationCardProps) {
  const agents = frame && !frame.preview ? frame.agents : [];
  const legend = lensLegend(agents, lens);
  const total = Math.max(1, agents.length);
  const noMinds = agents.length > 0 && agents.every((a) => !a.mind) && (lens === "archetype" || lens === "mood");

  return (
    <section className="hud-card population" aria-label="Population">
      <div className="hud-card__title">
        <p className="eyebrow">COLOR PEOPLE BY</p>
        {isolate && (
          <button type="button" className="btn btn--tiny" onClick={() => onIsolate(null)}>
            SHOW ALL
          </button>
        )}
      </div>
      <div className="lenses" role="radiogroup" aria-label="Color lens">
        {LENSES.map((item) => (
          <button key={item.id} type="button" role="radio" aria-checked={lens === item.id} className="lens" onClick={() => onLens(item.id)}>
            {item.label}
          </button>
        ))}
      </div>
      {staged ? (
        <p className="empty">Members are unassigned until the first night.</p>
      ) : noMinds ? (
        <p className="empty">This run has no mind trace (clone agents).</p>
      ) : (
        <>
          {lens === "wealth" && (
            <div className="ramp" aria-hidden="true">
              <i style={{ background: `linear-gradient(90deg, ${DIVERGING.warm}, ${DIVERGING.neutral}, ${DIVERGING.cool})` }} />
              <span>BEHIND</span>
              <span>EVEN</span>
              <span>AHEAD</span>
            </div>
          )}
          <ul className="legend">
            {legend.map((item) => (
              <li key={item.key}>
                <button
                  type="button"
                  className="legend__item"
                  aria-pressed={isolate === item.key}
                  data-dim={isolate != null && isolate !== item.key}
                  onClick={() => onIsolate(isolate === item.key ? null : item.key)}
                  title={item.hint ?? `Show only ${item.label.toLowerCase()}`}
                >
                  <i className="legend__dot" style={{ background: item.color }} aria-hidden="true" />
                  <span className="legend__label">{item.label}</span>
                  <span className="legend__count">{item.count}</span>
                  <span className="legend__bar" aria-hidden="true">
                    <span style={{ width: `${(item.count / total) * 100}%`, background: item.color }} />
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <p className="hud-hint">Click a legend row to isolate it. Click a person to read their mind.</p>
        </>
      )}
    </section>
  );
}

interface CrierCardProps {
  frame: Frame | null;
  log: string[];
  events: { t: number; message: string }[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export function CrierCard({ frame, log, events, selectedId, onSelect }: CrierCardProps) {
  const [tab, setTab] = useState<"thoughts" | "said" | "log">("thoughts");
  const live = frame && !frame.preview ? frame : null;
  const thoughts = live?.thoughts ?? [];
  const said = live?.broadcasts ?? [];
  const lies = thoughts.filter((t) => t.lied).length;
  const byId = new Map((live?.agents ?? []).map((a) => [a.id, a]));

  return (
    <section className="hud-card crier" aria-label="Town crier">
      <div className="tabs" role="tablist" aria-label="Feeds">
        <button type="button" role="tab" aria-selected={tab === "thoughts"} className="tab" onClick={() => setTab("thoughts")}>
          THOUGHTS <span className="tab__count">{lies ? `${lies} LIE` : thoughts.length}</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === "said"} className="tab" onClick={() => setTab("said")}>
          SAID ALOUD <span className="tab__count">{said.length}</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === "log"} className="tab" onClick={() => setTab("log")}>
          ENGINE LOG
        </button>
      </div>
      <ol className="crier__list" role="tabpanel">
        {tab === "thoughts" &&
          (thoughts.length ? (
            thoughts.map((t) => (
              <li key={`${t.agentId}-${t.note}`}>
                <button type="button" className="crier__item" data-lie={t.lied} aria-pressed={selectedId === t.agentId} onClick={() => onSelect(t.agentId)}>
                  <span className="crier__who">
                    <strong>{nameOf(t.agentId)}</strong>
                    <span>{handleOf(t.agentId)}</span>
                    <span>{t.archetype}</span>
                    {t.lied && <span className="crier__tag">LIED</span>}
                  </span>
                  <span className="crier__text">“{t.note}”</span>
                </button>
              </li>
            ))
          ) : (
            <li className="empty">{live ? "No private notes this night." : "Thoughts appear once the first night resolves."}</li>
          ))}
        {tab === "said" &&
          (said.length ? (
            said.map((s) => {
              const agent = byId.get(s.agentId);
              const lied = agent ? agent.strategy === "false_go" || agent.strategy === "false_stay" : false;
              return (
                <li key={`${s.agentId}-${s.text}`}>
                  <button type="button" className="crier__item" data-lie={lied} aria-pressed={selectedId === s.agentId} onClick={() => onSelect(s.agentId)}>
                    <span className="crier__who">
                      <strong>{nameOf(s.agentId)}</strong>
                      <span>{handleOf(s.agentId)}</span>
                      {agent && <span>CLAIMED {agent.stated.toUpperCase()}</span>}
                      {lied && <span className="crier__tag">FALSE</span>}
                    </span>
                    <span className="crier__text">{s.text}</span>
                  </button>
                </li>
              );
            })
          ) : (
            <li className="empty">{live ? "Nobody broadcast this night (control towns cannot)." : "Broadcasts appear once the first night resolves."}</li>
          ))}
        {tab === "log" &&
          (log.length || events.length ? (
            (log.length ? log.slice(-80) : events.slice(-80).map((e) => `${fmtClock(e.t)}  ${e.message}`))
              .slice()
              .reverse()
              .map((line, i) => (
                <li key={`${i}-${line}`} className="crier__log">
                  {line}
                </li>
              ))
          ) : (
            <li className="empty">No engine output yet.</li>
          ))}
      </ol>
    </section>
  );
}

interface RunsCardProps {
  trials: TrialEntry[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
  follow: boolean;
  onFollow: (value: boolean) => void;
  liveRunning: boolean;
}

export function RunsCard({ trials, selectedKey, onSelect, follow, onFollow, liveRunning }: RunsCardProps) {
  return (
    <section className="hud-card runs" aria-label="Runs">
      <div className="hud-card__title">
        <p className="eyebrow">TOWNS IN THIS SWEEP</p>
        <button type="button" className="btn btn--tiny" aria-pressed={follow && liveRunning} disabled={!liveRunning} onClick={() => onFollow(!follow)}>
          FOLLOW LIVE
        </button>
      </div>
      {trials.length === 0 ? (
        <p className="empty">No towns on disk yet.</p>
      ) : (
        <ul className="runs__list" role="listbox" aria-label="Trials">
          {trials.map((trial) => (
            <li key={trial.key} role="presentation">
              <button
                type="button"
                role="option"
                aria-selected={trial.key === selectedKey}
                className="runs__item"
                onClick={() => onSelect(trial.key)}
                style={{ ["--key" as string]: conditionColor(trial.condition) }}
              >
                <i className="runs__key" aria-hidden="true" />
                <span className="runs__name">
                  <strong>{conditionLabel(trial.condition)}</strong>
                  <span>SEED {trial.seed || "—"}</span>
                </span>
                <span className="status" data-tone={trial.isLive ? "live" : STATUS_TONE[trial.status] ?? "idle"}>
                  <i aria-hidden="true" />
                  {trial.isLive ? "LIVE" : trial.status.toUpperCase()}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
      <p className="hud-hint">Control towns have no broadcast channel. Delta 2 towns can talk, and lie.</p>
    </section>
  );
}
