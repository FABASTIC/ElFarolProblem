import { useMemo, useRef, useState, type CSSProperties } from "react";
import { fmt, fmtClock, fmtPct } from "../format";
import { STRATEGY_COLOR, conditionColor, conditionLabel, type Frame, type GameView, type TrialEntry } from "../model";
import { narrate, summarizeNight } from "../narrative";
import { LENSES, handleOf, lensLegend, nameOf, type Lens, DIVERGING } from "../people";
import Fold from "./Fold";
import { HoverTrail, Segmented } from "./Kinetic";
import ScrollProgress from "./ScrollProgress";
import SplitFlap from "./SplitFlap";
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
    <Fold
      id="tonight"
      index="01"
      title={staged ? "BEFORE THE FIRST NIGHT" : `NIGHT ${frame.epoch + 1}`}
      label="Tonight at the bar"
      className="tonight"
      meta={
        <span className="fold__pill" data-tone={attendance == null ? "idle" : crowded ? "fail" : "good"}>
          {attendance == null ? `${n} STAGED` : `${attendance} / ${game.threshold}`}
        </span>
      }
    >
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
    </Fold>
  );
}

interface SummaryCardProps {
  frame: Frame | null;
  previous: Frame | null;
  threshold: number;
  staged: boolean;
}

export function SummaryCard({ frame, previous, threshold, staged }: SummaryCardProps) {
  const summary = useMemo(() => (frame && !frame.preview && frame.agents.length ? summarizeNight(frame, previous, threshold) : null), [frame, previous, threshold]);
  const total = summary ? Math.max(1, summary.members) : 1;
  const segments = summary
    ? summary.control
      ? [
          { key: "honest_go", value: summary.attendance, label: "WENT" },
          { key: "honest_stay", value: summary.members - summary.attendance, label: "STAYED" },
        ]
      : [
          { key: "honest_go", value: summary.honestGo, label: "HONEST · WENT" },
          { key: "honest_stay", value: summary.honestStay, label: "HONEST · STAYED" },
          { key: "false_go", value: summary.bluff, label: "BLUFFED" },
          { key: "false_stay", value: summary.covert, label: "SLIPPED IN" },
        ]
    : [];

  return (
    <Fold
      id="summary"
      index="02"
      title="DAILY SUMMARY"
      label="Daily summary in plain English"
      className="summary"
      meta={
        summary && !summary.control ? (
          <span className="fold__pill" data-tone={summary.lieRate > 0.5 ? "fail" : "good"}>
            {Math.round((1 - summary.lieRate) * 100)}% HONEST
          </span>
        ) : summary ? (
          <span className="fold__pill" data-tone="idle">
            NO TALK
          </span>
        ) : null
      }
    >
      <div className="summary__stage">
        {summary ? (
          <div className="summary__content">
            <p className="summary__night">
              <SplitFlap text={`NIGHT ${String(summary.night).padStart(3, "0")}`} size="sm" staggerDelay={24} flipSpeed={45} />
            </p>
            <h3 className="summary__headline">{summary.headline}</h3>
            <div className="summary__split" role="img" aria-label={segments.map((s) => `${s.label} ${s.value}`).join(", ")}>
              {segments.map((segment) =>
                segment.value > 0 ? (
                  <span
                    key={segment.key}
                    className="summary__seg"
                    style={{ flexGrow: segment.value, ["--seg" as string]: STRATEGY_COLOR[segment.key as keyof typeof STRATEGY_COLOR] } as CSSProperties}
                    title={`${segment.label}: ${segment.value}`}
                  />
                ) : null,
              )}
            </div>
            <dl className="summary__keys">
              {segments.map((segment) => (
                <div key={segment.key}>
                  <dt>
                    <i style={{ background: STRATEGY_COLOR[segment.key as keyof typeof STRATEGY_COLOR] }} aria-hidden="true" />
                    {segment.label}
                  </dt>
                  <dd>
                    {segment.value}
                    <small> {Math.round((segment.value / total) * 100)}%</small>
                  </dd>
                </div>
              ))}
            </dl>
            <div className="summary__lines">
              {summary.lines.map((line) => (
                <p key={line}>{line}</p>
              ))}
            </div>
            {summary.verdict && (
              <p className="summary__verdict" data-verdict={summary.verdict}>
                {summary.verdict === "lying" ? "▲ DECEPTION PAID" : summary.verdict === "honesty" ? "✓ HONESTY PAID" : "= A WASH"}
              </p>
            )}
          </div>
        ) : (
          <div className="summary__content">
            <h3 className="summary__headline summary__headline--quiet">{staged ? "No nights yet." : "Waiting for the first night."}</h3>
            <p className="summary__lines">Once a night resolves, this panel explains in plain words who told the truth, who bluffed, and whether lying paid.</p>
          </div>
        )}
      </div>
    </Fold>
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
  const box = useRef<HTMLDivElement>(null);
  const current = LENSES.find((item) => item.id === lens);

  return (
    <Fold
      id="people"
      index="03"
      title="PEOPLE"
      label="Population"
      className="population"
      defaultOpen={false}
      meta={<span className="fold__pill">{isolate ? "ISOLATED" : current?.label ?? ""}</span>}
      actions={
        isolate ? (
          <button type="button" className="btn btn--tiny" onClick={() => onIsolate(null)}>
            SHOW ALL
          </button>
        ) : null
      }
    >
      <Segmented label="Color people by" options={LENSES.map((item) => ({ id: item.id, label: item.label }))} value={lens} onChange={(next) => onLens(next as Lens)} className="lenses" />
      {staged ? (
        <p className="empty">Members are unassigned until the first night.</p>
      ) : noMinds ? (
        <p className="empty">This run has no mind trace.</p>
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
          <div className="trailbox" ref={box}>
            <HoverTrail container={box} selector=".legend__item" />
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
          </div>
          <p className="hud-hint">Click a row to isolate it. Click a person to read their mind.</p>
        </>
      )}
    </Fold>
  );
}

interface CrierCardProps {
  frame: Frame | null;
  log: string[];
  events: { t: number; message: string }[];
  threshold: number;
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export function CrierCard({ frame, log, events, threshold, selectedId, onSelect }: CrierCardProps) {
  const [tab, setTab] = useState<"thoughts" | "said" | "log">("thoughts");
  const live = frame && !frame.preview ? frame : null;
  const thoughts = live?.thoughts ?? [];
  const said = live?.broadcasts ?? [];
  const lies = thoughts.filter((t) => t.lied).length;
  const byId = new Map((live?.agents ?? []).map((a) => [a.id, a]));
  const crowded = live ? live.agents.filter((a) => a.inBar).length > threshold : null;
  const list = useRef<HTMLOListElement>(null);
  const box = useRef<HTMLDivElement>(null);

  return (
    <Fold
      id="voices"
      index="04"
      title="VOICES"
      label="Town crier"
      className="crier"
      defaultOpen={false}
      meta={
        <span className="fold__pill" data-tone={lies ? "fail" : "idle"}>
          {lies ? `${lies} LIES` : `${thoughts.length} THOUGHTS`}
        </span>
      }
    >
      <Segmented
        kind="tab"
        label="Feeds"
        className="crier__tabs"
        value={tab}
        onChange={setTab}
        options={[
          { id: "thoughts", label: "WHY" },
          { id: "said", label: `SAID · ${said.length}` },
          { id: "log", label: "RUN LOG" },
        ]}
      />
      <div className="scrollwrap trailbox" ref={box}>
        <HoverTrail container={box} selector=".crier__item" />
        <ol className="crier__list" role="tabpanel" ref={list}>
          {tab === "thoughts" &&
            (thoughts.length ? (
              thoughts.map((t) => {
                const agent = byId.get(t.agentId);
                const story = agent ? narrate(agent, threshold, crowded) : null;
                return (
                  <li key={`${t.agentId}-${t.note}`}>
                    <button type="button" className="crier__item" data-lie={t.lied} aria-pressed={selectedId === t.agentId} onClick={() => onSelect(t.agentId)}>
                      <span className="crier__who">
                        <strong>{nameOf(t.agentId)}</strong>
                        <span>{handleOf(t.agentId)}</span>
                        <span>{t.archetype}</span>
                        {t.lied && <span className="crier__tag">LIED</span>}
                      </span>
                      <span className="crier__text">“{story?.headline ?? t.note}”</span>
                      {story?.context && <span className="crier__raw">{story.context}</span>}
                    </button>
                  </li>
                );
              })
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
              <li className="empty">No run output yet.</li>
            ))}
        </ol>
        <ScrollProgress target={list} watch={`${tab}:${thoughts.length}:${said.length}:${log.length}:${frame?.epoch ?? -1}`} />
      </div>
    </Fold>
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
  const list = useRef<HTMLUListElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const done = trials.filter((t) => t.status === "complete" || t.status === "resumed").length;

  return (
    <Fold
      id="towns"
      index="01"
      title="TOWNS IN THIS SWEEP"
      label="Runs"
      className="runs"
      meta={
        <span className="fold__pill">
          {done}/{trials.length}
        </span>
      }
      actions={
        liveRunning ? (
          <button type="button" className="btn btn--tiny" aria-pressed={follow} onClick={() => onFollow(!follow)}>
            FOLLOW LIVE
          </button>
        ) : null
      }
    >
      {trials.length === 0 ? (
        <p className="empty">No towns recorded yet.</p>
      ) : (
        <div className="scrollwrap trailbox" ref={box}>
          <HoverTrail container={box} selector=".runs__item" />
          <ul className="runs__list" role="listbox" aria-label="Trials" ref={list}>
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
          <ScrollProgress target={list} watch={trials.length} />
        </div>
      )}
      <p className="hud-hint">Control towns have no broadcast channel. Delta 2 towns can talk, and lie.</p>
    </Fold>
  );
}
