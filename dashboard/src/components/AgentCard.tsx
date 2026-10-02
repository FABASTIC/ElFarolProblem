import { useMemo } from "react";
import { fmt, fmtSigned, isNum } from "../format";
import { STRATEGIES, STRATEGY_COLOR, STRATEGY_LABEL, STRATEGY_STORY, type AgentState, type Frame } from "../model";
import { narrate } from "../narrative";
import { ARCHETYPE_BLURB, ARCHETYPE_COLOR, TRAIT_LABEL, handleOf, nameOf } from "../people";
import type { StrategyKey } from "../types";
import { Subfold } from "./Fold";

interface AgentCardProps {
  agent: AgentState;
  frames: Frame[];
  index: number;
  threshold: number;
  members: number;
  onClose: () => void;
}

interface Step {
  epoch: number;
  strategy: StrategyKey;
  inBar: boolean;
  cumulative: number;
}

function find(frame: Frame, id: number): AgentState | undefined {
  const guess = frame.agents[id - 1];
  return guess && guess.id === id ? guess : frame.agents.find((a) => a.id === id);
}

function Meter({ value, color, label }: { value: number | null; color?: string; label: string }) {
  const v = isNum(value) ? Math.max(0, Math.min(1, value)) : 0;
  return (
    <div className="meterline">
      <span className="meterline__label">{label}</span>
      <span className="meterline__track" aria-hidden="true">
        <span style={{ width: `${v * 100}%`, background: color ?? "var(--text-2)" }} />
      </span>
      <span className="meterline__value">{isNum(value) ? value.toFixed(2) : "—"}</span>
    </div>
  );
}

function History({ steps, current }: { steps: Step[]; current: number }) {
  if (steps.length < 2) return <p className="empty">History builds as nights pass.</p>;
  const w = 268;
  const h = 64;
  const lo = Math.min(0, ...steps.map((s) => s.cumulative));
  const hi = Math.max(1, ...steps.map((s) => s.cumulative));
  const step = w / Math.max(1, steps.length - 1);
  const sx = (i: number) => i * step;
  const sy = (v: number) => 22 + (1 - (v - lo) / (hi - lo || 1)) * (h - 26);
  const path = steps.map((s, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(s.cumulative).toFixed(1)}`).join("");
  const dot = Math.max(1.5, Math.min(3.5, step / 2 - 0.5));
  return (
    <svg className="history" width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Where this person was each night and their running utility">
      <line x1={0} x2={w} y1={sy(0)} y2={sy(0)} className="history__zero" />
      <path d={path} className="history__line" />
      {steps.map((s, i) => (
        <circle key={s.epoch} cx={sx(i)} cy={s.inBar ? 5 : 15} r={s.epoch === current ? dot + 1.5 : dot} fill={STRATEGY_COLOR[s.strategy]}>
          <title>
            Night {s.epoch + 1}: {STRATEGY_LABEL[s.strategy]} · total {fmt(s.cumulative, 1)}
          </title>
        </circle>
      ))}
    </svg>
  );
}

export default function AgentCard({ agent, frames, index, threshold, members, onClose }: AgentCardProps) {
  const steps = useMemo(() => {
    const out: Step[] = [];
    for (let i = 0; i <= Math.min(index, frames.length - 1); i += 1) {
      const a = find(frames[i], agent.id);
      if (a) out.push({ epoch: frames[i].epoch, strategy: a.strategy, inBar: a.inBar, cumulative: a.cumulative });
    }
    return out;
  }, [frames, index, agent.id]);

  const tonight = frames[index];
  const crowded = tonight ? tonight.agents.filter((a) => a.inBar).length > threshold : null;
  const story = narrate(agent, threshold, crowded);
  const mind = agent.mind ?? null;
  const lied = agent.strategy === "false_go" || agent.strategy === "false_stay";
  const instinct = mind ? STRATEGIES.map((s) => ({ key: s, p: mind.instinct[s] ?? 0 })) : [];
  const top = instinct.length ? instinct.reduce((best, item) => (item.p > best.p ? item : best), instinct[0]) : null;
  const followed = top ? top.key === agent.strategy : null;
  const visits = steps.filter((s) => s.inBar).length;
  const fibs = steps.filter((s) => s.strategy === "false_go" || s.strategy === "false_stay").length;
  const scale = Math.max(members, threshold + 10);
  const archetypeColor = mind ? ARCHETYPE_COLOR[mind.archetype] : undefined;

  return (
    <section className="hud-card agent" aria-label={`${nameOf(agent.id)}, agent ${agent.id}`}>
      <header className="agent__head">
        <span className="agent__avatar" style={{ background: STRATEGY_COLOR[agent.strategy] }} aria-hidden="true">
          {nameOf(agent.id).slice(0, 1)}
        </span>
        <div className="agent__id">
          <h2>{nameOf(agent.id)}</h2>
          <p>{handleOf(agent.id)}</p>
        </div>
        <button type="button" className="btn btn--ghost" onClick={onClose} aria-label="Close inspector">
          ✕
        </button>
      </header>

      {mind && (
        <p className="agent__archetype" title={ARCHETYPE_BLURB[mind.archetype]}>
          <i style={{ background: archetypeColor }} aria-hidden="true" />
          <strong>{mind.archetype}</strong>
          <span>{ARCHETYPE_BLURB[mind.archetype] ?? ""}</span>
        </p>
      )}

      <div className="agent__plain" data-lie={story.lied}>
        <p className="eyebrow">IN PLAIN WORDS</p>
        <p className="agent__plain-line">“{story.headline}”</p>
        {story.context && <p className="agent__plain-ctx">{story.context}</p>}
        {story.outcome && <p className="agent__plain-out">{story.outcome}</p>}
      </div>

      <div className="agent__tonight" data-lie={lied}>
        <span className="agent__chip" style={{ borderColor: STRATEGY_COLOR[agent.strategy] }}>
          <i style={{ background: STRATEGY_COLOR[agent.strategy] }} aria-hidden="true" />
          {STRATEGY_LABEL[agent.strategy]}
        </span>
        <p>{STRATEGY_STORY[agent.strategy]}.</p>
        <dl>
          <div>
            <dt>TONIGHT</dt>
            <dd>{fmtSigned(agent.utility, 1)}</dd>
          </div>
          <div>
            <dt>TOTAL</dt>
            <dd>{fmt(agent.cumulative, 1)}</dd>
          </div>
          <div>
            <dt>VISITS</dt>
            <dd>
              {visits}/{steps.length}
            </dd>
          </div>
          <div>
            <dt>FIBS</dt>
            <dd>{mind?.lies ?? fibs}</dd>
          </div>
        </dl>
      </div>

      {(mind?.note || agent.broadcast) && (
        <Subfold id="agent.raw" title="RAW SIGNALS" meta={agent.broadcast ? "SAID + THOUGHT" : "THOUGHT"}>
          {mind?.note && (
            <blockquote className="agent__thought" data-lie={lied}>
              <span>{lied ? "PRIVATE NOTE · WHILE LYING" : "PRIVATE NOTE"}</span>“{mind.note}”
            </blockquote>
          )}
          {agent.broadcast && (
            <blockquote className="agent__said">
              <span>SAID TO EVERYONE</span>“{agent.broadcast}”
            </blockquote>
          )}
        </Subfold>
      )}

      {mind ? (
        <>
          <Subfold id="agent.forecast" title="FORECAST" meta={isNum(mind.forecast) ? `${fmt(mind.forecast, 0)} / ${threshold}` : "—"} defaultOpen>
            <div className="forecast">
              <div className="forecast__track" aria-hidden="true">
                <span className="forecast__line" style={{ left: `${(threshold / scale) * 100}%` }} />
                {isNum(mind.forecast) && (
                  <span
                    className="forecast__dot"
                    data-tone={mind.forecast > threshold ? "fail" : "good"}
                    style={{ left: `${(Math.min(scale, Math.max(0, mind.forecast)) / scale) * 100}%` }}
                  />
                )}
              </div>
              <p>
                Expects <strong>{fmt(mind.forecast, 0)}</strong> in the bar against a comfort line of {threshold}
                {mind.predictor ? ` · method: ${mind.predictor}` : ""} · confidence {fmt(mind.forecastConfidence, 2)}
              </p>
            </div>
          </Subfold>

          <Subfold
            id="agent.instinct"
            title="GUT INSTINCT"
            meta={followed != null ? <span className="agent__badge" data-tone={followed ? "good" : "warn"}>{followed ? "✓ FOLLOWED" : "! OVERRODE"}</span> : null}
          >
            {instinct.map((item) => (
              <div key={item.key} className="instinct" data-chosen={item.key === agent.strategy}>
                <span className="instinct__label">{STRATEGY_LABEL[item.key]}</span>
                <span className="instinct__track" aria-hidden="true">
                  <span style={{ width: `${item.p * 100}%`, background: STRATEGY_COLOR[item.key] }} />
                </span>
                <span className="instinct__value">{(item.p * 100).toFixed(0)}%</span>
              </div>
            ))}
          </Subfold>

          <Subfold id="agent.temper" title="MOOD & TEMPERAMENT" meta={mind.mood.toUpperCase()}>
            <div className="agent__pair">
              <div>
                <p className="eyebrow">MOOD</p>
                <p className="agent__mood">
                  {mind.mood.toUpperCase()}
                  {mind.mood !== "calm" && <small> {fmt(mind.moodIntensity, 2)}</small>}
                </p>
              </div>
              <div>
                <p className="eyebrow">REPUTATION</p>
                <Meter value={mind.reputation} label="" />
              </div>
            </div>
            {mind.traits && (
              <div className="traits">
                {Object.entries(TRAIT_LABEL).map(([key, label]) => (
                  <Meter key={key} label={label} value={mind.traits?.[key] ?? null} />
                ))}
              </div>
            )}
          </Subfold>
        </>
      ) : (
        <p className="empty">No mind trace for this person.</p>
      )}

      <Subfold id="agent.history" title="NIGHTS" meta={`${visits} AT THE BAR`} defaultOpen>
        <p className="hud-hint">Top row at the bar, bottom at home. The line is running utility.</p>
        <History steps={steps} current={frames[index]?.epoch ?? -1} />
      </Subfold>
    </section>
  );
}
