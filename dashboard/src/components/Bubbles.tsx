import type { CSSProperties } from "react";
import { STRATEGY_COLOR, STRATEGY_SHORT, type AgentState, type Frame } from "../model";
import { narrate } from "../narrative";
import { nameOf } from "../people";
import type { Placements } from "../scene/layout";

interface BubblesProps {
  frame: Frame | null;
  selectedId: number | null;
  hoveredId: number | null;
  speech: boolean;
  placements: Placements;
  threshold: number;
  onSelect: (id: number) => void;
}

const MAX_SPEECH = 6;
const SPREAD = 3.5;

function clip(text: string, limit: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > limit ? `${flat.slice(0, limit - 1)}…` : flat;
}

function rotation(id: number, epoch: number): number {
  const v = Math.sin(id * 12.9898 + epoch * 78.233) * 43758.5453;
  return v - Math.floor(v);
}

function accent(agent: AgentState): CSSProperties {
  return { ["--accent" as string]: STRATEGY_COLOR[agent.strategy] } as CSSProperties;
}

function tagOf(agent: AgentState): string {
  return agent.broadcast ? STRATEGY_SHORT[agent.strategy] : agent.inBar ? "WENT" : "STAYED";
}

export default function Bubbles({ frame, selectedId, hoveredId, speech, placements, threshold, onSelect }: BubblesProps) {
  if (!frame || frame.preview) return null;
  const crowded = frame.agents.filter((a) => a.inBar).length > threshold;
  const selected = selectedId != null ? frame.agents.find((a) => a.id === selectedId) ?? null : null;
  const hovered = hoveredId != null && hoveredId !== selectedId ? frame.agents.find((a) => a.id === hoveredId) ?? null : null;
  const lies = (a: AgentState) => !!a.broadcast && (a.strategy === "false_go" || a.strategy === "false_stay");
  const candidates = speech
    ? frame.agents
        .filter((a) => a.id !== selectedId && (a.mind || a.broadcast))
        .map((agent) => ({ agent, lied: lies(agent), order: rotation(agent.id, frame.epoch) }))
        .sort((a, b) => Number(b.lied) - Number(a.lied) || a.order - b.order)
    : [];
  const speakers: typeof candidates = [];
  const said = new Set<string>();
  let liars = 0;
  for (const candidate of candidates) {
    if (speakers.length >= MAX_SPEECH) break;
    if (candidate.lied && liars >= Math.ceil(MAX_SPEECH * 0.67)) continue;
    const here = placements.get(candidate.agent.id);
    const close = speakers.some((s) => {
      const there = placements.get(s.agent.id);
      return here && there ? Math.hypot(here.x - there.x, here.z - there.z) < SPREAD : false;
    });
    if (close) continue;
    const line = narrate(candidate.agent, threshold, crowded).headline;
    if (said.has(line)) continue;
    said.add(line);
    speakers.push(candidate);
    if (candidate.lied) liars += 1;
  }
  const focus = selected ? narrate(selected, threshold, crowded) : null;

  return (
    <>
      <svg className="leaders" aria-hidden="true">
        {speakers.map(({ agent, lied }, slot) => (
          <g key={`l${agent.id}`} data-leader={agent.id} data-lie={lied} style={accent(agent)}>
            <path className="leaders__halo" />
            <path className="leaders__line" />
            <circle className="leaders__ring" r={7} />
            <circle className="leaders__dot" r={2.2} />
            <text className="leaders__num">{String(slot + 1).padStart(2, "0")}</text>
          </g>
        ))}
      </svg>
      {speakers.map(({ agent, lied }, slot) => {
        const story = narrate(agent, threshold, crowded);
        return (
          <div key={`s${agent.id}`} className="tag tag--callout" data-agent={agent.id} data-callout="true" data-slot={slot} data-lift="1.3">
            <button type="button" className="tag__body callout" data-lie={lied} style={accent(agent)} onClick={() => onSelect(agent.id)} aria-label={`${nameOf(agent.id)}: ${story.headline}`}>
              <span className="callout__meta">
                <span className="callout__num">{String(slot + 1).padStart(2, "0")}</span>
                <span className="callout__name">{nameOf(agent.id)}</span>
                {agent.mind && <span className="callout__arch">{agent.mind.archetype}</span>}
                <span className="callout__tag" data-lie={lied}>
                  {lied ? "LIE" : tagOf(agent)}
                </span>
              </span>
              <span className="callout__quote">“{story.headline}”</span>
              {story.context && <span className="callout__ctx">{story.context}</span>}
            </button>
          </div>
        );
      })}
      {selected && focus && (
        <div className="tag" data-agent={selected.id} data-lift="1.5">
          <div className="tag__body callout callout--focus" data-lie={focus.lied} style={accent(selected)}>
            <span className="callout__meta">
              <span className="callout__name">{nameOf(selected.id)}</span>
              {selected.mind && <span className="callout__arch">{selected.mind.archetype}</span>}
              <span className="callout__tag" data-lie={focus.lied}>
                {focus.lied ? "LIE" : tagOf(selected)}
              </span>
            </span>
            <span className="callout__quote">“{focus.headline}”</span>
            {selected.broadcast && <span className="callout__said">Said aloud · “{clip(selected.broadcast, 90)}”</span>}
            {focus.context && <span className="callout__ctx callout__ctx--open">{focus.context}</span>}
            {focus.outcome && <span className="callout__outcome">{focus.outcome}</span>}
          </div>
        </div>
      )}
      {hovered && (
        <div className="tag" data-agent={hovered.id} data-lift="1.2">
          <div className="tag__body nametag">
            {nameOf(hovered.id)}
            {hovered.mind ? <em>{hovered.mind.archetype}</em> : null}
          </div>
        </div>
      )}
    </>
  );
}
