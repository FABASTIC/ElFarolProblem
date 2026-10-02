import type { Frame } from "../model";
import { nameOf } from "../people";

interface BubblesProps {
  frame: Frame | null;
  selectedId: number | null;
  hoveredId: number | null;
  speech: boolean;
}

const MAX_SPEECH = 6;
const SPREAD = 6;

function clip(text: string, limit: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > limit ? `${flat.slice(0, limit - 1)}…` : flat;
}

export default function Bubbles({ frame, selectedId, hoveredId, speech }: BubblesProps) {
  if (!frame || frame.preview) return null;
  const selected = selectedId != null ? frame.agents.find((a) => a.id === selectedId) ?? null : null;
  const hovered = hoveredId != null && hoveredId !== selectedId ? frame.agents.find((a) => a.id === hoveredId) ?? null : null;
  const candidates = speech
    ? frame.agents
        .filter((a) => a.broadcast && a.id !== selectedId)
        .map((a) => ({ agent: a, lied: a.strategy === "false_go" || a.strategy === "false_stay" }))
        .sort((a, b) => Number(b.lied) - Number(a.lied) || a.agent.id - b.agent.id)
    : [];
  const speakers: typeof candidates = [];
  for (const candidate of candidates) {
    if (speakers.length >= MAX_SPEECH) break;
    const crowded = speakers.some((s) => Math.max(Math.abs(s.agent.x - candidate.agent.x), Math.abs(s.agent.y - candidate.agent.y)) < SPREAD);
    if (!crowded) speakers.push(candidate);
  }

  return (
    <>
      {speakers.map(({ agent, lied }) => (
        <div key={`s${agent.id}`} className="tag" data-agent={agent.id} data-lift="1.3">
          <div className="tag__body bubble" data-lie={lied}>
            <span className="bubble__who">
              {nameOf(agent.id)}
              {lied && <em className="bubble__flag">FALSE</em>}
            </span>
            {clip(agent.broadcast ?? "", 70)}
          </div>
        </div>
      ))}
      {selected && (
        <div className="tag" data-agent={selected.id} data-lift="1.45">
          <div className="tag__body bubble bubble--focus">
            <span className="bubble__who">
              {nameOf(selected.id)}
              {selected.mind ? ` · ${selected.mind.archetype}` : ""}
            </span>
            {selected.broadcast && <span className="bubble__said">“{clip(selected.broadcast, 90)}”</span>}
            {selected.mind?.note ? <span className="bubble__thought">{clip(selected.mind.note, 140)}</span> : !selected.broadcast && <span className="bubble__quiet">keeps to themself tonight</span>}
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
