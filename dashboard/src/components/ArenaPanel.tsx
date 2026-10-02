import { useCallback, useRef, useState } from "react";
import { fmt } from "../format";
import {
  STRATEGIES,
  STRATEGY_COLOR,
  STRATEGY_LABEL,
  conditionColor,
  conditionLabel,
  type Frame,
  type GameView,
  type TrialEntry,
} from "../model";
import Arena from "../scene/Arena";
import type { AgentHover } from "../scene/Agents";
import type { ReplayState } from "../telemetry";
import Boundary from "./Boundary";

interface ArenaPanelProps {
  frame: Frame | null;
  frameKey: string;
  game: GameView;
  heightRange: [number, number];
  trial: TrialEntry | null;
  mode: "live" | "replay" | "none";
  phase: string | null;
  epoch: number | null;
  lastEpoch: number | null;
  replayState: ReplayState;
}

export default function ArenaPanel({
  frame,
  frameKey,
  game,
  heightRange,
  trial,
  mode,
  phase,
  epoch,
  lastEpoch,
  replayState,
}: ArenaPanelProps) {
  const host = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<AgentHover | null>(null);
  const [resetToken, setResetToken] = useState(0);
  const onHover = useCallback((next: AgentHover | null) => setHover(next), []);
  const rect = host.current?.getBoundingClientRect();
  const attendance = frame ? frame.agents.filter((a) => a.inBar).length : null;
  const deceptive = frame ? frame.agents.filter((a) => a.strategy === "false_go" || a.strategy === "false_stay").length : null;
  const n = frame?.agents.length ?? game.n;
  const loadingEngine = mode === "live" && phase === "loading_engine";

  let notice: string | null = null;
  if (!trial) notice = "NO SIGNAL // python3 experiment.py --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4";
  else if (loadingEngine) notice = "ENGINE SPIN-UP // LOADING WEIGHTS INTO VRAM";
  else if (!frame && replayState === "loading") notice = "STREAMING TRACE…";
  else if (!frame && (replayState === "missing" || replayState === "error")) notice = "NO TRACE ON DISK FOR THIS TRIAL";
  else if (!frame) notice = "AWAITING FIRST EPOCH";

  return (
    <section className="panel panel--scene arena" ref={host} style={{ ["--stagger" as string]: 3 }} aria-label="Simulation arena">
      <Boundary
        resetKey={trial?.key}
        fallback={(error, retry) => (
          <div className="arena__fault" role="alert">
            <p className="eyebrow">RENDERER OFFLINE</p>
            <p>{error.message || "WebGL context unavailable"}</p>
            <button type="button" className="btn" onClick={retry}>
              RESTART RENDERER
            </button>
          </div>
        )}
      >
        <Arena
          frame={frame}
          frameKey={frameKey}
          gridSize={game.gridSize}
          barMin={game.barMin}
          barMax={game.barMax}
          heightRange={heightRange}
          resetToken={resetToken}
          onHover={onHover}
        />
      </Boundary>
      <div className="arena__hud">
        <div className="arena__tl">
          <p className="eyebrow">ARENA // 2.5D ORTHOGRAPHIC</p>
          {trial && (
            <p className="arena__trial">
              <i className="linekey" style={{ background: conditionColor(trial.condition) }} aria-hidden="true" />
              <strong>{conditionLabel(trial.condition)}</strong>
              <span>SEED {trial.seed}</span>
              <span className="chip" data-mode={mode}>
                {mode === "live" ? "LIVE" : mode === "replay" ? "REPLAY" : "—"}
              </span>
            </p>
          )}
          <p className="arena__caption">PILLAR HEIGHT = CUMULATIVE UTILITY · AMBER SQUARE = BAR ZONE · RINGS = BROADCASTS</p>
        </div>
        <div className="arena__tr">
          <p className="eyebrow">EPOCH</p>
          <p className="hero">
            {epoch ?? "—"}
            <span className="hero__of">/{lastEpoch ?? "—"}</span>
          </p>
          <dl className="readout">
            <div>
              <dt>IN BAR</dt>
              <dd>
                {attendance ?? "—"}
                <small> / {n}</small>
              </dd>
            </div>
            <div>
              <dt>NASH A*</dt>
              <dd>{game.primaryAttendance}</dd>
            </div>
            <div>
              <dt>COMFORT T</dt>
              <dd>{game.threshold}</dd>
            </div>
            <div>
              <dt>DECEIVING</dt>
              <dd>
                {deceptive ?? "—"}
                <small>{frame ? ` · ${fmt(deceptive != null ? deceptive / Math.max(1, n) : null, 2)}` : ""}</small>
              </dd>
            </div>
          </dl>
        </div>
        <ul className="arena__legend" aria-label="Strategy legend">
          {STRATEGIES.map((s) => (
            <li key={s}>
              <i className="swatch" style={{ background: STRATEGY_COLOR[s] }} aria-hidden="true" />
              {STRATEGY_LABEL[s]}
            </li>
          ))}
        </ul>
        <div className="arena__br">
          <span>DRAG ORBIT · WHEEL ZOOM</span>
          <button type="button" className="btn btn--tiny" onClick={() => setResetToken((t) => t + 1)}>
            RESET VIEW
          </button>
        </div>
        {notice && <p className="arena__notice">{notice}</p>}
        {hover && rect && (
          <div
            className="inspector"
            style={{ left: hover.clientX - rect.left + 14, top: hover.clientY - rect.top + 14 }}
            role="status"
          >
            <p className="inspector__head">
              <i className="swatch" style={{ background: STRATEGY_COLOR[hover.agent.strategy] }} aria-hidden="true" />
              AGENT {String(hover.agent.id).padStart(2, "0")}
            </p>
            <dl>
              <div>
                <dt>STRATEGY</dt>
                <dd>{STRATEGY_LABEL[hover.agent.strategy]}</dd>
              </div>
              <div>
                <dt>CELL</dt>
                <dd>
                  ({hover.agent.x}, {hover.agent.y}) {hover.agent.inBar ? "· IN BAR" : ""}
                </dd>
              </div>
              <div>
                <dt>STATED / TARGET</dt>
                <dd>
                  {hover.agent.stated || "—"} / {hover.agent.target || "—"}
                </dd>
              </div>
              <div>
                <dt>UTILITY</dt>
                <dd>{fmt(hover.agent.utility, 2)}</dd>
              </div>
              <div>
                <dt>CUMULATIVE</dt>
                <dd>{fmt(hover.agent.cumulative, 2)}</dd>
              </div>
            </dl>
            {hover.agent.broadcast && (
              <p className="inspector__said">
                <span>SAID</span>“{hover.agent.broadcast}”
              </p>
            )}
            {hover.agent.mind && (
              <div className="inspector__mind">
                <p className="inspector__archetype">
                  {hover.agent.mind.archetype}
                  <span>
                    MOOD {hover.agent.mind.mood.toUpperCase()}
                    {hover.agent.mind.mood !== "calm" ? ` ${fmt(hover.agent.mind.moodIntensity, 2)}` : ""}
                  </span>
                </p>
                <dl>
                  {hover.agent.mind.traits && (
                    <div>
                      <dt>HONESTY / RISK</dt>
                      <dd>
                        {fmt(hover.agent.mind.traits.honesty, 2)} / {fmt(hover.agent.mind.traits.risk_tolerance, 2)}
                      </dd>
                    </div>
                  )}
                  <div>
                    <dt>FORECAST</dt>
                    <dd>
                      {fmt(hover.agent.mind.forecast, 0)} · {hover.agent.mind.predictor ?? "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>REPUTATION</dt>
                    <dd>{fmt(hover.agent.mind.reputation, 2)}</dd>
                  </div>
                  <div>
                    <dt>INSTINCT</dt>
                    <dd>
                      {Object.entries(hover.agent.mind.instinct)
                        .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
                        .slice(0, 2)
                        .map(([k, v]) => `${STRATEGY_LABEL[k as keyof typeof STRATEGY_LABEL].split(" ·")[0]} ${fmt(v ?? 0, 2)}`)
                        .join(" · ")}
                    </dd>
                  </div>
                </dl>
                {hover.agent.mind.note && (
                  <p className="inspector__thought" data-lie={hover.agent.strategy === "false_go" || hover.agent.strategy === "false_stay"}>
                    <span>{hover.agent.strategy === "false_go" || hover.agent.strategy === "false_stay" ? "THOUGHT · LIED" : "THOUGHT"}</span>“
                    {hover.agent.mind.note}”
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
