import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AbstractIntro from "./components/AbstractIntro";
import AgentCard from "./components/AgentCard";
import BootOverlay from "./components/BootOverlay";
import Boundary from "./components/Boundary";
import Bubbles from "./components/Bubbles";
import Deck from "./components/Deck";
import { CrierCard, PopulationCard, RunsCard, TonightCard } from "./components/HudCards";
import MetricsRail from "./components/MetricsRail";
import NightTimeline from "./components/NightTimeline";
import SetupPanel from "./components/SetupPanel";
import SweepRail from "./components/SweepRail";
import TopBar, { type FeedHealth, type View, type WorldMode } from "./components/TopBar";
import { ACTIVE_STATES, useLauncher } from "./launcher";
import {
  RUNNING_STATES,
  appendLiveFrame,
  cumulativeRange,
  frameFromLive,
  gameView,
  mergeTrials,
  previewFrame,
  traceFromFrames,
  traceFromLive,
  type Frame,
} from "./model";
import type { Lens } from "./people";
import World from "./scene/World";
import { createMotion } from "./scene/People";
import type { CrowdState } from "./scene/Tavern";
import { RECOMMENDED, sanitizeConfig, type LaunchConfig } from "./setup";
import { usePolledJson, useReplay } from "./telemetry";
import type { AnalyticsReport, ComparisonRow, LiveState } from "./types";

const STALE_AFTER_S = 900;
const DRAFT_KEY = "elfarol.setup.draft";
const ABSTRACT_EXIT_MS = 1150;

type IntroState = "open" | "leaving" | "closed";

const STATUS: Record<string, { label: string; tone: string }> = {
  launching: { label: "STARTING", tone: "warn" },
  booting: { label: "BOOTING", tone: "warn" },
  loading_engine: { label: "LOADING ENGINE", tone: "warn" },
  running: { label: "RUNNING", tone: "live" },
  stopping: { label: "STOPPING", tone: "warn" },
  analyzing: { label: "ANALYZING", tone: "warn" },
  complete: { label: "COMPLETE", tone: "good" },
  failed: { label: "FAILED", tone: "fail" },
  interrupted: { label: "INTERRUPTED", tone: "warn" },
  stalled: { label: "STALLED", tone: "fail" },
  standby: { label: "STANDBY", tone: "idle" },
};

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function loadDraft(): LaunchConfig {
  try {
    const raw = window.localStorage.getItem(DRAFT_KEY);
    const parsed = raw ? sanitizeConfig(JSON.parse(raw)).config : null;
    return parsed ? { ...parsed, engine: RECOMMENDED.engine } : { ...RECOMMENDED, seeds: [...RECOMMENDED.seeds] };
  } catch {
    return { ...RECOMMENDED, seeds: [...RECOMMENDED.seeds] };
  }
}

export default function App() {
  const launcher = useLauncher();
  const [intro, setIntro] = useState<IntroState>("open");
  const enterTown = useCallback(() => setIntro("leaving"), []);
  useEffect(() => {
    if (intro !== "leaving") return;
    const id = window.setTimeout(() => setIntro("closed"), ABSTRACT_EXIT_MS);
    return () => window.clearTimeout(id);
  }, [intro]);
  const launcherState = launcher.status?.state ?? "idle";
  const launcherActive = ACTIVE_STATES.has(launcherState);
  const [hot, setHot] = useState(false);

  const comparison = usePolledJson<ComparisonRow[]>("comparison.json", hot ? 2500 : 5000);
  const report = usePolledJson<AnalyticsReport>("analytics_report.json", hot ? 4000 : 10000);
  const live = usePolledJson<LiveState>("live_state.json", hot || launcherActive ? 900 : 3000);

  const liveState = live.data;
  const liveFresh = !!liveState && Date.now() / 1000 - liveState.updated_at < STALE_AFTER_S;
  const liveRunning = !!liveState && RUNNING_STATES.has(liveState.status) && liveFresh;
  const liveKey = liveRunning ? liveState?.current?.trial_dir ?? null : null;
  const running = liveRunning || launcherActive;

  useEffect(() => setHot(running), [running]);

  const trials = useMemo(() => mergeTrials(comparison.data, liveState, report.data, liveKey), [comparison.data, liveState, report.data, liveKey]);

  const [view, setView] = useState<View>("world");
  const [follow, setFollow] = useState(true);
  const [picked, setPicked] = useState<string | null>(null);
  const lastLiveKey = useRef<string | null>(null);

  useEffect(() => {
    if (liveKey) {
      lastLiveKey.current = liveKey;
    } else if (lastLiveKey.current && follow) {
      setPicked(lastLiveKey.current);
      lastLiveKey.current = null;
    }
  }, [liveKey, follow]);

  const fallbackKey = trials.find((t) => t.status === "complete" || t.status === "resumed")?.key ?? trials[0]?.key ?? null;
  const selectedKey = follow && liveKey ? liveKey : picked && trials.some((t) => t.key === picked) ? picked : fallbackKey;
  const selected = trials.find((t) => t.key === selectedKey) ?? null;
  const liveView = selectedKey != null && selectedKey === liveKey;

  const liveFrameNow = useMemo(() => (liveRunning ? frameFromLive(liveState?.current) : null), [liveRunning, liveState]);
  const [liveFrames, setLiveFrames] = useState<{ key: string | null; frames: Frame[] }>({ key: null, frames: [] });
  useEffect(() => {
    if (!liveKey || !liveFrameNow) return;
    setLiveFrames((prev) => (prev.key === liveKey ? { key: liveKey, frames: appendLiveFrame(prev.frames, liveFrameNow) } : { key: liveKey, frames: [liveFrameNow] }));
  }, [liveKey, liveFrameNow]);

  const replayKey = !liveView && selected && !["pending", "running"].includes(selected.status) ? selected.key : null;
  const replay = useReplay(replayKey, selected?.revision ?? "");
  const watched = liveFrames.key != null && liveFrames.key === selectedKey ? liveFrames.frames : [];
  const frames = liveView ? watched : replay.frames.length ? replay.frames : replay.state === "loading" ? [] : watched;

  const gridHint = report.data?.trials.find((t) => t.trial === selectedKey)?.grid_size ?? null;
  const game = useMemo(() => gameView(report.data, liveState, gridHint), [report.data, liveState, gridHint]);

  const [draft, setDraftState] = useState<LaunchConfig>(loadDraft);
  const setDraft = useCallback((next: LaunchConfig) => {
    setDraftState(next);
    try {
      window.localStorage.setItem(DRAFT_KEY, JSON.stringify(next));
    } catch {
      return;
    }
  }, []);
  const [setupOpen, setSetupOpen] = useState(true);
  useEffect(() => {
    if (running) setSetupOpen(false);
  }, [running]);
  const showSetup = !running && (setupOpen || trials.length === 0);

  const [cursor, setCursor] = useState(0);
  const [atEdge, setAtEdge] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [deckView, setDeckView] = useState<"plot" | "table">("plot");

  useEffect(() => {
    setCursor(0);
    setAtEdge(true);
    setPlaying(!prefersReducedMotion());
  }, [replayKey]);

  useEffect(() => {
    if (!playing || liveView || frames.length < 2) return;
    if (cursor >= frames.length - 1) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setCursor((c) => Math.min(frames.length - 1, c + 1)), 1100 / speed);
    return () => window.clearTimeout(id);
  }, [playing, cursor, liveView, frames.length, speed]);

  const index = liveView && atEdge ? frames.length - 1 : Math.min(cursor, Math.max(0, frames.length - 1));
  const frame = frames[index] ?? null;

  const bootConfig = launcherActive ? launcher.status?.config ?? null : null;
  const stagedMembers = bootConfig?.members ?? draft.members;
  const stagedSeed = bootConfig?.seeds[0] ?? draft.seeds[0] ?? 42;
  const preview = useMemo(
    () => previewFrame(Math.min(256, Math.max(1, stagedMembers)), stagedSeed, game.gridSize, game.barMin, game.barMax),
    [stagedMembers, stagedSeed, game.gridSize, game.barMin, game.barMax],
  );

  const starting = launcherState === "launching" || (launcherState === "running" && !liveKey && follow);
  const winding = launcherState === "stopping" || launcherState === "analyzing";
  let mode: WorldMode;
  if (showSetup) mode = "setup";
  else if (liveView && frame) mode = "live";
  else if (liveView || starting) mode = "boot";
  else if (frames.length) mode = "replay";
  else if (running && !winding) mode = "boot";
  else mode = "empty";

  const worldFrame = mode === "setup" || mode === "boot" ? preview : frame;
  const frameKey = mode === "setup" || mode === "boot" ? `preview:${stagedMembers}:${stagedSeed}` : `${selectedKey ?? "none"}:${frame?.epoch ?? "none"}`;
  const attendance = worldFrame && !worldFrame.preview ? worldFrame.agents.filter((a) => a.inBar).length : null;
  const crowd: CrowdState = attendance == null ? "idle" : attendance > game.threshold ? "crowded" : "comfortable";
  const fill = attendance == null ? 0 : attendance / Math.max(1, game.threshold);

  const heightRange = useMemo(() => cumulativeRange(frames), [frames]);
  const wealthSpan = Math.max(Math.abs(heightRange[0]), Math.abs(heightRange[1]), 1);
  const trace = useMemo(
    () => (liveView ? traceFromLive(liveState?.current, game) : frames.length ? traceFromFrames(frames, game) : null),
    [liveView, liveState, frames, game],
  );

  const totalNights =
    mode === "setup" ? draft.epochs : mode === "boot" ? bootConfig?.epochs ?? liveState?.sweep.num_epochs ?? draft.epochs : liveView ? liveState?.current?.epochs ?? frames.length : selected?.epochs ?? (liveState?.trials.some((t) => t.trial_dir === selectedKey) ? liveState.sweep.num_epochs : frames.length);
  const nightAttendance = useMemo(() => {
    const out: (number | null)[] = [];
    if (mode === "setup" || mode === "boot" || !trace) return out;
    trace.epochs.forEach((epoch, i) => {
      out[epoch] = trace.attendance[i];
    });
    return out;
  }, [mode, trace]);
  const available = useMemo(() => new Set(frames.map((f) => f.epoch)), [frames]);
  const cursorEpoch = mode === "live" || mode === "replay" ? frame?.epoch ?? null : null;

  const [lens, setLens] = useState<Lens>("strategy");
  const [isolate, setIsolate] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);
  const hoveredRef = useRef<number | null>(null);
  const motion = useRef(createMotion());
  const overlay = useRef<HTMLDivElement>(null);
  const [resetToken, setResetToken] = useState(0);
  const [speech, setSpeech] = useState(true);
  const reduced = useMemo(() => prefersReducedMotion(), []);
  const worldRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = worldRef.current;
    if (!element || reduced) return;
    const target = { x: 0, y: 0 };
    const current = { x: 0, y: 0 };
    let frame = 0;
    const tick = () => {
      current.x += (target.x - current.x) * 0.06;
      current.y += (target.y - current.y) * 0.06;
      element.style.setProperty("--mx", current.x.toFixed(4));
      element.style.setProperty("--my", current.y.toFixed(4));
      frame = Math.abs(target.x - current.x) + Math.abs(target.y - current.y) > 0.0005 ? requestAnimationFrame(tick) : 0;
    };
    const aim = (x: number, y: number) => {
      target.x = x;
      target.y = y;
      if (!frame) frame = requestAnimationFrame(tick);
    };
    const onMove = (event: PointerEvent) => {
      if (event.pointerType !== "mouse") return;
      if ((event.target as HTMLElement | null)?.closest(".hud, .world__tools, .boot")) return;
      aim((event.clientX / window.innerWidth) * 2 - 1, (event.clientY / window.innerHeight) * 2 - 1);
    };
    const onLeave = () => aim(0, 0);
    element.addEventListener("pointermove", onMove, { passive: true });
    element.addEventListener("pointerleave", onLeave);
    return () => {
      cancelAnimationFrame(frame);
      element.removeEventListener("pointermove", onMove);
      element.removeEventListener("pointerleave", onLeave);
    };
  }, [reduced, view]);

  const inSetup = mode === "setup";
  useEffect(() => {
    setSelectedId(null);
    setIsolate(null);
  }, [selectedKey, inSetup]);

  const selectedAgent = selectedId != null && worldFrame && !worldFrame.preview ? worldFrame.agents.find((a) => a.id === selectedId) ?? null : null;

  const onSeek = useCallback(
    (epoch: number) => {
      const i = frames.findIndex((f) => f.epoch === epoch);
      if (i < 0) return;
      setPlaying(false);
      setCursor(i);
      if (liveView) setAtEdge(i === frames.length - 1);
    },
    [frames, liveView],
  );

  const onStep = useCallback(
    (delta: number) => {
      setPlaying(false);
      const next = Math.max(0, Math.min(frames.length - 1, index + delta));
      setCursor(next);
      if (liveView) setAtEdge(next === frames.length - 1);
    },
    [frames.length, index, liveView],
  );

  const onPlaying = useCallback(
    (value: boolean) => {
      if (value && frames.length > 1 && index >= frames.length - 1) setCursor(0);
      setPlaying(value);
    },
    [frames.length, index],
  );

  const onSelectTrial = useCallback(
    (key: string) => {
      setPicked(key);
      setFollow(key === liveKey);
      setSetupOpen(false);
    },
    [liveKey],
  );

  const onBegin = useCallback(
    async (config: LaunchConfig) => {
      const ok = await launcher.launch(config);
      if (ok) {
        setSetupOpen(false);
        setFollow(true);
        setPicked(null);
        setLiveFrames({ key: null, frames: [] });
      }
    },
    [launcher],
  );

  const onNew = useCallback(() => {
    launcher.clearError();
    setSetupOpen(true);
    setView("world");
  }, [launcher]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (intro === "open") return;
      if (target?.closest("input, textarea, select, [contenteditable='true'], .chart")) return;
      if (event.key === "Escape") {
        if (selectedId != null) setSelectedId(null);
        else if (showSetup && trials.length && !running) setSetupOpen(false);
        return;
      }
      if (showSetup) return;
      if (event.key === " ") {
        if (target?.closest("button")) return;
        event.preventDefault();
        onPlaying(!playing);
      } else if (event.key === "ArrowRight") {
        onStep(1);
      } else if (event.key === "ArrowLeft") {
        onStep(-1);
      } else if (event.key === "Home") {
        setCursor(0);
        setAtEdge(false);
      } else if (event.key === "End") {
        setCursor(Math.max(0, frames.length - 1));
        setAtEdge(true);
      } else if (event.key.toLowerCase() === "f" && !event.ctrlKey && !event.metaKey) {
        setFollow((f) => !f);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frames.length, intro, onPlaying, onStep, playing, running, selectedId, showSetup, trials.length]);

  const feeds: FeedHealth[] = [
    { name: "LIVE", file: "live_state.json", state: live.state, receivedAt: live.receivedAt },
    { name: "SWEEP", file: "comparison.json", state: comparison.state, receivedAt: comparison.receivedAt },
    { name: "ANALYTICS", file: "analytics_report.json", state: report.state, receivedAt: report.receivedAt },
  ];

  const statusKey =
    launcherState === "stopping" || launcherState === "analyzing" || launcherState === "launching"
      ? launcherState
      : liveState
        ? RUNNING_STATES.has(liveState.status) && !liveFresh
          ? "stalled"
          : liveState.status
        : running
          ? "launching"
          : launcherState === "failed"
            ? "failed"
            : "standby";
  const status = STATUS[statusKey] ?? { label: statusKey.toUpperCase(), tone: "idle" };
  const model = liveState?.model ?? report.data?.trials.find((t) => t.model_name)?.model_name ?? null;

  const calibration = useMemo(() => {
    const agents = liveState?.sweep.num_agents ?? report.data?.trials[0]?.agents ?? null;
    if (!agents || !comparison.data) return null;
    const rows = comparison.data.filter((r) => (r.runtime?.engine_build_s ?? 0) > 0 && r.epoch_timings_mean_s);
    if (!rows.length) return null;
    return rows.reduce((acc, r) => acc + (r.epoch_timings_mean_s as number), 0) / rows.length / agents;
  }, [comparison.data, liveState, report.data]);

  const night = mode === "live" || mode === "replay" ? (frame ? frame.epoch + 1 : null) : null;
  const runLog = launcher.status?.log ?? [];

  return (
    <div className="app" data-view={view}>
      <TopBar
        view={view}
        onView={setView}
        model={model}
        statusLabel={status.label}
        statusTone={status.tone}
        night={night}
        nights={totalNights || null}
        trial={mode === "setup" ? null : selected}
        mode={mode}
        elapsed={running ? liveState?.elapsed_s ?? (launcher.status?.startedAt ? Date.now() / 1000 - launcher.status.startedAt : null) : null}
        eta={running ? liveState?.eta_s ?? null : null}
        feeds={feeds}
        canStop={running && launcherState !== "analyzing"}
        stopping={launcherState === "stopping" || launcher.pending}
        onStop={() => void launcher.stop()}
        canNew={!running && !showSetup}
        onNew={onNew}
        onAbstract={() => setIntro("open")}
      />
      {intro !== "closed" && <AbstractIntro leaving={intro === "leaving"} onEnter={enterTown} />}

      {view === "world" ? (
        <main className="world" data-mode={mode} ref={worldRef}>
          <div className="world__sky" aria-hidden="true">
            <svg className="world__rings" viewBox="-500 -500 1000 1000">
              {[150, 250, 360, 480].map((r) => (
                <circle key={r} r={r} />
              ))}
              <line x1={-1400} y1={0} x2={1400} y2={0} />
              <line x1={0} y1={-1400} x2={0} y2={1400} strokeDasharray="2 7" />
              <circle className="world__rings-dot" cx={-480} cy={0} r={2.5} />
              <circle className="world__rings-dot" cx={480} cy={0} r={2.5} />
            </svg>
          </div>
          <Boundary
            resetKey={selectedKey ?? "none"}
            fallback={(error, retry) => (
              <div className="world__fault" role="alert">
                <p className="eyebrow">RENDERER OFFLINE</p>
                <p>{error.message || "WebGL context unavailable"}</p>
                <button type="button" className="btn" onClick={retry}>
                  RESTART RENDERER
                </button>
              </div>
            )}
          >
            <div className="world__canvas">
              {intro !== "open" && (
              <World
                frame={worldFrame}
                frameKey={frameKey}
                gridSize={game.gridSize}
                barMin={game.barMin}
                barMax={game.barMax}
                crowd={crowd}
                fill={fill}
                lens={lens}
                isolate={isolate}
                wealthSpan={wealthSpan}
                selectedId={selectedId}
                hoveredId={hoveredRef}
                motion={motion}
                overlay={overlay}
                resetToken={resetToken}
                drift={mode === "setup"}
                reduced={reduced}
                onHover={setHoverId}
                onSelect={setSelectedId}
              />
              )}
            </div>
          </Boundary>
          <div className="world__labels" ref={overlay} aria-hidden="true">
            <Bubbles frame={worldFrame} selectedId={selectedId} hoveredId={hoverId} speech={speech} />
          </div>
          <div className="world__vignette" aria-hidden="true" />

          <div className="hud hud--left">
            <TonightCard frame={worldFrame} game={game} mode={mode} members={stagedMembers} />
            {mode !== "setup" && (
              <PopulationCard frame={worldFrame} lens={lens} onLens={(next) => { setLens(next); setIsolate(null); }} isolate={isolate} onIsolate={setIsolate} staged={mode === "boot" || mode === "empty"} />
            )}
            {mode !== "setup" && (
              <CrierCard frame={worldFrame} log={runLog} events={liveState?.events ?? []} selectedId={selectedId} onSelect={setSelectedId} />
            )}
          </div>

          <div className="hud hud--right">
            {mode === "setup" ? (
              <SetupPanel
                draft={draft}
                onDraft={setDraft}
                launcher={launcher}
                calibration={calibration}
                canClose={trials.length > 0}
                onClose={() => setSetupOpen(false)}
                onBegin={(config) => void onBegin(config)}
              />
            ) : selectedAgent ? (
              <AgentCard
                agent={selectedAgent}
                frames={frames}
                index={index}
                threshold={game.threshold}
                members={worldFrame?.agents.length ?? game.n}
                onClose={() => setSelectedId(null)}
              />
            ) : (
              <RunsCard trials={trials} selectedKey={selectedKey} onSelect={onSelectTrial} follow={follow} onFollow={setFollow} liveRunning={liveRunning} />
            )}
          </div>

          {mode === "boot" && (
            <BootOverlay
              launcher={launcher.status}
              liveStatus={liveState && liveFresh ? liveState.status : null}
              livePhase={liveState?.current?.phase ?? null}
              onBack={() => {
                launcher.clearError();
                setSetupOpen(true);
              }}
            />
          )}
          {mode === "empty" && (
            <div className="boot boot--empty" role="status">
              <p className="eyebrow">NOTHING TO SHOW</p>
              <h2>{replay.state === "loading" ? "Loading this town…" : "This town left no trace on disk"}</h2>
              <button type="button" className="btn btn--go" onClick={onNew}>
                + NEW SIMULATION
              </button>
            </div>
          )}

          <div className="world__tools">
            <button type="button" className="btn btn--tiny" aria-pressed={speech} onClick={() => setSpeech((s) => !s)}>
              SPEECH BUBBLES
            </button>
            <button type="button" className="btn btn--tiny" onClick={() => setResetToken((t) => t + 1)}>
              RESET VIEW
            </button>
            <span>DRAG ORBIT · RIGHT-DRAG PAN · WHEEL ZOOM</span>
          </div>

          <div className="hud hud--bottom">
            <NightTimeline
              total={totalNights || 1}
              attendance={nightAttendance}
              available={available}
              cursorEpoch={cursorEpoch}
              threshold={game.threshold}
              members={worldFrame?.agents.length || stagedMembers}
              mode={mode}
              playing={playing && mode === "replay"}
              onPlaying={onPlaying}
              speed={speed}
              onSpeed={setSpeed}
              onSeek={onSeek}
              onStep={onStep}
              atEdge={atEdge}
              onEdge={() => setAtEdge(true)}
            />
          </div>
        </main>
      ) : (
        <main className="lab">
          <SweepRail
            live={liveState}
            liveRunning={liveRunning}
            trials={trials}
            selectedKey={selectedKey}
            onSelect={onSelectTrial}
            follow={follow && liveRunning}
            onFollow={setFollow}
            frame={frame}
          />
          <Deck
            trace={trace}
            report={report.data}
            game={game}
            condition={selected?.condition ?? null}
            cursorEpoch={cursorEpoch}
            frameCount={frames.length}
            cursorIndex={index}
            onCursor={(i) => {
              setCursor(i);
              if (liveView) setAtEdge(i >= frames.length - 1);
            }}
            playing={playing && !liveView && frames.length > 1}
            onPlaying={onPlaying}
            speed={speed}
            onSpeed={setSpeed}
            mode={liveView ? "live" : frames.length ? "replay" : "none"}
            view={deckView}
            onView={setDeckView}
          />
          <MetricsRail report={report.data} reportState={report.state} game={game} />
        </main>
      )}
    </div>
  );
}
