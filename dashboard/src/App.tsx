import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ArenaPanel from "./components/ArenaPanel";
import Deck from "./components/Deck";
import Header, { type FeedHealth } from "./components/Header";
import MetricsRail from "./components/MetricsRail";
import Stage from "./components/Stage";
import SweepRail from "./components/SweepRail";
import {
  RUNNING_STATES,
  cumulativeRange,
  frameFromLive,
  gameView,
  mergeTrials,
  traceFromFrames,
  traceFromLive,
} from "./model";
import { usePolledJson, useReplay } from "./telemetry";
import type { AnalyticsReport, ComparisonRow, LiveState } from "./types";

const STALE_AFTER_S = 900;

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export default function App() {
  const comparison = usePolledJson<ComparisonRow[]>("comparison.json", 4000);
  const report = usePolledJson<AnalyticsReport>("analytics_report.json", 12000);
  const live = usePolledJson<LiveState>("live_state.json", 1500);

  const liveState = live.data;
  const liveFresh = !!liveState && Date.now() / 1000 - liveState.updated_at < STALE_AFTER_S;
  const liveRunning = !!liveState && RUNNING_STATES.has(liveState.status) && liveFresh;
  const liveKey = liveRunning ? liveState?.current?.trial_dir ?? null : null;
  const displayStatus = liveState ? (RUNNING_STATES.has(liveState.status) && !liveFresh ? "stalled" : liveState.status) : null;

  const trials = useMemo(
    () => mergeTrials(comparison.data, liveState, report.data, liveKey),
    [comparison.data, liveState, report.data, liveKey],
  );

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

  const liveFrame = useMemo(() => (liveView ? frameFromLive(liveState?.current) : null), [liveView, liveState]);
  const replayKey = !liveView && selected && !["pending", "running"].includes(selected.status) ? selected.key : null;
  const replay = useReplay(replayKey, selected?.revision ?? "");
  const frames = replay.frames;

  const gridHint = report.data?.trials.find((t) => t.trial === selectedKey)?.grid_size ?? null;
  const game = useMemo(() => gameView(report.data, liveState, gridHint), [report.data, liveState, gridHint]);

  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [view, setView] = useState<"plot" | "table">("plot");

  useEffect(() => {
    setCursor(0);
    setPlaying(!prefersReducedMotion());
  }, [replayKey]);

  useEffect(() => {
    if (!playing || liveView || frames.length < 2) return;
    if (cursor >= frames.length - 1) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setCursor((c) => Math.min(frames.length - 1, c + 1)), 1000 / speed);
    return () => window.clearTimeout(id);
  }, [playing, cursor, liveView, frames.length, speed]);

  const safeCursor = Math.min(cursor, Math.max(0, frames.length - 1));
  const frame = liveView ? liveFrame : frames[safeCursor] ?? null;
  const frameKey = `${selectedKey ?? "none"}:${frame?.epoch ?? "none"}`;
  const heightRange = useMemo<[number, number]>(
    () => cumulativeRange(liveView ? (liveFrame ? [liveFrame] : []) : frames),
    [liveView, liveFrame, frames],
  );
  const trace = useMemo(
    () => (liveView ? traceFromLive(liveState?.current, game) : frames.length ? traceFromFrames(frames, game) : null),
    [liveView, liveState, frames, game],
  );

  const mode: "live" | "replay" | "none" = liveView ? "live" : frames.length ? "replay" : "none";
  const cursorEpoch = liveView ? liveState?.current?.epoch ?? null : frame?.epoch ?? null;
  const liveEpochs = liveState?.current?.epochs;
  const lastEpoch = liveView ? (liveEpochs ? liveEpochs - 1 : null) : frames.length ? frames[frames.length - 1].epoch : null;

  const onPlaying = useCallback(
    (value: boolean) => {
      if (value && frames.length > 1 && safeCursor >= frames.length - 1) setCursor(0);
      setPlaying(value);
    },
    [frames.length, safeCursor],
  );

  const onSelect = useCallback(
    (key: string) => {
      setPicked(key);
      setFollow(key === liveKey);
    },
    [liveKey],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true'], .chart")) return;
      if (event.key === " ") {
        if (target?.closest("button")) return;
        event.preventDefault();
        setPlaying((p) => !p);
      } else if (event.key === "ArrowRight") {
        setPlaying(false);
        setCursor((c) => Math.min(Math.max(0, frames.length - 1), c + 1));
      } else if (event.key === "ArrowLeft") {
        setPlaying(false);
        setCursor((c) => Math.max(0, c - 1));
      } else if (event.key === "Home") {
        setCursor(0);
      } else if (event.key === "End") {
        setCursor(Math.max(0, frames.length - 1));
      } else if (event.key.toLowerCase() === "f" && !event.ctrlKey && !event.metaKey) {
        setFollow((f) => !f);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frames.length]);

  const feeds: FeedHealth[] = [
    { name: "LIVE", file: "live_state.json", state: live.state, receivedAt: live.receivedAt },
    { name: "SWEEP", file: "comparison.json", state: comparison.state, receivedAt: comparison.receivedAt },
    { name: "ANALYTICS", file: "analytics_report.json", state: report.state, receivedAt: report.receivedAt },
  ];
  const model = liveState?.model ?? report.data?.trials.find((t) => t.model_name)?.model_name ?? null;

  return (
    <Stage>
      <div className="shell">
        <Header model={model} status={displayStatus} elapsed={liveState?.elapsed_s ?? null} eta={liveState?.eta_s ?? null} feeds={feeds} />
        <SweepRail
          live={liveState}
          liveRunning={liveRunning}
          trials={trials}
          selectedKey={selectedKey}
          onSelect={onSelect}
          follow={follow && liveRunning}
          onFollow={setFollow}
          frame={frame}
        />
        <ArenaPanel
          frame={frame}
          frameKey={frameKey}
          game={game}
          heightRange={heightRange}
          trial={selected}
          mode={mode}
          phase={liveView ? liveState?.current?.phase ?? null : null}
          epoch={cursorEpoch}
          lastEpoch={lastEpoch}
          replayState={replay.state}
        />
        <MetricsRail report={report.data} reportState={report.state} game={game} />
        <Deck
          trace={trace}
          report={report.data}
          game={game}
          condition={selected?.condition ?? null}
          cursorEpoch={cursorEpoch}
          frameCount={frames.length}
          cursorIndex={safeCursor}
          onCursor={setCursor}
          playing={playing && !liveView && frames.length > 1}
          onPlaying={onPlaying}
          speed={speed}
          onSpeed={setSpeed}
          mode={mode}
          view={view}
          onView={setView}
        />
      </div>
    </Stage>
  );
}
