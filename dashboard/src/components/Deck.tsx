import { useMemo, type ChangeEvent } from "react";
import { fmt, fmtPct } from "../format";
import { CONDITION_COLOR, STRATEGIES, STRATEGY_SHORT, type GameView, type TrialTrace } from "../model";
import type { AnalyticsReport, ConditionSeries } from "../types";
import LineChart, { type ChartReference, type ChartSeries } from "./LineChart";
import StrategyStrip from "./StrategyStrip";

interface DeckProps {
  trace: TrialTrace | null;
  report: AnalyticsReport | null;
  game: GameView;
  condition: string | null;
  cursorEpoch: number | null;
  frameCount: number;
  cursorIndex: number;
  onCursor: (index: number) => void;
  playing: boolean;
  onPlaying: (value: boolean) => void;
  speed: number;
  onSpeed: (value: number) => void;
  mode: "live" | "replay" | "none";
  view: "plot" | "table";
  onView: (value: "plot" | "table") => void;
}

const SPEEDS = [1, 2, 4, 8];

function align(target: number[], source: number[], values: (number | null)[] | undefined): (number | null)[] {
  if (!values) return target.map(() => null);
  const position = new Map(source.map((x, i) => [x, i]));
  return target.map((x) => {
    const i = position.get(x);
    return i == null ? null : values[i] ?? null;
  });
}

function conditionSeries(
  report: AnalyticsReport | null,
  key: string,
  x: number[],
): ChartSeries[] {
  const out: ChartSeries[] = [];
  for (const label of ["control", "delta2"]) {
    const series = report?.conditions?.[label]?.series as ConditionSeries | null | undefined;
    if (!series) continue;
    const mean = series[`${key}_mean`];
    const sd = series[`${key}_sd`];
    const values = align(x, series.epoch, mean);
    const spread = align(x, series.epoch, sd);
    out.push({
      id: label,
      label: label === "control" ? "CONTROL μ" : "DELTA 2 μ",
      color: CONDITION_COLOR[label],
      values,
      band: {
        lo: values.map((v, i) => (v != null && spread[i] != null ? v - (spread[i] as number) : null)),
        hi: values.map((v, i) => (v != null && spread[i] != null ? v + (spread[i] as number) : null)),
      },
      width: 1.75,
    });
  }
  return out;
}

export default function Deck({
  trace,
  report,
  game,
  condition,
  cursorEpoch,
  frameCount,
  cursorIndex,
  onCursor,
  playing,
  onPlaying,
  speed,
  onSpeed,
  mode,
  view,
  onView,
}: DeckProps) {
  const x = useMemo(() => {
    const fromReport = report?.conditions?.control?.series?.epoch ?? report?.conditions?.delta2?.series?.epoch ?? [];
    const fromTrace = trace?.epochs ?? [];
    return [...new Set([...fromReport, ...fromTrace])].sort((a, b) => a - b);
  }, [report, trace]);

  const traceSeries = (values: (number | null)[] | undefined): ChartSeries[] =>
    trace && values
      ? [{ id: "trial", label: "THIS TRIAL", color: "#ebebeb", values: align(x, trace.epochs, values), width: 1.25, opacity: 0.9 }]
      : [];

  const scrub = (epoch: number) => {
    if (mode !== "replay" || !trace) return;
    const i = trace.epochs.indexOf(epoch);
    if (i >= 0) {
      onPlaying(false);
      onCursor(i);
    }
  };

  const attendanceRefs: ChartReference[] = [
    { y: game.primaryAttendance, label: `A*=${game.primaryAttendance}`, color: "#ffffff" },
    { y: game.threshold, label: `T=${game.threshold}${game.congestion ? "" : " UNREACHABLE"}`, color: "#ffd700", dashed: true },
  ];

  const onRange = (event: ChangeEvent<HTMLInputElement>) => {
    onPlaying(false);
    onCursor(Number(event.target.value));
  };

  const live = mode === "live";
  const progress = frameCount > 1 ? (cursorIndex / (frameCount - 1)) * 100 : 0;

  return (
    <section className="panel deck" style={{ ["--stagger" as string]: 4 }} aria-label="Timeline and telemetry">
      <div className="timeline">
        <div className="timeline__controls">
          <button type="button" className="btn" onClick={() => onCursor(0)} disabled={live || frameCount === 0} aria-label="First epoch">
            |◀
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => onCursor(Math.max(0, cursorIndex - 1))}
            disabled={live || frameCount === 0}
            aria-label="Previous epoch"
          >
            ◀
          </button>
          <button
            type="button"
            className="btn btn--primary"
            aria-pressed={playing}
            onClick={() => onPlaying(!playing)}
            disabled={live || frameCount < 2}
            aria-label={playing ? "Pause" : "Play"}
          >
            {playing ? "❚❚ PAUSE" : "▶ PLAY"}
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => onCursor(Math.min(frameCount - 1, cursorIndex + 1))}
            disabled={live || frameCount === 0}
            aria-label="Next epoch"
          >
            ▶
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => onCursor(Math.max(0, frameCount - 1))}
            disabled={live || frameCount === 0}
            aria-label="Last epoch"
          >
            ▶|
          </button>
        </div>
        <input
          className="scrub"
          type="range"
          min={0}
          max={Math.max(0, frameCount - 1)}
          step={1}
          value={Math.min(cursorIndex, Math.max(0, frameCount - 1))}
          onChange={onRange}
          disabled={live || frameCount < 2}
          aria-label="Epoch"
          style={{ ["--progress" as string]: `${live ? 100 : progress}%` }}
        />
        <span className="timeline__epoch">
          {live ? "LIVE" : "EPOCH"} <strong>{cursorEpoch != null ? cursorEpoch : "—"}</strong>
        </span>
        <div className="segmented" role="group" aria-label="Playback speed">
          {SPEEDS.map((s) => (
            <button key={s} type="button" className="btn btn--seg" aria-pressed={speed === s} onClick={() => onSpeed(s)} disabled={live}>
              {s}×
            </button>
          ))}
        </div>
        <div className="segmented" role="group" aria-label="View">
          <button type="button" className="btn btn--seg" aria-pressed={view === "plot"} onClick={() => onView("plot")}>
            PLOT
          </button>
          <button type="button" className="btn btn--seg" aria-pressed={view === "table"} onClick={() => onView("table")}>
            TABLE
          </button>
        </div>
      </div>

      {view === "plot" ? (
        <div className="charts">
          <LineChart
            index="A"
            title="ATTENDANCE VS NASH"
            unit="AGENTS"
            x={x}
            series={[...conditionSeries(report, "bar_attendance", x), ...traceSeries(trace?.attendance)]}
            references={attendanceRefs}
            cursor={cursorEpoch}
            yMin={0}
            digits={1}
            onScrub={scrub}
          />
          <LineChart
            index="B"
            title="DECEPTION INDEX"
            unit="SHARE"
            x={x}
            series={[...conditionSeries(report, "deception_index", x), ...traceSeries(trace?.deception)]}
            cursor={cursorEpoch}
            yMin={0}
            digits={3}
            onScrub={scrub}
          />
          <LineChart
            index="C"
            title="ε-NASH REGRET"
            unit="UTILITY"
            x={x}
            series={[...conditionSeries(report, "regret", x), ...traceSeries(trace?.regret)]}
            cursor={cursorEpoch}
            yMin={0}
            digits={4}
            onScrub={scrub}
          />
          <LineChart
            index="D"
            title="STRATEGY DRIFT // KL"
            unit="NATS"
            x={x}
            series={[...conditionSeries(report, "kl_step", x), ...traceSeries(trace?.klStep)]}
            cursor={cursorEpoch}
            yMin={0}
            digits={4}
            onScrub={scrub}
          />
          <StrategyStrip trace={trace} cursor={cursorEpoch} condition={condition} onScrub={scrub} />
        </div>
      ) : (
        <div className="tablewrap">
          {trace ? (
            <table className="datatable">
              <thead>
                <tr>
                  <th scope="col">EPOCH</th>
                  <th scope="col">IN BAR</th>
                  <th scope="col">DECEPTION</th>
                  <th scope="col">REGRET</th>
                  <th scope="col">KL STEP</th>
                  {STRATEGIES.map((s) => (
                    <th scope="col" key={s}>
                      {STRATEGY_SHORT[s]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {trace.epochs.map((epoch, i) => (
                  <tr key={epoch} aria-current={epoch === cursorEpoch ? "true" : undefined}>
                    <th scope="row">{epoch}</th>
                    <td>{trace.attendance[i]}</td>
                    <td>{fmt(trace.deception[i], 3)}</td>
                    <td>{fmt(trace.regret[i], 4)}</td>
                    <td>{fmt(trace.klStep[i], 4)}</td>
                    {STRATEGIES.map((s) => (
                      <td key={s}>{fmtPct(trace.shares[s][i], 0)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="empty">NO TRACE SELECTED</p>
          )}
        </div>
      )}
    </section>
  );
}
