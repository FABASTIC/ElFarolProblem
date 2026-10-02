import { fmt, fmtP, fmtSigned, isNum } from "../format";
import { CONDITION_COLOR, type GameView } from "../model";
import type { FeedState } from "../telemetry";
import type { AnalyticsReport, ContrastMetric } from "../types";

interface MetricsRailProps {
  report: AnalyticsReport | null;
  reportState: FeedState;
  game: GameView;
}

interface TileSpec {
  section: string;
  metric: string;
  title: string;
  caption: string;
  digits: number;
}

const TILES: TileSpec[] = [
  { section: "gini", metric: "final", title: "GINI // SUCCESS", caption: "MAX-NORMALISED GINI · FINAL CUMULATIVE UTILITY", digits: 4 },
  { section: "drift", metric: "kl_step_mean", title: "KL // STRATEGY DRIFT", caption: "MEAN D_KL(p_t || p_t-1) · NATS", digits: 4 },
  { section: "nash", metric: "mean_abs_gap", title: "NASH // |A − A*|", caption: "MEAN ABSOLUTE ATTENDANCE GAP · AGENTS", digits: 2 },
  { section: "nash", metric: "mean_regret", title: "ε-NASH // REGRET", caption: "MEAN EX-POST DEVIATION GAIN · UTILITY", digits: 4 },
];

function CiGlyph({ estimate, lo, hi }: { estimate: number | null; lo: number | null; hi: number | null }) {
  const width = 132;
  const height = 16;
  const values = [estimate, lo, hi, 0].filter(isNum);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min || 1) * 0.12;
  const sx = (v: number) => 4 + ((v - (min - pad)) / (max - min + 2 * pad)) * (width - 8);
  return (
    <svg className="ci" width={width} height={height} role="img" aria-label={`95% interval ${fmt(lo)} to ${fmt(hi)}`}>
      <line x1={4} x2={width - 4} y1={height / 2} y2={height / 2} className="ci__axis" />
      <line x1={sx(0)} x2={sx(0)} y1={2} y2={height - 2} className="ci__zero" />
      {isNum(lo) && isNum(hi) && <line x1={sx(lo)} x2={sx(hi)} y1={height / 2} y2={height / 2} className="ci__span" />}
      {isNum(lo) && <line x1={sx(lo)} x2={sx(lo)} y1={4} y2={height - 4} className="ci__span" />}
      {isNum(hi) && <line x1={sx(hi)} x2={sx(hi)} y1={4} y2={height - 4} className="ci__span" />}
      {isNum(estimate) && <circle cx={sx(estimate)} cy={height / 2} r={3.5} className="ci__dot" />}
    </svg>
  );
}

function Tile({ spec, metric, index }: { spec: TileSpec; metric: ContrastMetric | null | undefined; index: number }) {
  return (
    <article className="tile" style={{ ["--stagger" as string]: index }}>
      <header className="tile__head">
        <h3>{spec.title}</h3>
        <span>{spec.caption}</span>
      </header>
      {metric ? (
        <>
          <dl className="tile__groups">
            {(["control", "delta2"] as const).map((group) => (
              <div key={group}>
                <dt>
                  <i className="linekey" style={{ background: CONDITION_COLOR[group] }} aria-hidden="true" />
                  {group === "control" ? "CONTROL" : "DELTA 2"}
                </dt>
                <dd>
                  {fmt(metric[group].mean, spec.digits)}
                  <small>{isNum(metric[group].sd) ? ` ±${fmt(metric[group].sd, spec.digits)}` : ""}</small>
                </dd>
              </div>
            ))}
          </dl>
          <div className="tile__delta">
            <span className="tile__estimate">Δ {fmtSigned(metric.estimate, spec.digits)}</span>
            <CiGlyph estimate={metric.estimate} lo={metric.ci95[0]} hi={metric.ci95[1]} />
          </div>
          <p className="tile__foot">
            CI95 [{fmtSigned(metric.ci95[0], spec.digits)}, {fmtSigned(metric.ci95[1], spec.digits)}] · p {fmtP(metric.p_value)} ·{" "}
            {metric.design.toUpperCase()} n={metric.delta2.n}
          </p>
        </>
      ) : (
        <p className="empty">AWAITING BOTH CONDITIONS</p>
      )}
    </article>
  );
}

export default function MetricsRail({ report, reportState, game }: MetricsRailProps) {
  const contrast = report?.contrast ?? null;
  return (
    <aside className="panel rail rail--right" style={{ ["--stagger" as string]: 2 }} aria-label="Analytics">
      <section className="section">
        <h2 className="section__title">
          <strong>04 // GAME</strong>
          <span>{game.source === "analyzer" ? "ANALYZER" : "DERIVED"}</span>
        </h2>
        <dl className="gamecard">
          <div>
            <dt>N AGENTS</dt>
            <dd>{game.n}</dd>
          </div>
          <div>
            <dt>T COMFORT</dt>
            <dd>{game.threshold}</dd>
          </div>
          <div>
            <dt>A* PURE</dt>
            <dd>{game.pureAttendance.join(" / ") || "—"}</dd>
          </div>
          <div>
            <dt>p* MIXED</dt>
            <dd>{fmt(game.mixedProbability, 4)}</dd>
          </div>
          <div>
            <dt>DOMINANT</dt>
            <dd>{game.dominant ? game.dominant.toUpperCase() : "NONE"}</dd>
          </div>
          <div>
            <dt>CONGESTION</dt>
            <dd data-tone={game.congestion ? undefined : "warn"}>{game.congestion ? "POSSIBLE" : "IMPOSSIBLE"}</dd>
          </div>
        </dl>
        <p className="payoffs">
          PAYOFF · BAR≤T {fmtSigned(game.payoffs.bar_comfortable, 1)} · BAR&gt;T {fmtSigned(game.payoffs.bar_overcrowded, 1)} · HOME{" "}
          {fmtSigned(game.payoffs.home, 1)}
        </p>
      </section>

      <section className="section section--grow">
        <h2 className="section__title">
          <strong>05 // CONTRAST</strong>
          <span>{contrast ? contrast.label.toUpperCase() : "—"}</span>
        </h2>
        {report ? (
          <div className="tiles">
            {TILES.map((spec, i) => (
              <Tile key={`${spec.section}.${spec.metric}`} spec={spec} metric={contrast?.metrics?.[spec.section]?.[spec.metric]} index={i} />
            ))}
          </div>
        ) : (
          <p className="empty">
            {reportState === "missing" ? "NO analytics_report.json // python3 analyzer.py" : "CONNECTING TO ANALYZER FEED…"}
          </p>
        )}
      </section>

      {report && Object.values(report.conditions).some((c) => c.minds) && (
        <section className="section">
          <h2 className="section__title">
            <strong>06 // MINDS</strong>
            <span>LIE RATE BY ARCHETYPE</span>
          </h2>
          <dl className="gamecard">
            {(["control", "delta2"] as const).map((group) => {
              const minds = report.conditions[group]?.minds;
              return (
                <div key={group}>
                  <dt>
                    <i className="linekey" style={{ background: CONDITION_COLOR[group] }} aria-hidden="true" />
                    {group === "control" ? "CONTROL" : "DELTA 2"}
                  </dt>
                  <dd>{minds ? fmt(minds.lie_rate.mean, 3) : "—"}</dd>
                </div>
              );
            })}
            <div>
              <dt>HONESTY~LIE ρ</dt>
              <dd>{fmt(report.conditions.delta2?.minds?.honesty_lie_spearman.mean ?? report.conditions.control?.minds?.honesty_lie_spearman.mean, 2)}</dd>
            </div>
          </dl>
          <div className="archbars">
            {[...new Set(Object.values(report.conditions).flatMap((c) => Object.keys(c.minds?.lie_rate_by_archetype ?? {})))].sort().map((name) => {
              const control = report.conditions.control?.minds?.lie_rate_by_archetype?.[name] ?? null;
              const treatment = report.conditions.delta2?.minds?.lie_rate_by_archetype?.[name] ?? null;
              return (
                <div className="archbar" key={name}>
                  <span>{name}</span>
                  <span className="archbar__track">
                    <span className="archbar__fill" style={{ width: `${(control ?? 0) * 100}%`, background: CONDITION_COLOR.control }} />
                    <span className="archbar__fill" style={{ width: `${(treatment ?? 0) * 100}%`, background: CONDITION_COLOR.delta2 }} />
                  </span>
                  <span className="archbar__value">
                    {fmt(control, 2)} / {fmt(treatment, 2)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="payoffs">
            REPUTATION GAP (HONEST − LIARS) {fmtSigned(report.conditions.delta2?.minds?.reputation_gap_honest_minus_liars.mean, 3)} · ISOLATED PYTORCH TENSORS FOLLOW INSTINCT{" "}
            {fmt(report.conditions.delta2?.minds?.instinct_agreement.mean ?? report.conditions.control?.minds?.instinct_agreement.mean, 2)}
          </p>
        </section>
      )}

      {report && report.warnings.length > 0 && (
        <section className="section section--warnings">
          <h2 className="section__title">
            <strong>07 // FLAGS</strong>
            <span>{report.warnings.length}</span>
          </h2>
          <ul className="warnings">
            {report.warnings.map((warning) => (
              <li key={warning}>
                <span className="warnings__icon" aria-hidden="true">
                  !
                </span>
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
      <footer className="rail__foot">
        {report ? `REPORT ${report.generated_at} · ${report.source.trials_analyzed}/${report.source.trials_discovered} TRIALS` : "REPORT —"}
      </footer>
    </aside>
  );
}
