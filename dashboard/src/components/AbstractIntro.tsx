import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import CrowdScene from "../scene/CrowdScene";
import "../abstract.css";

interface AbstractIntroProps {
  leaving: boolean;
  onEnter: () => void;
}

const CHAPTERS = ["OPENING", "PROBLEM", "PARADOX", "AGENTS", "TENSORS", "THE NIGHT", "MEASURED", "ENTER"];

function RollText({ text, className }: { text: string; className?: string }) {
  return (
    <span className={`roll ${className ?? ""}`} aria-label={text}>
      {Array.from(text).map((ch, i) => (
        <span key={i} className="roll__char" style={{ ["--i" as string]: i } as CSSProperties} aria-hidden="true">
          <span>{ch === " " ? " " : ch}</span>
          <span>{ch === " " ? " " : ch}</span>
        </span>
      ))}
    </span>
  );
}

function Lines({ children, delay = 0 }: { children: ReactNode[]; delay?: number }) {
  return (
    <>
      {children.map((line, i) => (
        <span key={i} className="mask" data-reveal style={{ ["--d" as string]: `${delay + i * 90}ms` } as CSSProperties}>
          <span className="mask__in">{line}</span>
        </span>
      ))}
    </>
  );
}

function Seat({ x, y, state }: { x: number; y: number; state: "empty" | "comfort" | "over" }) {
  if (state === "over") {
    return (
      <g className="seat seat--over">
        <line x1={x - 4} y1={y - 4} x2={x + 4} y2={y + 4} />
        <line x1={x + 4} y1={y - 4} x2={x - 4} y2={y + 4} />
      </g>
    );
  }
  return <circle className={`seat seat--${state}`} cx={x} cy={y} r={state === "comfort" ? 5 : 4.2} />;
}

function BarDiagram({ progress }: { progress: number }) {
  const filled = Math.round(Math.min(1, Math.max(0, progress)) * 100);
  const seats = Array.from({ length: 100 }, (_, i) => {
    const state = i < filled ? (i < 60 ? "comfort" : "over") : "empty";
    return <Seat key={i} x={40 + (i % 10) * 28} y={40 + Math.floor(i / 10) * 28} state={state} />;
  });
  return (
    <svg className="diagram" viewBox="0 0 360 400" role="img" aria-label="One bar with one hundred seats; sixty are comfortable">
      <rect className="diagram__frame" x={20} y={20} width={300} height={300} />
      <line className="diagram__axis" x1={170} y1={0} x2={170} y2={340} />
      {seats}
      <line className="diagram__rule" x1={20} y1={360} x2={320} y2={360} />
      {[0, 20, 40, 60, 80, 100].map((v) => (
        <g key={v}>
          <line className="diagram__tick" x1={20 + v * 3} y1={354} x2={20 + v * 3} y2={366} />
          <text className="diagram__label" x={20 + v * 3} y={384} textAnchor="middle">
            {v}
          </text>
        </g>
      ))}
      <line className="diagram__line" x1={200} y1={340} x2={200} y2={372} />
      <text className="diagram__label diagram__label--lamp" x={206} y={350}>
        T 60
      </text>
      <text className="diagram__count" x={340} y={30} textAnchor="end">
        {String(filled).padStart(3, "0")}
      </text>
    </svg>
  );
}

const SCHEMATIC = [44, 71, 38, 66, 79, 41, 55, 73, 47, 62, 58, 69, 52, 64, 57, 61, 59, 63, 56, 60];

function Oscillation() {
  const w = 1000;
  const h = 260;
  const x = (i: number) => (i / (SCHEMATIC.length - 1)) * w;
  const y = (v: number) => h - (v / 100) * h;
  const d = SCHEMATIC.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  return (
    <svg className="osc" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label="Schematic: attendance swings above and below the comfort line">
      <line className="osc__line" x1={0} y1={y(60)} x2={w} y2={y(60)} />
      <path className="osc__path" d={d} pathLength={1} />
      {SCHEMATIC.map((v, i) => (
        <circle key={i} className="osc__dot" data-over={v > 60} cx={x(i)} cy={y(v)} r={3.2} style={{ ["--k" as string]: i / SCHEMATIC.length } as CSSProperties} />
      ))}
    </svg>
  );
}

function TensorStack() {
  const layers = 14;
  return (
    <div className="stack" aria-hidden="true">
      {Array.from({ length: layers }, (_, i) => (
        <div key={i} className="stack__plane" data-hot={i === 9} style={{ ["--z" as string]: i - (layers - 1) / 2 } as CSSProperties}>
          {Array.from({ length: 48 }, (_, j) => (
            <i key={j} style={{ opacity: 0.18 + (((i * 31 + j * 17) % 23) / 23) * 0.7 } as CSSProperties} />
          ))}
          <em>{i === 9 ? "W · agent 042" : `W · ${String(i * 7 + 1).padStart(3, "0")}`}</em>
        </div>
      ))}
    </div>
  );
}

function Rings() {
  return (
    <svg className="rings" viewBox="-500 -500 1000 1000" aria-hidden="true">
      {[120, 200, 290, 390, 480].map((r) => (
        <circle key={r} r={r} />
      ))}
      <line x1={-500} y1={0} x2={500} y2={0} />
      <line x1={0} y1={-500} x2={0} y2={500} strokeDasharray="2 6" />
      <circle className="rings__dot" cx={-480} cy={0} r={3} />
      <circle className="rings__dot" cx={480} cy={0} r={3} />
      <text x={-470} y={-10}>N / 100</text>
      <text x={400} y={-10}>T / 060</text>
      <text x={8} y={-392}>W / 000</text>
      <text x={8} y={404}>ρ / 0.85</text>
    </svg>
  );
}

const NIGHT = [
  { k: "01", title: "BROADCAST", body: "Each agent reads its own history and tells the town 1 or 0: going, or staying home." },
  { k: "02", title: "TOWN RATIO", body: "The claims are averaged into town_broadcast_ratio. That number is all anyone hears." },
  { k: "03", title: "ACTION", body: "Every brain decides for real, from its history, the ratio, and the promise it just made." },
  { k: "04", title: "CREDIT", body: "A bluff that thins tomorrow's crowd is paid back through replay. Lying is learnable." },
];

export default function AbstractIntro({ leaving, onEnter }: AbstractIntroProps) {
  const root = useRef<HTMLDivElement>(null);
  const [chapter, setChapter] = useState(0);
  const [barProgress, setBarProgress] = useState(0);
  const [ratio, setRatio] = useState(0);

  const enter = useCallback(() => {
    if (!leaving) onEnter();
  }, [leaving, onEnter]);

  useEffect(() => {
    const el = root.current;
    if (!el) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const sections = Array.from(el.querySelectorAll<HTMLElement>("[data-chapter]"));
    let target = el.scrollTop;
    let current = el.scrollTop;
    let animating = false;
    let frame = 0;

    const measure = () => {
      const top = el.scrollTop;
      const view = el.clientHeight;
      let active = 0;
      sections.forEach((section, i) => {
        const span = Math.max(1, section.offsetHeight - view);
        const p = Math.min(1, Math.max(0, (top - section.offsetTop) / span));
        section.style.setProperty("--p", p.toFixed(4));
        if (top + view * 0.5 >= section.offsetTop) active = i;
        if (section.dataset.chapter === "PROBLEM") setBarProgress(p);
        if (section.dataset.chapter === "THE NIGHT") setRatio(p);
      });
      const total = Math.max(1, el.scrollHeight - view);
      el.style.setProperty("--g", (top / total).toFixed(4));
      setChapter(active);
    };

    const step = () => {
      current += (target - current) * 0.085;
      if (Math.abs(target - current) < 0.5) {
        current = target;
        animating = false;
      }
      el.scrollTop = current;
      measure();
      if (animating) frame = requestAnimationFrame(step);
    };

    const onWheel = (event: WheelEvent) => {
      if (reduced || event.ctrlKey) return;
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 32 : event.deltaMode === 2 ? el.clientHeight : 1;
      target = Math.min(el.scrollHeight - el.clientHeight, Math.max(0, target + event.deltaY * unit));
      if (!animating) {
        animating = true;
        current = el.scrollTop;
        frame = requestAnimationFrame(step);
      }
    };

    const onScroll = () => {
      if (!animating) {
        target = el.scrollTop;
        current = el.scrollTop;
        measure();
      }
    };

    const reveal = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            (entry.target as HTMLElement).dataset.in = "true";
            reveal.unobserve(entry.target);
          }
        });
      },
      { root: el, threshold: 0.15 },
    );
    el.querySelectorAll("[data-reveal]").forEach((node) => reveal.observe(node));

    measure();
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", measure);
    el.focus({ preventScroll: true });
    return () => {
      cancelAnimationFrame(frame);
      reveal.disconnect();
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", measure);
    };
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") enter();
      if (event.key === "Enter" && chapter === CHAPTERS.length - 1) enter();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chapter, enter]);

  return (
    <div className="abs" ref={root} data-leaving={leaving} tabIndex={-1} role="dialog" aria-label="El Farol abstract">
      <header className="abs__chrome abs__chrome--top">
        <span className="abs__mark">
          <i aria-hidden="true" />
          EL FAROL
          <span className="abs__mark-sub">
            <em>⁄</em> ABSTRACT
          </span>
        </span>
        <span className="abs__meta">ISOLATED PYTORCH TENSORS · RTX 4060</span>
        <button type="button" className="abs__skip" onClick={enter}>
          <RollText text="SKIP TO THE TOWN" className="abs__skip-long" />
          <RollText text="SKIP" className="abs__skip-short" />
          <span className="abs__arrow" aria-hidden="true">
            →
          </span>
        </button>
      </header>

      <div className="abs__rail" aria-hidden="true">
        <i />
      </div>

      <nav className="abs__pill" aria-label="Chapter">
        <span>§ {String(chapter).padStart(2, "0")}</span>
        <i aria-hidden="true" />
        <strong>{CHAPTERS[chapter]}</strong>
        <span className="abs__pill-of">/ {String(CHAPTERS.length - 1).padStart(2, "0")}</span>
      </nav>

      <section className="sec sec--hero" data-chapter="OPENING">
        <div className="stage">
          <div className="hero__depth">
            <Rings />
          </div>
          <p className="micro hero__top">
            <span>N° 01 — AN ABSTRACT</span>
            <span>SANTA FE, NEW MEXICO · 1994 → 2026</span>
          </p>
          <h1 className="hero__word" aria-label="El Farol">
            {Array.from("EL FAROL").map((ch, i) =>
              ch === " " ? (
                <span key={i} className="hero__gap" aria-hidden="true" />
              ) : (
                <span key={i} className="hero__letter" style={{ ["--i" as string]: i, ["--k" as string]: [0.6, 1.4, 0, 0.9, 1.8, 0.7, 1.2, 1.6][i] } as CSSProperties} aria-hidden="true">
                  <span>{ch}</span>
                </span>
              ),
            )}
          </h1>
          <div className="hero__under">
            <p className="hero__lede">
              <span className="mask mask--auto" style={{ ["--d" as string]: "900ms" } as CSSProperties}>
                <span className="mask__in">The bar problem, retold by</span>
              </span>
              <span className="mask mask--auto" style={{ ["--d" as string]: "990ms" } as CSSProperties}>
                <span className="mask__in">
                  <em className="serif">a hundred separate minds.</em>
                </span>
              </span>
            </p>
            <p className="micro hero__scroll">
              <span>SCROLL</span>
              <i aria-hidden="true" />
            </p>
          </div>
        </div>
      </section>

      <section className="sec sec--problem" data-chapter="PROBLEM">
        <div className="stage split">
          <div className="split__text">
            <p className="micro" data-reveal>
              § 01 — THE PROBLEM
            </p>
            <h2 className="editorial">
              <Lines>
                {[
                  <>
                    One hundred people. <span className="glyph">◯</span>
                  </>,
                  <>One bar. Sixty comfortable</>,
                  <>seats. Every Thursday,</>,
                  <>
                    <em className="serif">each of them decides alone</em>
                  </>,
                  <>whether to go.</>,
                ]}
              </Lines>
            </h2>
            <p className="footnote" data-reveal style={{ ["--d" as string]: "400ms" } as CSSProperties}>
              If sixty or fewer show up, everyone at the bar has a good night. If more show up, the bar is packed and they wish they had stayed home.
              After W. Brian Arthur, <em className="serif">Inductive Reasoning and Bounded Rationality</em>, 1994.
            </p>
          </div>
          <div className="split__figure">
            <BarDiagram progress={barProgress} />
            <p className="micro figure__cap">
              <span>FIG. 01</span>
              <span>SEATS FILL AS YOU SCROLL · ○ COMFORT · ✕ OVERCROWDED</span>
            </p>
          </div>
        </div>
      </section>

      <section className="sec sec--paradox" data-chapter="PARADOX">
        <div className="stage paradox">
          <p className="micro" data-reveal>
            § 02 — THE PARADOX
          </p>
          <h2 className="editorial editorial--wide">
            <Lines>
              {[
                <>
                  If everyone expects a quiet night, <em className="serif">the bar overflows.</em>
                </>,
                <>
                  If everyone expects a crowd, <em className="serif">it stands empty.</em>
                </>,
                <>Any forecast that everyone shares defeats itself.</>,
              ]}
            </Lines>
          </h2>
          <div className="paradox__chart">
            <Oscillation />
            <p className="micro figure__cap">
              <span>FIG. 02 — SCHEMATIC</span>
              <span>ATTENDANCE AROUND THE COMFORT LINE · ● ABOVE 60</span>
            </p>
          </div>
        </div>
      </section>

      <section className="sec sec--agents" data-chapter="AGENTS">
        <div className="stage agents">
          <div className="agents__head">
            <p className="micro" data-reveal>
              § 03 — WHAT WE BUILT
            </p>
            <h2 className="editorial">
              <Lines>
                {[
                  <>No shared model. No central server.</>,
                  <>
                    So we gave every patron <em className="serif">a mind of its own.</em>
                  </>,
                ]}
              </Lines>
            </h2>
          </div>
          <dl className="stats">
            {[
              ["100", "AGENTS, SCALABLE TO ANY N"],
              ["2", "NETWORKS PER AGENT"],
              ["4,006", "PARAMETERS PER BRAIN"],
              ["0", "SHARED WEIGHTS"],
            ].map(([value, label], i) => (
              <div key={label} data-reveal style={{ ["--d" as string]: `${i * 80}ms` } as CSSProperties}>
                <dt>
                  <RollText text={label} />
                </dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
          <div className="agents__crowd">
            <CrowdScene />
          </div>
        </div>
      </section>

      <section className="sec sec--tensors" data-chapter="TENSORS">
        <div className="stage tensors">
          <div className="tensors__text">
            <p className="micro" data-reveal>
              § 04 — HOW
            </p>
            <h2 className="editorial">
              <Lines>
                {[
                  <>Every brain is a set of</>,
                  <>
                    <em className="serif">isolated PyTorch tensors.</em>
                  </>,
                ]}
              </Lines>
            </h2>
            <p className="footnote" data-reveal style={{ ["--d" as string]: "300ms" } as CSSProperties}>
              Its own weights. Its own Adam optimizer. Its own replay memory. Nothing is pooled. On a single RTX 4060 the hundred brains are stacked into
              one batched matrix multiply, and gradients never cross from one agent to another.
            </p>
            <pre className="code" data-reveal style={{ ["--d" as string]: "450ms" } as CSSProperties}>
              <span>
                <b>W</b> = stack(W<sub>i</sub> for i in N) <i>→ [N, H, D]</i>
              </span>
              <span>
                <b>q</b> = baddbmm(b, x, W.mT) <i>→ [N, B, A]</i>
              </span>
              <span>
                <b>loss</b>.sum().backward() <i>→ isolated grads</i>
              </span>
              <span>
                <b>opt</b>
                <sub>i</sub>.step() for i in N <i>→ N optimizers</i>
              </span>
            </pre>
          </div>
          <div className="tensors__figure">
            <TensorStack />
            <p className="micro figure__cap">
              <span>FIG. 03</span>
              <span>ONE PLANE PER AGENT · EXPLODED BY SCROLL</span>
            </p>
          </div>
        </div>
      </section>

      <section className="sec sec--night" data-chapter="THE NIGHT">
        <div className="stage night">
          <div className="night__head">
            <p className="micro" data-reveal>
              § 05 — ONE NIGHT, TWO STEPS
            </p>
            <div className="night__ratio">
              <span className="micro">town_broadcast_ratio</span>
              <strong>{(0.85 * ratio).toFixed(2)}</strong>
              <span className="night__bar" aria-hidden="true">
                <i style={{ width: `${85 * ratio}%` }} />
              </span>
            </div>
          </div>
          <ol className="night__steps">
            {NIGHT.map((row, i) => (
              <li key={row.k} data-lit={ratio > i * 0.22 + 0.04}>
                <span className="night__k">{row.k}</span>
                <span className="night__title">
                  <RollText text={row.title} />
                </span>
                <span className="night__body">{row.body}</span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="sec sec--measured" data-chapter="MEASURED">
        <div className="stage measured">
          <p className="micro" data-reveal>
            § 06 — MEASURED
          </p>
          <p className="measured__big" data-reveal>
            47.3<span>s</span>
          </p>
          <p className="footnote measured__cap" data-reveal style={{ ["--d" as string]: "200ms" } as CSSProperties}>
            Two towns × one hundred agents × fifty nights, every brain trained after every night, on one laptop GPU.
          </p>
          <dl className="ledger" data-reveal style={{ ["--d" as string]: "350ms" } as CSSProperties}>
            <div>
              <dt>BRAINS ONLINE</dt>
              <dd>≈ 1 s</dd>
            </div>
            <div>
              <dt>COMFORT LINE</dt>
              <dd>60 / 100</dd>
            </div>
            <div>
              <dt>CONDITIONS</dt>
              <dd>SILENT · BROADCAST</dd>
            </div>
            <div>
              <dt>OUTPUT</dt>
              <dd>comparison.json · CSV</dd>
            </div>
          </dl>
        </div>
      </section>

      <section className="sec sec--enter" data-chapter="ENTER">
        <div className="stage enter">
          <div className="enter__rings" aria-hidden="true">
            <i />
            <i />
            <i />
          </div>
          <p className="micro" data-reveal>
            § 07 — THE TOWN IS STAGED
          </p>
          <button type="button" className="enter__button" onClick={enter}>
            <RollText text="ENTER THE TOWN" className="enter__roll" />
            <span className="enter__arrow" aria-hidden="true">
              <i />
            </span>
          </button>
          <p className="micro enter__hint">PRESS ENTER · ESC SKIPS ANYTIME</p>
        </div>
      </section>
    </div>
  );
}
