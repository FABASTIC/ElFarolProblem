import { useEffect, useRef, useState, type CSSProperties } from "react";

export const FLAP_DIGITS = " 0123456789";
export const FLAP_TEXT = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/·:%";

interface CellProps {
  target: string;
  charset: string;
  delay: number;
  speed: number;
  reduced: boolean;
}

function FlapCell({ target, charset, delay, speed, reduced }: CellProps) {
  const [current, setCurrent] = useState(target);
  const [previous, setPrevious] = useState(target);
  const [flip, setFlip] = useState(0);
  const now = useRef(target);

  useEffect(() => {
    if (reduced) {
      now.current = target;
      setPrevious(target);
      setCurrent(target);
      return;
    }
    if (now.current === target) return;
    let timer: number | undefined;
    const start = window.setTimeout(() => {
      const step = () => {
        const from = Math.max(0, charset.indexOf(now.current));
        const goal = charset.indexOf(target);
        const next = goal < 0 ? target : charset[(from + 1) % charset.length];
        setPrevious(now.current);
        now.current = next;
        setCurrent(next);
        setFlip((value) => value + 1);
        if (next !== target) timer = window.setTimeout(step, speed);
      };
      step();
    }, delay);
    return () => {
      window.clearTimeout(start);
      window.clearTimeout(timer);
    };
  }, [target, charset, delay, speed, reduced]);

  const show = (ch: string) => (ch === " " ? " " : ch);

  return (
    <span className="flap" style={{ ["--flip" as string]: `${speed}ms` } as CSSProperties} aria-hidden="true">
      <span className="flap__half flap__half--top">
        <span>{show(current)}</span>
      </span>
      <span className="flap__half flap__half--bottom">
        <span>{show(previous)}</span>
      </span>
      {flip > 0 && (
        <span key={flip} className="flap__leaf">
          <span className="flap__half flap__half--top flap__leaf-front">
            <span>{show(previous)}</span>
          </span>
          <span className="flap__half flap__half--bottom flap__leaf-back">
            <span>{show(current)}</span>
          </span>
        </span>
      )}
    </span>
  );
}

interface SplitFlapProps {
  text: string;
  columns?: number;
  charset?: string;
  size?: "sm" | "md" | "lg";
  accentColor?: string;
  showIndicators?: boolean;
  staggerDelay?: number;
  flipSpeed?: number;
  align?: "left" | "right";
  className?: string;
  label?: string;
}

export default function SplitFlap({
  text,
  columns,
  charset = FLAP_TEXT,
  size = "md",
  accentColor,
  showIndicators = false,
  staggerDelay = 30,
  flipSpeed = 55,
  align = "left",
  className,
  label,
}: SplitFlapProps) {
  const [reduced] = useState(() => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const upper = text.toUpperCase();
  const width = columns ?? upper.length;
  const padded = align === "right" ? upper.padStart(width, " ").slice(-width) : upper.padEnd(width, " ").slice(0, width);

  return (
    <span
      className={["splitflap", `splitflap--${size}`, className].filter(Boolean).join(" ")}
      role="img"
      aria-label={label ?? text}
      style={accentColor ? ({ ["--flap-accent" as string]: accentColor } as CSSProperties) : undefined}
      data-indicators={showIndicators || undefined}
    >
      {Array.from(padded).map((ch, i) => (
        <FlapCell key={i} target={ch} charset={charset} delay={i * staggerDelay} speed={flipSpeed} reduced={reduced} />
      ))}
    </span>
  );
}
