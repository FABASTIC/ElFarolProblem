import { useEffect, useState, type CSSProperties, type RefObject } from "react";

interface ScrollProgressProps {
  target: RefObject<HTMLElement | null>;
  watch?: string | number;
}

export default function ScrollProgress({ target, watch }: ScrollProgressProps) {
  const [state, setState] = useState({ p: 0, overflow: false });

  useEffect(() => {
    const element = target.current;
    if (!element) return;
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const max = element.scrollHeight - element.clientHeight;
        const next = { p: max > 1 ? Math.min(1, Math.max(0, element.scrollTop / max)) : 0, overflow: max > 1 };
        setState((prev) => (prev.overflow === next.overflow && Math.abs(prev.p - next.p) < 0.002 ? prev : next));
      });
    };
    measure();
    element.addEventListener("scroll", measure, { passive: true });
    const resize = new ResizeObserver(measure);
    resize.observe(element);
    return () => {
      cancelAnimationFrame(frame);
      element.removeEventListener("scroll", measure);
      resize.disconnect();
    };
  }, [target, watch]);

  if (!state.overflow) return null;

  return (
    <div className="scrollprog" aria-hidden="true" style={{ ["--sp" as string]: state.p.toFixed(4) } as CSSProperties}>
      <span className="scrollprog__track">
        <i />
      </span>
      <span className="scrollprog__value">{String(Math.round(state.p * 100)).padStart(3, "0")}</span>
    </div>
  );
}
