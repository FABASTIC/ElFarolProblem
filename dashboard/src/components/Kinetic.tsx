import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from "react";

export function GooDefs() {
  return (
    <svg className="goo-defs" width="0" height="0" aria-hidden="true" focusable="false">
      <defs>
        <filter id="elf-goo" x="-20%" y="-50%" width="140%" height="200%" colorInterpolationFilters="sRGB">
          <feGaussianBlur in="SourceGraphic" stdDeviation="2.6" result="blur" />
          <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7" result="goo" />
          <feComposite in="SourceGraphic" in2="goo" operator="atop" />
        </filter>
        <filter id="elf-goo-soft" x="-30%" y="-30%" width="160%" height="160%" colorInterpolationFilters="sRGB">
          <feGaussianBlur in="SourceGraphic" stdDeviation="7" result="blur" />
          <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 22 -10" result="goo" />
          <feComposite in="SourceGraphic" in2="goo" operator="atop" />
        </filter>
      </defs>
    </svg>
  );
}

export interface SegmentOption<T extends string | number> {
  id: T;
  label: ReactNode;
  title?: string;
}

interface SegmentedProps<T extends string | number> {
  options: SegmentOption<T>[];
  value: T | null;
  onChange: (id: T) => void;
  label: string;
  kind?: "radio" | "tab";
  disabled?: boolean;
  className?: string;
}

export function Segmented<T extends string | number>({ options, value, onChange, label, kind = "radio", disabled, className }: SegmentedProps<T>) {
  const root = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<{ x: number; w: number } | null>(null);
  const signature = options.map((o) => String(o.id)).join("|");

  useLayoutEffect(() => {
    const container = root.current;
    if (!container) return;
    const update = () => {
      const active = container.querySelector<HTMLElement>('[data-active="true"]');
      setBox(active ? { x: active.offsetLeft, w: active.offsetWidth } : null);
    };
    update();
    const resize = new ResizeObserver(update);
    resize.observe(container);
    return () => resize.disconnect();
  }, [value, signature]);

  return (
    <div ref={root} className={`seg ${className ?? ""}`} role={kind === "tab" ? "tablist" : "radiogroup"} aria-label={label} data-disabled={disabled || undefined}>
      <span className="seg__goo" aria-hidden="true">
        {box && (
          <>
            <i className="seg__blob" style={{ ["--x" as string]: `${box.x}px`, ["--w" as string]: `${box.w}px` } as CSSProperties} />
            <i className="seg__blob seg__blob--trail" style={{ ["--x" as string]: `${box.x}px`, ["--w" as string]: `${box.w}px` } as CSSProperties} />
          </>
        )}
      </span>
      {options.map((option) => {
        const active = option.id === value;
        return (
          <button
            key={String(option.id)}
            type="button"
            role={kind}
            aria-checked={kind === "radio" ? active : undefined}
            aria-selected={kind === "tab" ? active : undefined}
            data-active={active}
            className="seg__opt"
            disabled={disabled}
            title={option.title}
            onClick={() => onChange(option.id)}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export function HoverTrail({ container, selector }: { container: RefObject<HTMLElement | null>; selector: string }) {
  const trail = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const root = container.current;
    const element = trail.current;
    if (!root || !element) return;
    const onOver = (event: PointerEvent) => {
      if (event.pointerType !== "mouse") return;
      const item = (event.target as HTMLElement | null)?.closest<HTMLElement>(selector);
      if (!item || !root.contains(item)) return;
      const rect = item.getBoundingClientRect();
      const base = root.getBoundingClientRect();
      element.style.setProperty("--ty", `${rect.top - base.top}px`);
      element.style.setProperty("--tx", `${rect.left - base.left}px`);
      element.style.setProperty("--tw", `${rect.width}px`);
      element.style.setProperty("--th", `${rect.height}px`);
      element.dataset.on = "true";
    };
    const onLeave = () => {
      element.dataset.on = "false";
    };
    root.addEventListener("pointerover", onOver);
    root.addEventListener("pointerleave", onLeave);
    root.addEventListener("scroll", onLeave, { capture: true, passive: true });
    return () => {
      root.removeEventListener("pointerover", onOver);
      root.removeEventListener("pointerleave", onLeave);
      root.removeEventListener("scroll", onLeave, { capture: true });
    };
  }, [container, selector]);

  return <span ref={trail} className="trail" data-on="false" aria-hidden="true" />;
}

export function CopyButton({ text, label = "Copy command" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!done) return;
    const id = window.setTimeout(() => setDone(false), 1600);
    return () => window.clearTimeout(id);
  }, [done]);

  return (
    <button
      type="button"
      className="copybtn"
      data-done={done}
      aria-label={done ? "Copied" : label}
      title={done ? "Copied" : label}
      onClick={() => {
        void navigator.clipboard
          ?.writeText(text)
          .then(() => setDone(true))
          .catch(() => undefined);
      }}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <g className="copybtn__sheets">
          <rect x="5.5" y="5.5" width="8" height="8" />
          <path d="M3 10.5 V2.5 H10.5" />
        </g>
        <path className="copybtn__check" d="M3 8.5 L6.5 12 L13 4.5" pathLength={1} />
      </svg>
    </button>
  );
}
