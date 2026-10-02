import { useEffect, useRef, useState, type CSSProperties, type RefObject } from "react";

export interface DockHandle {
  update: (position: number, progress: number[]) => void;
}

interface SectionDockProps {
  chapters: string[];
  active: number;
  handle: RefObject<DockHandle | null>;
  onGo: (index: number) => void;
}

const ROW = 30;
const ORB = 52;

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

export default function SectionDock({ chapters, active, handle, onGo }: SectionDockProps) {
  const [open, setOpen] = useState(false);
  const items = useRef<(HTMLButtonElement | null)[]>([]);
  const label = useRef<HTMLSpanElement>(null);
  const [labelWidth, setLabelWidth] = useState(120);
  const closer = useRef<number | undefined>(undefined);

  useEffect(() => {
    handle.current = {
      update: (_position, progress) => {
        items.current.forEach((item, i) => {
          item?.style.setProperty("--fill", (progress[i] ?? 0).toFixed(4));
        });
      },
    };
    return () => {
      handle.current = null;
    };
  }, [handle]);

  useEffect(() => {
    if (label.current) setLabelWidth(label.current.offsetWidth);
  }, [active]);

  useEffect(() => () => window.clearTimeout(closer.current), []);

  const show = () => {
    window.clearTimeout(closer.current);
    setOpen(true);
  };

  const hide = () => {
    window.clearTimeout(closer.current);
    closer.current = window.setTimeout(() => setOpen(false), 260);
  };

  const menuHeight = chapters.length * ROW + 24;

  return (
    <nav
      className="gnav"
      aria-label="Sections"
      data-open={open}
      style={{ ["--orb" as string]: `${ORB}px`, ["--pill" as string]: `${labelWidth + 34}px`, ["--menu" as string]: `${menuHeight}px` } as CSSProperties}
      onPointerEnter={(event) => {
        if (event.pointerType === "mouse") show();
      }}
      onPointerLeave={(event) => {
        if (event.pointerType === "mouse") hide();
      }}
      onFocus={show}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) hide();
      }}
    >
      <div className="gnav__goo" aria-hidden="true">
        <i className="gnav__blob gnav__blob--menu" />
        <i className="gnav__blob gnav__blob--pill" />
        <i className="gnav__blob gnav__blob--orb" />
      </div>

      <ol className="gnav__menu" aria-hidden={!open}>
        {chapters.map((chapter, i) => (
          <li key={chapter} style={{ height: ROW, ["--i" as string]: chapters.length - 1 - i } as CSSProperties}>
            <button
              type="button"
              tabIndex={open ? 0 : -1}
              ref={(node) => {
                items.current[i] = node;
              }}
              className="gnav__item"
              data-active={i === active}
              aria-current={i === active ? "true" : undefined}
              onClick={() => {
                onGo(i);
                hide();
              }}
            >
              <span className="gnav__idx">{pad(i)}</span>
              <span className="gnav__title">{chapter}</span>
              <i className="gnav__fill" aria-hidden="true" />
            </button>
          </li>
        ))}
      </ol>

      <span className="gnav__label" ref={label} aria-live="polite">
        {chapters[active]}
      </span>

      <button type="button" className="gnav__orb" aria-expanded={open} aria-label={`Section ${active} of ${chapters.length - 1}: ${chapters[active]}. Show all sections`} onClick={() => setOpen((value) => !value)}>
        <svg className="gnav__ring" viewBox="0 0 52 52" aria-hidden="true">
          <circle className="gnav__track" cx="26" cy="26" r="21" pathLength={100} />
          <circle className="gnav__progress" cx="26" cy="26" r="21" pathLength={100} />
        </svg>
        <span className="gnav__num">{pad(active)}</span>
      </button>
    </nav>
  );
}
