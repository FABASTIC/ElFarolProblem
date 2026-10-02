import { useId, useState, type ReactNode } from "react";

const STORE = "elfarol.folds";

function readStore(): Record<string, boolean> {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORE) ?? "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writeStore(id: string, open: boolean) {
  try {
    const store = readStore();
    store[id] = open;
    window.localStorage.setItem(STORE, JSON.stringify(store));
  } catch {
    return;
  }
}

function useFold(id: string, defaultOpen: boolean): [boolean, () => void] {
  const [open, setOpen] = useState<boolean>(() => readStore()[id] ?? defaultOpen);
  const toggle = () => {
    const next = !open;
    setOpen(next);
    writeStore(id, next);
  };
  return [open, toggle];
}

export function PlusMinus({ open }: { open: boolean }) {
  return (
    <span className="plusminus" data-open={open} aria-hidden="true">
      <i />
      <i />
    </span>
  );
}

interface FoldProps {
  id: string;
  index: string;
  title: string;
  meta?: ReactNode;
  defaultOpen?: boolean;
  actions?: ReactNode;
  className?: string;
  label?: string;
  children: ReactNode;
}

export default function Fold({ id, index, title, meta, defaultOpen = true, actions, className, label, children }: FoldProps) {
  const [open, toggle] = useFold(id, defaultOpen);
  const bodyId = useId();

  return (
    <section className={`hud-card fold ${className ?? ""}`} data-open={open} aria-label={label ?? title}>
      <header className="fold__head">
        <button type="button" className="fold__toggle" aria-expanded={open} aria-controls={bodyId} onClick={toggle}>
          <span className="fold__index">{index}</span>
          <span className="fold__title">{title}</span>
          {meta != null && <span className="fold__meta">{meta}</span>}
          <span className="fold__tip" aria-hidden="true">
            {open ? "COLLAPSE" : "EXPAND"}
          </span>
          <PlusMinus open={open} />
        </button>
        {actions && <div className="fold__actions">{actions}</div>}
      </header>
      <div className="fold__body" id={bodyId} inert={!open}>
        <div className="fold__inner">
          <div className="fold__content">{children}</div>
        </div>
      </div>
    </section>
  );
}

interface SubfoldProps {
  id: string;
  title: ReactNode;
  meta?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function Subfold({ id, title, meta, defaultOpen = false, children }: SubfoldProps) {
  const [open, toggle] = useFold(id, defaultOpen);
  const bodyId = useId();

  return (
    <div className="subfold" data-open={open}>
      <button type="button" className="subfold__toggle" aria-expanded={open} aria-controls={bodyId} onClick={toggle}>
        <span className="subfold__title">{title}</span>
        {meta != null && <span className="subfold__meta">{meta}</span>}
        <PlusMinus open={open} />
      </button>
      <div className="fold__body" id={bodyId} inert={!open}>
        <div className="fold__inner">
          <div className="fold__content">{children}</div>
        </div>
      </div>
    </div>
  );
}
