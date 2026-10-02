import { useEffect, useRef } from "react";

type Mode = "ring" | "lens" | "dot" | "native" | "hidden";

const NATIVE = 'input:not([type="range"]):not([type="checkbox"]):not([type="radio"]), textarea, select, [contenteditable="true"]';
const INTERACTIVE = 'a, button, [role="button"], [role="tab"], [role="radio"], [role="option"], [role="slider"], summary, label, input[type="range"], .bubble, .legend__item';
const QUIET = ".hud, .topbar, .lab, .boot, .world__tools, .tooltip, .nametag";

function modeFor(target: Element | null): Mode {
  if (!target) return "ring";
  if (target.closest(NATIVE)) return "native";
  const interactive = target.closest(INTERACTIVE);
  if (interactive && !(interactive as HTMLButtonElement).disabled && interactive.getAttribute("aria-disabled") !== "true") return "lens";
  if (target.closest(QUIET)) return "dot";
  return "ring";
}

export default function Cursor() {
  const root = useRef<HTMLDivElement>(null);
  const ringPos = useRef<HTMLDivElement>(null);
  const dotPos = useRef<HTMLDivElement>(null);
  const blobPos = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = root.current;
    if (!element || !ringPos.current || !dotPos.current || !blobPos.current) return;
    const fine = window.matchMedia("(hover: hover) and (pointer: fine)");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (!fine.matches || reduced.matches) return;
    const ring = ringPos.current;
    const dot = dotPos.current;
    const blob = blobPos.current;
    const html = document.documentElement;
    html.classList.add("has-cursor");

    const target = { x: -100, y: -100 };
    const slow = { x: -100, y: -100 };
    const mid = { x: -100, y: -100 };
    const fast = { x: -100, y: -100 };
    let frame = 0;
    let mode: Mode = "hidden";
    let seen = false;

    const setMode = (next: Mode) => {
      if (next === mode) return;
      mode = next;
      element.dataset.mode = next;
    };

    const tick = () => {
      slow.x += (target.x - slow.x) * 0.16;
      slow.y += (target.y - slow.y) * 0.16;
      mid.x += (target.x - mid.x) * 0.32;
      mid.y += (target.y - mid.y) * 0.32;
      fast.x += (target.x - fast.x) * 0.55;
      fast.y += (target.y - fast.y) * 0.55;
      ring.style.transform = `translate3d(${slow.x}px, ${slow.y}px, 0)`;
      blob.style.transform = `translate3d(${mid.x}px, ${mid.y}px, 0)`;
      dot.style.transform = `translate3d(${fast.x}px, ${fast.y}px, 0)`;
      const settled = Math.abs(target.x - slow.x) + Math.abs(target.y - slow.y) < 0.15;
      frame = settled ? 0 : requestAnimationFrame(tick);
    };

    const kick = () => {
      if (!frame) frame = requestAnimationFrame(tick);
    };

    const onMove = (event: PointerEvent) => {
      if (event.pointerType !== "mouse") return;
      target.x = event.clientX;
      target.y = event.clientY;
      if (!seen) {
        seen = true;
        slow.x = mid.x = fast.x = target.x;
        slow.y = mid.y = fast.y = target.y;
      }
      setMode(modeFor(event.target as Element | null));
      kick();
    };

    const onDown = () => element.setAttribute("data-pressed", "true");
    const onUp = () => element.removeAttribute("data-pressed");
    const onLeave = (event: MouseEvent) => {
      if (!event.relatedTarget) setMode("hidden");
    };
    const onBlur = () => setMode("hidden");

    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });
    window.addEventListener("pointerup", onUp, { passive: true });
    document.addEventListener("mouseout", onLeave);
    window.addEventListener("blur", onBlur);
    return () => {
      cancelAnimationFrame(frame);
      html.classList.remove("has-cursor");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      document.removeEventListener("mouseout", onLeave);
      window.removeEventListener("blur", onBlur);
    };
  }, []);

  return (
    <div ref={root} className="cursor" data-mode="hidden" aria-hidden="true">
      <div ref={ringPos} className="cursor__pos">
        <span className="cursor__ring" />
      </div>
      <div ref={blobPos} className="cursor__pos">
        <span className="cursor__blob" />
      </div>
      <div ref={dotPos} className="cursor__pos">
        <span className="cursor__dot" />
      </div>
    </div>
  );
}
