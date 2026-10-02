import { useEffect, useRef, type ReactNode } from "react";

export default function Stage({ children }: { children: ReactNode }) {
  const rig = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = rig.current;
    if (!element) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const narrow = window.matchMedia("(max-width: 1180px)");
    let frame = 0;
    const settle = () => {
      element.style.setProperty("--tilt-x", "0deg");
      element.style.setProperty("--tilt-y", "0deg");
    };
    const onMove = (event: PointerEvent) => {
      if (reduced.matches || narrow.matches || event.buttons !== 0 || event.pointerType === "touch") return;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const nx = event.clientX / window.innerWidth - 0.5;
        const ny = event.clientY / window.innerHeight - 0.5;
        element.style.setProperty("--tilt-x", `${(-ny * 1.4).toFixed(3)}deg`);
        element.style.setProperty("--tilt-y", `${(nx * 2.0).toFixed(3)}deg`);
      });
    };
    window.addEventListener("pointermove", onMove);
    document.documentElement.addEventListener("pointerleave", settle);
    window.addEventListener("blur", settle);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("pointermove", onMove);
      document.documentElement.removeEventListener("pointerleave", settle);
      window.removeEventListener("blur", settle);
    };
  }, []);

  useEffect(() => {
    const element = rig.current;
    if (!element) return;
    const anchor = () => {
      const perspective = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--persp")) || 2400;
      const originX = element.clientWidth / 2;
      const originY = element.clientHeight / 2;
      element.querySelectorAll<HTMLElement>(".panel").forEach((panel) => {
        const depth = parseFloat(getComputedStyle(panel).getPropertyValue("--z")) || 0;
        const centerX = panel.offsetLeft + panel.offsetWidth / 2;
        const centerY = panel.offsetTop + panel.offsetHeight / 2;
        panel.style.setProperty("--shift-x", `${(((originX - centerX) * depth) / perspective).toFixed(2)}px`);
        panel.style.setProperty("--shift-y", `${(((originY - centerY) * depth) / perspective).toFixed(2)}px`);
      });
    };
    anchor();
    const observer = new ResizeObserver(anchor);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="stage">
      <div className="stage__rig" ref={rig}>
        <div className="stage__deep" aria-hidden="true">
          <div className="stage__lattice" />
          <div className="stage__scan" />
        </div>
        {children}
      </div>
    </div>
  );
}
