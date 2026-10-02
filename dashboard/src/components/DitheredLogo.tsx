import { useEffect, useRef } from "react";

interface DitheredLogoProps {
  svg: string;
  size?: number;
  grid?: number;
  threshold?: number;
  diffusionStrength?: number;
  serpentine?: boolean;
  dotScale?: number;
  particleColor?: string;
  accentColor?: string;
  className?: string;
  label?: string;
}

interface Particle {
  hx: number;
  hy: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  accent: boolean;
}

function dither(data: Uint8ClampedArray, n: number, threshold: number, strength: number, serpentine: boolean) {
  const lum = new Float32Array(n * n);
  const hue = new Uint8Array(n * n);
  for (let i = 0; i < n * n; i += 1) {
    const a = data[i * 4 + 3] / 255;
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    lum[i] = (0.299 * r + 0.587 * g + 0.114 * b) * a;
    hue[i] = r > 200 && b < 160 && g < 230 ? 1 : 0;
  }
  const out: { x: number; y: number; accent: boolean }[] = [];
  for (let y = 0; y < n; y += 1) {
    const reverse = serpentine && y % 2 === 1;
    for (let k = 0; k < n; k += 1) {
      const x = reverse ? n - 1 - k : k;
      const i = y * n + x;
      const old = lum[i];
      const on = old >= threshold;
      const error = (old - (on ? 255 : 0)) * strength;
      if (on) out.push({ x, y, accent: hue[i] === 1 });
      const dir = reverse ? -1 : 1;
      const spread = (dx: number, dy: number, w: number) => {
        const nx = x + dx * dir;
        const ny = y + dy;
        if (nx < 0 || nx >= n || ny >= n) return;
        lum[ny * n + nx] += (error * w) / 16;
      };
      spread(1, 0, 7);
      spread(-1, 1, 3);
      spread(0, 1, 5);
      spread(1, 1, 1);
    }
  }
  return out;
}

export default function DitheredLogo({
  svg,
  size = 40,
  grid = 40,
  threshold = 110,
  diffusionStrength = 1,
  serpentine = true,
  dotScale = 1,
  particleColor = "#ece9e2",
  accentColor = "#ffcf7a",
  className,
  label = "El Farol",
}: DitheredLogoProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const pad = size * 0.5;
    const span = size + pad * 2;
    canvas.width = Math.round(span * dpr);
    canvas.height = Math.round(span * dpr);
    canvas.style.width = `${span}px`;
    canvas.style.height = `${span}px`;
    canvas.style.margin = `${-pad}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    let particles: Particle[] = [];
    let frame = 0;
    let running = false;
    let cancelled = false;
    const pointer = { x: -999, y: -999, inside: false };
    const ripples: { x: number; y: number; t: number }[] = [];
    const cell = size / grid;
    const dot = Math.max(0.8, cell * 0.82 * dotScale);

    const draw = () => {
      ctx.clearRect(0, 0, span, span);
      for (const p of particles) {
        ctx.fillStyle = p.accent ? accentColor : particleColor;
        ctx.fillRect(p.x - dot / 2, p.y - dot / 2, dot, dot);
      }
    };

    const tick = (now: number) => {
      let moving = false;
      for (const p of particles) {
        let fx = (p.hx - p.x) * 0.075;
        let fy = (p.hy - p.y) * 0.075;
        if (pointer.inside) {
          const dx = p.x - pointer.x;
          const dy = p.y - pointer.y;
          const d2 = dx * dx + dy * dy;
          const radius = size * 0.32;
          if (d2 < radius * radius) {
            const d = Math.sqrt(d2) || 1;
            const push = (1 - d / radius) * 2.4;
            fx += (dx / d) * push;
            fy += (dy / d) * push;
          }
        }
        for (const ripple of ripples) {
          const age = (now - ripple.t) / 1000;
          const front = age * size * 1.6;
          const dx = p.hx - ripple.x;
          const dy = p.hy - ripple.y;
          const d = Math.sqrt(dx * dx + dy * dy) || 1;
          const band = Math.abs(d - front);
          if (band < cell * 3) {
            const kick = (1 - band / (cell * 3)) * (1 - age) * 1.8;
            fx += (dx / d) * kick;
            fy += (dy / d) * kick;
          }
        }
        p.vx = (p.vx + fx) * 0.78;
        p.vy = (p.vy + fy) * 0.78;
        p.x += p.vx;
        p.y += p.vy;
        if (Math.abs(p.vx) + Math.abs(p.vy) > 0.01 || Math.abs(p.hx - p.x) + Math.abs(p.hy - p.y) > 0.05) moving = true;
      }
      for (let i = ripples.length - 1; i >= 0; i -= 1) if (now - ripples[i].t > 1000) ripples.splice(i, 1);
      draw();
      if (moving || pointer.inside || ripples.length) frame = requestAnimationFrame(tick);
      else running = false;
    };

    const wake = () => {
      if (running || reduced) return;
      running = true;
      frame = requestAnimationFrame(tick);
    };

    const image = new Image();
    image.onload = () => {
      if (cancelled) return;
      const sample = document.createElement("canvas");
      sample.width = grid;
      sample.height = grid;
      const sctx = sample.getContext("2d", { willReadFrequently: true });
      if (!sctx) return;
      sctx.drawImage(image, 0, 0, grid, grid);
      const points = dither(sctx.getImageData(0, 0, grid, grid).data, grid, threshold, diffusionStrength, serpentine);
      particles = points.map((point) => {
        const hx = pad + (point.x + 0.5) * cell;
        const hy = pad + (point.y + 0.5) * cell;
        const angle = Math.random() * Math.PI * 2;
        const far = reduced ? 0 : size * (0.4 + Math.random() * 0.5);
        return { hx, hy, x: hx + Math.cos(angle) * far, y: hy + Math.sin(angle) * far, vx: 0, vy: 0, accent: point.accent };
      });
      draw();
      wake();
    };
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;

    const locate = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * span;
      pointer.y = ((event.clientY - rect.top) / rect.height) * span;
    };
    const host = canvas.parentElement ?? canvas;
    const onMove = (event: PointerEvent) => {
      locate(event);
      pointer.inside = true;
      wake();
    };
    const onLeave = () => {
      pointer.inside = false;
      wake();
    };
    const onDown = (event: PointerEvent) => {
      locate(event);
      ripples.push({ x: pointer.x, y: pointer.y, t: performance.now() });
      wake();
    };
    host.addEventListener("pointermove", onMove);
    host.addEventListener("pointerleave", onLeave);
    host.addEventListener("pointerdown", onDown);
    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
      host.removeEventListener("pointermove", onMove);
      host.removeEventListener("pointerleave", onLeave);
      host.removeEventListener("pointerdown", onDown);
    };
  }, [svg, size, grid, threshold, diffusionStrength, serpentine, dotScale, particleColor, accentColor]);

  return (
    <span className={["dither", className].filter(Boolean).join(" ")} role="img" aria-label={label} style={{ width: size, height: size }}>
      <canvas ref={canvasRef} />
    </span>
  );
}
