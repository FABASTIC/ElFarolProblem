import { useEffect, useRef, useState } from "react";

const VERTEX = `
attribute vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const FRAGMENT = `
precision highp float;
uniform vec2 uResolution;
uniform vec2 uMouse;
uniform float uTime;
uniform float uPresence;
uniform float uDpr;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * noise(p);
    p = p * 2.02 + vec2(11.7, 5.3);
    a *= 0.5;
  }
  return v;
}

void main() {
  vec2 frag = gl_FragCoord.xy;
  vec2 uv = frag / uResolution;
  float aspect = uResolution.x / uResolution.y;
  vec2 p = vec2(uv.x * aspect, uv.y);
  vec2 m = vec2(uMouse.x * aspect, uMouse.y);
  vec2 toward = m - p;
  float d = length(toward);
  float pull = exp(-d * d * 9.0) * uPresence;
  vec2 q = p - toward * pull * 0.42;
  float t = uTime;

  float grain = hash(frag + fract(t * 7.0) * 113.0);
  vec3 color = vec3(0.0275) + (grain - 0.5) * 0.006;

  vec2 space = q * (uResolution.y / uDpr) / 9.0;
  vec2 cell = floor(space);
  vec2 inside = fract(space);
  float seed = hash(cell);
  float star = 0.0;
  if (seed > 0.962) {
    vec2 spot = vec2(hash(cell + 1.7), hash(cell + 9.3)) * 0.7 + 0.15;
    float dist = length(inside - spot) * 9.0;
    float bright = 0.22 + 0.6 * hash(cell + 4.1);
    float twinkle = 0.65 + 0.35 * sin(t * (0.5 + hash(cell + 2.2) * 1.7) + seed * 40.0);
    star = smoothstep(1.15, 0.0, dist) * bright * twinkle;
  }
  star *= 1.0 + pull * 2.4;
  color += vec3(0.93, 0.92, 0.9) * star;

  gl_FragColor = vec4(color, 1.0);
}
`;

function compile(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export default function AuraBackground({ className }: { className?: string }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const root = rootRef.current;
    const host = root?.closest<HTMLElement>(".world") ?? root?.parentElement;
    if (!root || !host) return;
    const canvas = document.createElement("canvas");
    root.appendChild(canvas);
    let rebuild: number | undefined;
    const onLost = (event: Event) => {
      event.preventDefault();
      rebuild = window.setTimeout(() => setGeneration((value) => value + 1), 250);
    };
    canvas.addEventListener("webglcontextlost", onLost);
    const gl = canvas.getContext("webgl", { antialias: false, alpha: false, premultipliedAlpha: false, powerPreference: "low-power" });
    const discard = () => {
      window.clearTimeout(rebuild);
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.remove();
    };
    if (!gl || gl.isContextLost()) return discard;
    const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX);
    const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT);
    if (!vertex || !fragment) return discard;
    const program = gl.createProgram();
    if (!program) return discard;
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return discard;
    gl.useProgram(program);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    const uResolution = gl.getUniformLocation(program, "uResolution");
    const uMouse = gl.getUniformLocation(program, "uMouse");
    const uTime = gl.getUniformLocation(program, "uTime");
    const uPresence = gl.getUniformLocation(program, "uPresence");
    const uDpr = gl.getUniformLocation(program, "uDpr");
    let dpr = 1;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5, presence: 0, target: 0 };
    const start = performance.now();
    let frame = 0;
    let visible = true;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const scale = Math.min(1.5, window.devicePixelRatio || 1);
      dpr = scale;
      canvas.width = Math.max(1, Math.round(rect.width * scale));
      canvas.height = Math.max(1, Math.round(rect.height * scale));
      gl.viewport(0, 0, canvas.width, canvas.height);
    };

    const render = (now: number) => {
      mouse.x += (mouse.tx - mouse.x) * 0.045;
      mouse.y += (mouse.ty - mouse.y) * 0.045;
      mouse.presence += (mouse.target - mouse.presence) * 0.035;
      gl.uniform2f(uResolution, canvas.width, canvas.height);
      gl.uniform2f(uMouse, mouse.x, mouse.y);
      gl.uniform1f(uTime, reduced ? 12 : (now - start) / 1000);
      gl.uniform1f(uPresence, mouse.presence);
      gl.uniform1f(uDpr, dpr);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };

    const loop = (now: number) => {
      render(now);
      frame = visible && !document.hidden ? requestAnimationFrame(loop) : 0;
    };

    const wake = () => {
      if (!frame && !reduced && visible && !document.hidden) frame = requestAnimationFrame(loop);
    };

    const onMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.tx = (event.clientX - rect.left) / rect.width;
      mouse.ty = 1 - (event.clientY - rect.top) / rect.height;
      mouse.target = 1;
      if (reduced) render(performance.now());
    };
    const onLeave = () => {
      mouse.target = 0;
    };

    resize();
    render(performance.now());
    wake();
    const observer = new ResizeObserver(() => {
      resize();
      render(performance.now());
    });
    observer.observe(canvas);
    const sight = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      wake();
    });
    sight.observe(canvas);
    document.addEventListener("visibilitychange", wake);
    host.addEventListener("pointermove", onMove, { passive: true });
    host.addEventListener("pointerleave", onLeave);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      sight.disconnect();
      document.removeEventListener("visibilitychange", wake);
      host.removeEventListener("pointermove", onMove);
      host.removeEventListener("pointerleave", onLeave);
      if (!gl.isContextLost()) {
        gl.deleteBuffer(buffer);
        gl.deleteProgram(program);
        gl.deleteShader(vertex);
        gl.deleteShader(fragment);
      }
      discard();
    };
  }, [generation]);

  return <div ref={rootRef} className={["aura", className].filter(Boolean).join(" ")} aria-hidden="true" />;
}
