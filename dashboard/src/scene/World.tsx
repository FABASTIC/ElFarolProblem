import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type MutableRefObject, type RefObject } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Frame } from "../model";
import type { Lens } from "../people";
import Ground, { BORDER } from "./Ground";
import type { Placements } from "./layout";
import People, { type Motion } from "./People";
import Tavern, { type CrowdState } from "./Tavern";

interface WorldProps {
  frame: Frame | null;
  frameKey: string;
  gridSize: number;
  barMin: number;
  barMax: number;
  placements: Placements;
  crowd: CrowdState;
  fill: number;
  lens: Lens;
  isolate: string | null;
  wealthSpan: number;
  selectedId: number | null;
  hoveredId: MutableRefObject<number | null>;
  motion: MutableRefObject<Motion>;
  overlay: RefObject<HTMLDivElement | null>;
  resetToken: number;
  drift: boolean;
  reduced: boolean;
  onHover: (id: number | null) => void;
  onSelect: (id: number) => void;
}

interface Pose {
  target: THREE.Vector3;
  zoom: number;
  phi: number;
  theta: number;
}

interface Glide {
  start: number;
  duration: number;
  from: Pose;
  to: Pose;
}

const ELEVATION = 0.62;
const AZIMUTH = Math.PI / 4;
const DISTANCE = 140;
const FOCUS = new THREE.Vector3(-3.5, 0, -3.5);

function expoOut(t: number): number {
  return t >= 1 ? 1 : 1 - Math.pow(2, -10 * t);
}

function shortestAngle(from: number, to: number): number {
  let delta = (to - from) % (Math.PI * 2);
  if (delta > Math.PI) delta -= Math.PI * 2;
  if (delta < -Math.PI) delta += Math.PI * 2;
  return from + delta;
}

interface RigProps {
  extent: number;
  resetToken: number;
  drift: boolean;
  reduced: boolean;
  selectedId: number | null;
  motion: MutableRefObject<Motion>;
}

function Rig({ extent, resetToken, drift, reduced, selectedId, motion }: RigProps) {
  const camera = useThree((state) => state.camera) as THREE.OrthographicCamera;
  const gl = useThree((state) => state.gl);
  const width = useThree((state) => state.size.width);
  const height = useThree((state) => state.size.height);
  const controls = useMemo(() => new OrbitControls(camera, gl.domElement), [camera, gl]);
  const shift = useMemo(() => new THREE.Vector3(), []);
  const spherical = useMemo(() => new THREE.Spherical(), []);
  const offset = useMemo(() => new THREE.Vector3(), []);
  const glide = useRef<Glide | null>(null);
  const entered = useRef(false);
  const lastReset = useRef(resetToken);

  const fitZoom = useMemo(() => {
    const diagonal = extent * Math.SQRT2;
    return Math.max(1.2, Math.min(width / (diagonal * 1.02), height / (diagonal * 0.64)));
  }, [extent, width, height]);

  const home = useMemo<Pose>(() => ({ target: FOCUS.clone(), zoom: fitZoom, phi: Math.PI / 2 - ELEVATION, theta: AZIMUTH }), [fitZoom]);

  const apply = (pose: Pose) => {
    spherical.set(DISTANCE, pose.phi, pose.theta);
    offset.setFromSpherical(spherical);
    controls.target.copy(pose.target);
    camera.position.copy(pose.target).add(offset);
    camera.zoom = pose.zoom;
    camera.lookAt(pose.target);
    camera.updateProjectionMatrix();
  };

  const current = (): Pose => {
    offset.copy(camera.position).sub(controls.target);
    spherical.setFromVector3(offset);
    return { target: controls.target.clone(), zoom: camera.zoom, phi: spherical.phi, theta: spherical.theta };
  };

  const glideTo = (to: Pose, duration: number) => {
    if (reduced) {
      apply(to);
      controls.update();
      return;
    }
    glide.current = { start: performance.now(), duration, from: current(), to: { ...to, theta: shortestAngle(current().theta, to.theta) } };
    controls.enabled = false;
  };

  useEffect(() => {
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.panSpeed = 0.9;
    controls.rotateSpeed = 0.5;
    controls.zoomSpeed = 0.9;
    controls.minPolarAngle = 0.3;
    controls.maxPolarAngle = 1.3;
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    camera.near = -1000;
    camera.far = 1000;
    return () => controls.dispose();
  }, [controls, camera]);

  useEffect(() => {
    controls.minZoom = fitZoom * 0.5;
    controls.maxZoom = fitZoom * 6;
    if (!entered.current) {
      entered.current = true;
      apply({ target: FOCUS.clone().add(new THREE.Vector3(6, 0, -10)), zoom: fitZoom * 0.42, phi: 0.2, theta: AZIMUTH + 1.15 });
      glideTo(home, 2800);
      return;
    }
    if (!glide.current) {
      camera.zoom = Math.min(controls.maxZoom, Math.max(controls.minZoom, camera.zoom));
      camera.updateProjectionMatrix();
    }
  }, [fitZoom]);

  useEffect(() => {
    if (lastReset.current === resetToken) return;
    lastReset.current = resetToken;
    glideTo(home, 1500);
  }, [resetToken, home]);

  useEffect(() => {
    if (selectedId == null) return;
    const index = motion.current.indexById.get(selectedId);
    if (index == null) return;
    const p = motion.current.positions;
    const now = current();
    glideTo({ target: new THREE.Vector3(p[index * 3], 0, p[index * 3 + 2]), zoom: Math.max(now.zoom, fitZoom * 1.7), phi: Math.min(now.phi, 0.95), theta: now.theta }, 1300);
  }, [selectedId]);

  useEffect(() => {
    controls.autoRotate = drift && !reduced;
    controls.autoRotateSpeed = 0.35;
  }, [controls, drift, reduced]);

  useFrame(() => {
    const active = glide.current;
    if (active) {
      const t = Math.min(1, (performance.now() - active.start) / active.duration);
      const e = expoOut(t);
      apply({
        target: active.from.target.clone().lerp(active.to.target, e),
        zoom: active.from.zoom + (active.to.zoom - active.from.zoom) * e,
        phi: active.from.phi + (active.to.phi - active.from.phi) * e,
        theta: active.from.theta + (active.to.theta - active.from.theta) * e,
      });
      if (t >= 1) {
        glide.current = null;
        controls.enabled = true;
        controls.update();
      }
      return;
    }
    controls.update();
    const limit = extent / 2;
    const target = controls.target;
    shift.set(THREE.MathUtils.clamp(target.x, -limit, limit) - target.x, -target.y, THREE.MathUtils.clamp(target.z, -limit, limit) - target.z);
    if (shift.lengthSq() > 0) {
      target.add(shift);
      camera.position.add(shift);
    }
  });

  return null;
}

interface Callout {
  x: number;
  y: number;
  side: -1 | 1;
  frame: number;
}

interface Bounds {
  at: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

const GAP = 10;
const EDGE = 22;

function hudEdge(selector: string, base: DOMRect): DOMRect | null {
  const element = document.querySelector<HTMLElement>(selector);
  if (!element || getComputedStyle(element).position !== "absolute") return null;
  const rect = element.getBoundingClientRect();
  if (!rect.width || !rect.height || rect.bottom < base.top || rect.top > base.bottom) return null;
  return rect;
}

function Projector({ overlay, motion }: { overlay: RefObject<HTMLDivElement | null>; motion: MutableRefObject<Motion> }) {
  const camera = useThree((state) => state.camera);
  const width = useThree((state) => state.size.width);
  const height = useThree((state) => state.size.height);
  const point = useMemo(() => new THREE.Vector3(), []);
  const callouts = useRef(new Map<number, Callout>());
  const bounds = useRef<Bounds>({ at: -1, left: EDGE, right: 0, top: EDGE, bottom: 0 });
  const tick = useRef(0);

  useFrame((_, delta) => {
    const root = overlay.current;
    if (!root) return;
    const m = motion.current;
    const now = performance.now();
    tick.current += 1;
    if (now - bounds.current.at > 400 || bounds.current.right === 0) {
      const base = root.getBoundingClientRect();
      const left = hudEdge(".hud--left", base);
      const right = hudEdge(".hud--right", base);
      const bottom = hudEdge(".hud--bottom", base);
      const tools = hudEdge(".world__tools", base);
      const floor = Math.min(bottom ? bottom.top - base.top : height, tools ? tools.top - base.top : height);
      bounds.current = {
        at: now,
        left: left ? left.right - base.left + EDGE : EDGE,
        right: right ? right.left - base.left - EDGE : width - EDGE,
        top: EDGE + 8,
        bottom: floor - EDGE,
      };
    }
    camera.updateMatrixWorld();
    const area = bounds.current;
    const ease = 1 - Math.exp(-Math.min(delta, 0.1) * 9);
    const project = (index: number, lift: number) => {
      point.set(m.positions[index * 3], m.positions[index * 3 + 1] + lift, m.positions[index * 3 + 2]).project(camera);
      return { x: ((point.x + 1) / 2) * width, y: ((1 - point.y) / 2) * height };
    };

    const columns: Record<string, { element: HTMLElement; id: number; slot: number; ax: number; ay: number; w: number; h: number; state: Callout; target: number }[]> = { "-1": [], "1": [] };
    const single = area.right - area.left < 224 * 2 + 160;
    const middle = single ? Infinity : (area.left + area.right) / 2;

    root.querySelectorAll<HTMLElement>("[data-agent]").forEach((element) => {
      const id = Number(element.dataset.agent);
      const index = m.indexById.get(id);
      if (index == null) {
        element.style.visibility = "hidden";
        return;
      }
      const anchor = project(index, Number(element.dataset.lift ?? "1.25"));
      if (!element.dataset.callout) {
        element.style.transform = `translate3d(${anchor.x.toFixed(1)}px, ${anchor.y.toFixed(1)}px, 0)`;
        element.style.visibility = "visible";
        return;
      }
      const body = element.firstElementChild as HTMLElement | null;
      const w = body?.offsetWidth ?? 220;
      const h = body?.offsetHeight ?? 56;
      const offscreen = anchor.x < area.left - 60 || anchor.x > area.right + 60 || anchor.y < -40 || anchor.y > height + 40;
      if (offscreen) {
        element.style.visibility = "hidden";
        element.dataset.off = "true";
        return;
      }
      element.dataset.off = "false";
      let state = callouts.current.get(id);
      if (!state) {
        const side: -1 | 1 = anchor.x < middle ? -1 : 1;
        state = { x: side < 0 ? area.left : area.right - w, y: anchor.y - h / 2, side, frame: tick.current };
        callouts.current.set(id, state);
      } else if (single) state.side = -1;
      else if (state.side === -1 && anchor.x > middle + 80) state.side = 1;
      else if (state.side === 1 && anchor.x < middle - 80) state.side = -1;
      state.frame = tick.current;
      columns[String(state.side)].push({ element, id, slot: Number(element.dataset.slot ?? id), ax: anchor.x, ay: anchor.y, w, h, state, target: anchor.y - h / 2 });
    });

    for (const key of ["-1", "1"]) {
      const limit = 3;
      const keep = new Set(columns[key].slice().sort((a, b) => a.slot - b.slot).slice(0, limit).map((item) => item.id));
      for (const item of columns[key]) {
        if (keep.has(item.id)) continue;
        item.element.style.visibility = "hidden";
        item.state.frame = -1;
      }
      const list = columns[key].filter((item) => keep.has(item.id)).sort((a, b) => a.ay - b.ay || a.id - b.id);
      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        const floor = i ? list[i - 1].target + list[i - 1].h + GAP : area.top;
        item.target = Math.max(floor, Math.min(area.bottom - item.h, item.target));
      }
      for (let i = list.length - 1; i >= 0; i -= 1) {
        const item = list[i];
        const ceiling = i < list.length - 1 ? list[i + 1].target - GAP - item.h : area.bottom - item.h;
        item.target = Math.max(area.top, Math.min(item.target, ceiling));
      }
      for (const item of list) {
        const goalX = item.state.side < 0 ? area.left : area.right - item.w;
        item.state.x += (goalX - item.state.x) * ease;
        item.state.y += (item.target - item.state.y) * ease;
        item.element.style.transform = `translate3d(${item.state.x.toFixed(1)}px, ${item.state.y.toFixed(1)}px, 0)`;
        item.element.style.visibility = "visible";
        const leader = root.querySelector<SVGGElement>(`[data-leader="${item.id}"]`);
        if (leader) {
          const ex = item.state.side < 0 ? item.state.x + item.w : item.state.x;
          const ey = item.state.y + item.h / 2;
          const out = item.state.side < 0 ? 1 : -1;
          const reach = Math.max(40, Math.abs(item.ax - ex) * 0.45);
          const d = `M${ex.toFixed(1)} ${ey.toFixed(1)} C${(ex + out * reach).toFixed(1)} ${ey.toFixed(1)} ${(item.ax - out * reach * 0.5).toFixed(1)} ${(item.ay - 20).toFixed(1)} ${item.ax.toFixed(1)} ${(item.ay - 7).toFixed(1)}`;
          leader.querySelectorAll("path").forEach((path) => path.setAttribute("d", d));
          const label = leader.querySelector("text");
          label?.setAttribute("x", (item.ax + 10).toFixed(1));
          label?.setAttribute("y", (item.ay - 10).toFixed(1));
          leader.querySelectorAll("circle").forEach((dot) => {
            dot.setAttribute("cx", item.ax.toFixed(1));
            dot.setAttribute("cy", item.ay.toFixed(1));
          });
          leader.style.visibility = "visible";
        }
      }
    }

    root.querySelectorAll<SVGGElement>("[data-leader]").forEach((leader) => {
      const state = callouts.current.get(Number(leader.dataset.leader));
      if (!state || state.frame !== tick.current) leader.style.visibility = "hidden";
    });
    for (const [id, state] of callouts.current) {
      if (tick.current - state.frame > 120) callouts.current.delete(id);
    }
  });

  return null;
}

export default function World({
  frame,
  frameKey,
  gridSize,
  barMin,
  barMax,
  placements,
  crowd,
  fill,
  lens,
  isolate,
  wealthSpan,
  selectedId,
  hoveredId,
  motion,
  overlay,
  resetToken,
  drift,
  reduced,
  onHover,
  onSelect,
}: WorldProps) {
  return (
    <Canvas
      orthographic
      flat
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      camera={{ position: [70, 57, 70], zoom: 8, near: -1000, far: 1000 }}
      aria-label="Town of El Farol: faceted figures walk between their homes and the bar each night"
    >
      <Rig extent={gridSize + BORDER} resetToken={resetToken} drift={drift} reduced={reduced} selectedId={selectedId} motion={motion} />
      <Ground gridSize={gridSize} barMin={barMin} barMax={barMax} />
      <Tavern gridSize={gridSize} barMin={barMin} barMax={barMax} crowd={crowd} fill={fill} reduced={reduced} />
      <People
        frame={frame}
        frameKey={frameKey}
        gridSize={gridSize}
        placements={placements}
        lens={lens}
        isolate={isolate}
        wealthSpan={wealthSpan}
        selectedId={selectedId}
        hoveredId={hoveredId}
        motion={motion}
        reduced={reduced}
        onHover={onHover}
        onSelect={onSelect}
      />
      <Projector overlay={overlay} motion={motion} />
    </Canvas>
  );
}
