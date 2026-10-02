import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type MutableRefObject, type RefObject } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Frame } from "../model";
import type { Lens } from "../people";
import Ground, { BORDER } from "./Ground";
import People, { type Motion } from "./People";
import Tavern, { type CrowdState } from "./Tavern";

interface WorldProps {
  frame: Frame | null;
  frameKey: string;
  gridSize: number;
  barMin: number;
  barMax: number;
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

function Projector({ overlay, motion }: { overlay: RefObject<HTMLDivElement | null>; motion: MutableRefObject<Motion> }) {
  const camera = useThree((state) => state.camera);
  const width = useThree((state) => state.size.width);
  const height = useThree((state) => state.size.height);
  const point = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const root = overlay.current;
    if (!root) return;
    const m = motion.current;
    root.querySelectorAll<HTMLElement>("[data-agent]").forEach((element) => {
      const index = m.indexById.get(Number(element.dataset.agent));
      if (index == null) {
        element.style.visibility = "hidden";
        return;
      }
      const lift = Number(element.dataset.lift ?? "1.25");
      point.set(m.positions[index * 3], m.positions[index * 3 + 1] + lift, m.positions[index * 3 + 2]).project(camera);
      const x = ((point.x + 1) / 2) * width;
      const y = ((1 - point.y) / 2) * height;
      element.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
      element.style.visibility = "visible";
    });
  });

  return null;
}

export default function World({
  frame,
  frameKey,
  gridSize,
  barMin,
  barMax,
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
