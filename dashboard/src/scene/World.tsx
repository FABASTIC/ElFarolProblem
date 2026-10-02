import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, type MutableRefObject, type RefObject } from "react";
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

const ELEVATION = 0.62;
const AZIMUTH = Math.PI / 4;
const DISTANCE = 140;
const FOCUS = new THREE.Vector3(-3.5, 0, -3.5);

function Rig({ extent, resetToken, drift, reduced }: { extent: number; resetToken: number; drift: boolean; reduced: boolean }) {
  const camera = useThree((state) => state.camera) as THREE.OrthographicCamera;
  const gl = useThree((state) => state.gl);
  const width = useThree((state) => state.size.width);
  const height = useThree((state) => state.size.height);
  const controls = useMemo(() => new OrbitControls(camera, gl.domElement), [camera, gl]);
  const shift = useMemo(() => new THREE.Vector3(), []);

  const fitZoom = useMemo(() => {
    const diagonal = extent * Math.SQRT2;
    return Math.max(1.2, Math.min(width / (diagonal * 1.02), height / (diagonal * 0.64)));
  }, [extent, width, height]);

  useEffect(() => {
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.panSpeed = 0.9;
    controls.rotateSpeed = 0.5;
    controls.zoomSpeed = 0.9;
    controls.minPolarAngle = 0.3;
    controls.maxPolarAngle = 1.3;
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    return () => controls.dispose();
  }, [controls]);

  useEffect(() => {
    camera.position.set(
      FOCUS.x + DISTANCE * Math.cos(ELEVATION) * Math.sin(AZIMUTH),
      DISTANCE * Math.sin(ELEVATION),
      FOCUS.z + DISTANCE * Math.cos(ELEVATION) * Math.cos(AZIMUTH),
    );
    camera.zoom = fitZoom;
    camera.near = -1000;
    camera.far = 1000;
    camera.lookAt(FOCUS);
    camera.updateProjectionMatrix();
    controls.minZoom = fitZoom * 0.6;
    controls.maxZoom = fitZoom * 6;
    controls.target.copy(FOCUS);
    controls.update();
  }, [camera, controls, fitZoom, resetToken]);

  useEffect(() => {
    controls.autoRotate = drift && !reduced;
    controls.autoRotateSpeed = 0.35;
  }, [controls, drift, reduced]);

  useFrame(() => {
    controls.update();
    const limit = extent / 2;
    const t = controls.target;
    shift.set(THREE.MathUtils.clamp(t.x, -limit, limit) - t.x, -t.y, THREE.MathUtils.clamp(t.z, -limit, limit) - t.z);
    if (shift.lengthSq() > 0) {
      t.add(shift);
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
      aria-label="Town of El Farol: people walk between their homes and the bar each night"
    >
      <hemisphereLight args={["#8597cc", "#0b0e14", 2.6]} />
      <directionalLight position={[-36, 60, 24]} intensity={1.7} color="#b7c4ff" />
      <Rig extent={gridSize + BORDER} resetToken={resetToken} drift={drift} reduced={reduced} />
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
