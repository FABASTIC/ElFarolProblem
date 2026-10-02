import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Frame } from "../model";
import Agents, { type AgentHover } from "./Agents";
import Floor from "./Floor";

interface ArenaProps {
  frame: Frame | null;
  frameKey: string;
  gridSize: number;
  barMin: number;
  barMax: number;
  heightRange: [number, number];
  resetToken: number;
  onHover: (hover: AgentHover | null) => void;
}

const ELEVATION = Math.atan(1 / Math.SQRT2);
const AZIMUTH = Math.PI / 4;
const DISTANCE = 120;

function Rig({ gridSize, resetToken }: { gridSize: number; resetToken: number }) {
  const camera = useThree((state) => state.camera) as THREE.OrthographicCamera;
  const gl = useThree((state) => state.gl);
  const width = useThree((state) => state.size.width);
  const height = useThree((state) => state.size.height);
  const controls = useMemo(() => new OrbitControls(camera, gl.domElement), [camera, gl]);

  const fitZoom = useMemo(() => {
    const diagonal = gridSize * Math.SQRT2;
    return Math.max(1.5, Math.min(width / (diagonal * 1.06), height / (diagonal * 0.74)));
  }, [gridSize, width, height]);

  useEffect(() => {
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = false;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.9;
    controls.minPolarAngle = 0.32;
    controls.maxPolarAngle = 1.32;
    controls.target.set(0, 0, 0);
    return () => controls.dispose();
  }, [controls]);

  useEffect(() => {
    camera.position.set(
      DISTANCE * Math.cos(ELEVATION) * Math.sin(AZIMUTH),
      DISTANCE * Math.sin(ELEVATION),
      DISTANCE * Math.cos(ELEVATION) * Math.cos(AZIMUTH),
    );
    camera.zoom = fitZoom;
    camera.near = -1000;
    camera.far = 1000;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    controls.minZoom = fitZoom * 0.55;
    controls.maxZoom = fitZoom * 4.5;
    controls.target.set(0, 0, 0);
    controls.update();
  }, [camera, controls, fitZoom, resetToken]);

  useFrame(() => {
    controls.update();
  });

  return null;
}

export default function Arena({ frame, frameKey, gridSize, barMin, barMax, heightRange, resetToken, onHover }: ArenaProps) {
  return (
    <Canvas
      orthographic
      flat
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      camera={{ position: [70, 57, 70], zoom: 8, near: -1000, far: 1000 }}
      onPointerMissed={() => onHover(null)}
      aria-label="Isometric arena: agents as pillars on the grid, bar zone outlined in amber"
    >
      <ambientLight intensity={0.62} />
      <directionalLight position={[40, 90, 25]} intensity={1.35} />
      <directionalLight position={[-50, 25, -35]} intensity={0.3} />
      <Rig gridSize={gridSize} resetToken={resetToken} />
      <Floor gridSize={gridSize} barMin={barMin} barMax={barMax} />
      <Agents frame={frame} frameKey={frameKey} gridSize={gridSize} heightRange={heightRange} onHover={onHover} />
    </Canvas>
  );
}
