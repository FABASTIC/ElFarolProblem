import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { beamGeometry, createBeamMaterial } from "./beam";
import { createEdgeMaterial } from "./edgeMaterial";
import { cuboid, facetedGeometry, lighthouseGeometry } from "./facets";
import { LIGHTHOUSE_OFFSET } from "./Ground";

export type CrowdState = "idle" | "comfortable" | "crowded";

interface TavernProps {
  gridSize: number;
  barMin: number;
  barMax: number;
  crowd: CrowdState;
  fill: number;
  reduced: boolean;
}

const PALETTE: Record<CrowdState, { light: string; floor: string; beam: number; glow: number; spin: number }> = {
  idle: { light: "#8a8780", floor: "#3a3835", beam: 0.03, glow: 0.12, spin: 0.16 },
  comfortable: { light: "#ffcf7a", floor: "#ffb347", beam: 0.08, glow: 0.2, spin: 0.3 },
  crowded: { light: "#ff5a4f", floor: "#d03b3b", beam: 0.11, glow: 0.26, spin: 0.6 },
};

const FLOOR_BASE = new THREE.Color("#0b0a09");
const BULB = new THREE.OctahedronGeometry(0.07);
const BEAM = beamGeometry(4.6, 36);
const BEAM_GAIN = 2.4;
const LIGHTHOUSE = lighthouseGeometry();
const LANTERN = new THREE.CylinderGeometry(0.5, 0.5, 0.9, 8, 1, true);
const LANTERN_HALO = new THREE.IcosahedronGeometry(1.5, 0);

function paintSign(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "#070707";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(236, 233, 226, 0.55)";
  ctx.lineWidth = 2;
  ctx.strokeRect(6, 6, canvas.width - 12, canvas.height - 12);
  ctx.fillStyle = "#ffcf7a";
  ctx.beginPath();
  ctx.arc(34, canvas.height / 2, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#ece9e2";
  ctx.font = "800 62px 'Inter Tight', 'Helvetica Neue', Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const letters = "EL FAROL".split("");
  const step = 50;
  const startX = canvas.width / 2 - ((letters.length - 1) * step) / 2 + 12;
  letters.forEach((ch, i) => ctx.fillText(ch, startX + i * step, canvas.height / 2 + 4));
}

function signTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 112;
  paintSign(canvas);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts
      .load("800 62px 'Inter Tight'")
      .then(() => {
        paintSign(canvas);
        texture.needsUpdate = true;
      })
      .catch(() => undefined);
  }
  return texture;
}

function wallSegments(x0: number, x1: number, door: number): [number, number, number, number][] {
  const mid = (x0 + x1) / 2;
  const out: [number, number, number, number][] = [];
  const sides: [number, number, number, number, boolean][] = [
    [x0, x0, x1, x0, true],
    [x1, x0, x1, x1, false],
    [x1, x1, x0, x1, true],
    [x0, x1, x0, x0, false],
  ];
  for (const [ax, az, bx, bz, horizontal] of sides) {
    if (horizontal) {
      const lo = Math.min(ax, bx);
      const hi = Math.max(ax, bx);
      out.push([lo, az, mid - door / 2, az], [mid + door / 2, az, hi, az]);
    } else {
      const lo = Math.min(az, bz);
      const hi = Math.max(az, bz);
      out.push([ax, lo, ax, mid - door / 2], [ax, mid + door / 2, ax, hi]);
    }
  }
  return out;
}

function Lighthouse({ position, crowd, reduced }: { position: [number, number, number]; crowd: CrowdState; reduced: boolean }) {
  const beams = useRef<THREE.Group>(null);
  const lantern = useRef<THREE.MeshBasicMaterial>(null);
  const halo = useRef<THREE.MeshBasicMaterial>(null);
  const beamA = useMemo(() => createBeamMaterial({ color: PALETTE.idle.light, intensity: PALETTE.idle.beam * BEAM_GAIN }), []);
  const beamB = useMemo(() => createBeamMaterial({ color: PALETTE.idle.light, intensity: PALETTE.idle.beam * BEAM_GAIN }), []);
  const color = useMemo(() => new THREE.Color(PALETTE.idle.light), []);
  const target = useMemo(() => new THREE.Color(), []);
  const level = useRef(PALETTE.idle.beam);
  const tower = useMemo(() => createEdgeMaterial({ face: "#0c0c0c", edge: "#8f8c86", ink: "#ece9e2", headInk: 0.8, width: 1.05, rim: 0.06 }), []);

  useEffect(
    () => () => {
      tower.dispose();
      beamA.dispose();
      beamB.dispose();
    },
    [tower, beamA, beamB],
  );

  useFrame((_, delta) => {
    const spec = PALETTE[crowd];
    target.set(spec.light);
    color.lerp(target, Math.min(1, delta * 2.5));
    level.current += (spec.beam - level.current) * Math.min(1, delta * 2.5);
    if (beams.current && !reduced) beams.current.rotation.y += delta * spec.spin;
    for (const material of [lantern.current, halo.current]) material?.color.copy(color);
    for (const material of [beamA, beamB]) {
      material.uniforms.uColor.value.copy(color);
      material.uniforms.uIntensity.value = level.current * BEAM_GAIN;
    }
    if (halo.current) halo.current.opacity = 0.12 + level.current;
  });

  return (
    <group position={position}>
      <mesh geometry={LIGHTHOUSE} material={tower} />
      <mesh geometry={LANTERN} position={[0, 7.4, 0]}>
        <meshBasicMaterial ref={lantern} toneMapped={false} side={THREE.DoubleSide} />
      </mesh>
      <mesh geometry={LANTERN_HALO} position={[0, 7.4, 0]}>
        <meshBasicMaterial ref={halo} transparent opacity={0.16} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      <group ref={beams} position={[0, 7.4, 0]}>
        <mesh geometry={BEAM} material={beamA} rotation={[0, 0, -0.16]} renderOrder={4} frustumCulled={false} />
        <mesh geometry={BEAM} material={beamB} rotation={[0, Math.PI, -0.16]} renderOrder={4} frustumCulled={false} />
      </group>
    </group>
  );
}

export default function Tavern({ gridSize, barMin, barMax, crowd, fill, reduced }: TavernProps) {
  const half = gridSize / 2;
  const x0 = barMin - half;
  const x1 = barMax - half;
  const side = barMax - barMin;
  const mid = (x0 + x1) / 2;
  const floor = useRef<THREE.MeshBasicMaterial>(null);
  const bulbs = useRef<THREE.InstancedMesh>(null);
  const sign = useRef<THREE.Group>(null);
  const camera = useThree((state) => state.camera);
  const texture = useMemo(() => signTexture(), []);
  const tint = useMemo(() => new THREE.Color(), []);
  const goal = useMemo(() => new THREE.Color(), []);
  const lightColor = useMemo(() => new THREE.Color(PALETTE.idle.light), []);
  const glow = useRef(PALETTE.idle.glow);
  const outline = useMemo(() => createEdgeMaterial({ face: "#0d0c0b", edge: "#6d6a64", ink: "#ece9e2", headInk: 0, width: 1.05, rim: 0.05 }), []);

  useEffect(
    () => () => {
      texture.dispose();
      outline.dispose();
    },
    [texture, outline],
  );

  const walls = useMemo(() => wallSegments(x0, x1, 2), [x0, x1]);
  const structure = useMemo(() => {
    const faces = walls.map(([ax, az, bx, bz]) => {
      const horizontal = Math.abs(bz - az) < 1e-6;
      const length = Math.hypot(bx - ax, bz - az);
      return cuboid((ax + bx) / 2, 0, (az + bz) / 2, horizontal ? length : 0.16, 0.4, horizontal ? 0.16 : length);
    });
    const posts = [
      [x0, x0],
      [x1, x0],
      [x1, x1],
      [x0, x1],
    ].map(([x, z]) => cuboid(x, 0, z, 0.26, 0.9, 0.26));
    const signPost = cuboid(x0, 0.45, x0, 0.1, 0.9, 0.1);
    return facetedGeometry([...faces, ...posts, signPost].map((f) => ({ faces: f, tone: 0 })));
  }, [walls, x0, x1]);

  useEffect(() => () => structure.dispose(), [structure]);

  const bulbSpots = useMemo(() => {
    const spots: [number, number][] = [];
    for (const [ax, az, bx, bz] of walls) {
      const length = Math.hypot(bx - ax, bz - az);
      const steps = Math.max(1, Math.round(length / 0.5));
      for (let i = 0; i <= steps; i += 1) spots.push([ax + ((bx - ax) * i) / steps, az + ((bz - az) * i) / steps]);
    }
    return spots;
  }, [walls]);

  const planks = useMemo(() => {
    const points: number[] = [];
    for (let i = 1; i < side; i += 1) points.push(x0 + i, 0.012, x0, x0 + i, 0.012, x1);
    return new Float32Array(points);
  }, [side, x0, x1]);

  useLayoutEffect(() => {
    const mesh = bulbs.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    bulbSpots.forEach(([x, z], i) => {
      dummy.position.set(x, 0.5, z);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, tint.set("#ffcf7a"));
    });
    mesh.count = bulbSpots.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [bulbSpots, tint]);

  useFrame((_, delta) => {
    const spec = PALETTE[crowd];
    const now = performance.now();
    const k = Math.min(1, delta * 2.5);
    goal.set(spec.light);
    lightColor.lerp(goal, k);
    const pulse = crowd === "crowded" && !reduced ? 0.75 + 0.25 * Math.sin(now * 0.006) : 1;
    glow.current += (spec.glow * pulse * (0.5 + 0.5 * Math.min(1.2, fill)) - glow.current) * k;
    if (floor.current) {
      goal.set(spec.floor);
      floor.current.color.copy(FLOOR_BASE).lerp(goal, glow.current);
    }
    const mesh = bulbs.current;
    if (mesh && mesh.instanceColor) {
      for (let i = 0; i < bulbSpots.length; i += 1) {
        const twinkle = reduced ? 1 : crowd === "crowded" ? (Math.sin(now * 0.012 + i * 0.9) > 0 ? 1 : 0.35) : 0.7 + 0.3 * Math.sin(now * 0.003 + i * 1.3);
        tint.copy(lightColor).multiplyScalar(crowd === "idle" ? 0.35 : twinkle);
        mesh.setColorAt(i, tint);
      }
      mesh.instanceColor.needsUpdate = true;
    }
    if (sign.current) sign.current.rotation.y = Math.atan2(camera.position.x - sign.current.position.x, camera.position.z - sign.current.position.z);
  });

  const lighthouseAt: [number, number, number] = [-half - LIGHTHOUSE_OFFSET, 0, -half - LIGHTHOUSE_OFFSET];

  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position={[mid, 0.005, mid]}>
        <planeGeometry args={[side, side]} />
        <meshBasicMaterial ref={floor} color="#0b0a09" toneMapped={false} />
      </mesh>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[planks, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#2b241b" toneMapped={false} />
      </lineSegments>
      <mesh geometry={structure} material={outline} />
      <instancedMesh ref={bulbs} args={[BULB, undefined, 256]} frustumCulled={false}>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      <group ref={sign} position={[x0, 1.75, x0]}>
        <mesh>
          <planeGeometry args={[3.4, 0.74]} />
          <meshBasicMaterial map={texture} toneMapped={false} side={THREE.DoubleSide} />
        </mesh>
      </group>
      <Lighthouse position={lighthouseAt} crowd={crowd} reduced={reduced} />
    </group>
  );
}
