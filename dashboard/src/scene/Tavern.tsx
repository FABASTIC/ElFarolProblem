import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
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

const PALETTE: Record<CrowdState, { light: string; floor: string; beam: number; glow: number; spin: number; lamp: number }> = {
  idle: { light: "#9fb3d9", floor: "#6b7fa6", beam: 0.035, glow: 0.05, spin: 0.18, lamp: 1.5 },
  comfortable: { light: "#ffcf7a", floor: "#ffb347", beam: 0.085, glow: 0.22, spin: 0.32, lamp: 7 },
  crowded: { light: "#ff5a4f", floor: "#d03b3b", beam: 0.12, glow: 0.3, spin: 0.62, lamp: 9 },
};

const BULB = new THREE.SphereGeometry(0.075, 8, 6);
const BEAM = new THREE.ConeGeometry(4.2, 40, 32, 1, true).translate(0, -20, 0).rotateZ(Math.PI / 2);

function signTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = "#1a1210";
    ctx.fillRect(0, 0, 512, 128);
    ctx.strokeStyle = "#ffcf7a";
    ctx.lineWidth = 6;
    ctx.strokeRect(8, 8, 496, 112);
    ctx.fillStyle = "#ffe2a8";
    ctx.font = "700 64px 'JetBrains Mono', Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.shadowColor = "#ffb347";
    ctx.shadowBlur = 18;
    ctx.fillText("EL FAROL", 256, 68);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
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
  const beamA = useRef<THREE.MeshBasicMaterial>(null);
  const beamB = useRef<THREE.MeshBasicMaterial>(null);
  const color = useMemo(() => new THREE.Color(), []);
  const target = useMemo(() => new THREE.Color(), []);
  const level = useRef(PALETTE.idle.beam);

  useFrame((_, delta) => {
    const spec = PALETTE[crowd];
    target.set(spec.light);
    color.lerp(target, Math.min(1, delta * 2.5));
    level.current += (spec.beam - level.current) * Math.min(1, delta * 2.5);
    if (beams.current && !reduced) beams.current.rotation.y += delta * spec.spin;
    for (const material of [lantern.current, halo.current, beamA.current, beamB.current]) material?.color.copy(color);
    if (beamA.current) beamA.current.opacity = level.current;
    if (beamB.current) beamB.current.opacity = level.current;
    if (halo.current) halo.current.opacity = 0.18 + level.current;
  });

  const stripes = [
    { y: 0.6, h: 2.2, r0: 1.05, r1: 0.92, c: "#e7e1d4" },
    { y: 2.8, h: 2.0, r0: 0.92, r1: 0.8, c: "#b5433c" },
    { y: 4.8, h: 2.0, r0: 0.8, r1: 0.68, c: "#e7e1d4" },
  ];

  return (
    <group position={position}>
      <mesh position={[0, 0.3, 0]}>
        <cylinderGeometry args={[1.7, 1.9, 0.6, 18]} />
        <meshLambertMaterial color="#3a3d47" toneMapped={false} />
      </mesh>
      {stripes.map((s) => (
        <mesh key={s.y} position={[0, s.y + s.h / 2, 0]}>
          <cylinderGeometry args={[s.r1, s.r0, s.h, 20]} />
          <meshLambertMaterial color={s.c} toneMapped={false} />
        </mesh>
      ))}
      <mesh position={[0, 6.88, 0]}>
        <cylinderGeometry args={[0.98, 0.98, 0.14, 24]} />
        <meshLambertMaterial color="#2b2e37" toneMapped={false} />
      </mesh>
      <mesh position={[0, 7.4, 0]}>
        <cylinderGeometry args={[0.5, 0.5, 0.9, 16]} />
        <meshBasicMaterial ref={lantern} toneMapped={false} />
      </mesh>
      <mesh position={[0, 8.2, 0]}>
        <coneGeometry args={[0.72, 0.75, 16]} />
        <meshLambertMaterial color="#b5433c" toneMapped={false} />
      </mesh>
      <mesh position={[0, 7.4, 0]}>
        <sphereGeometry args={[1.6, 20, 14]} />
        <meshBasicMaterial ref={halo} transparent opacity={0.2} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      <group ref={beams} position={[0, 7.4, 0]}>
        <mesh geometry={BEAM} rotation={[0, 0, -0.16]}>
          <meshBasicMaterial ref={beamA} transparent side={THREE.DoubleSide} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
        </mesh>
        <mesh geometry={BEAM} rotation={[0, Math.PI, -0.16]}>
          <meshBasicMaterial ref={beamB} transparent side={THREE.DoubleSide} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
        </mesh>
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
  const floor = useRef<THREE.MeshLambertMaterial>(null);
  const lamp = useRef<THREE.PointLight>(null);
  const bulbs = useRef<THREE.InstancedMesh>(null);
  const sign = useRef<THREE.Group>(null);
  const camera = useThree((state) => state.camera);
  const texture = useMemo(() => signTexture(), []);
  const tint = useMemo(() => new THREE.Color(), []);
  const goal = useMemo(() => new THREE.Color(), []);
  const lightColor = useMemo(() => new THREE.Color(PALETTE.idle.light), []);

  useEffect(() => () => texture.dispose(), [texture]);

  const walls = useMemo(() => wallSegments(x0, x1, 2), [x0, x1]);
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
      dummy.position.set(x, 0.52, z);
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
    if (lamp.current) {
      lamp.current.color.copy(lightColor);
      const throb = crowd === "crowded" && !reduced ? 0.8 + 0.2 * Math.sin(now * 0.008) : 1;
      lamp.current.intensity += (spec.lamp * (0.6 + 0.4 * Math.min(1.2, fill)) * throb - lamp.current.intensity) * k;
    }
    if (floor.current) {
      goal.set(spec.floor);
      floor.current.emissive.lerp(goal, k);
      const pulse = crowd === "crowded" && !reduced ? 0.75 + 0.25 * Math.sin(now * 0.006) : 1;
      floor.current.emissiveIntensity += (spec.glow * pulse * (0.5 + 0.5 * Math.min(1.2, fill)) - floor.current.emissiveIntensity) * k;
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
    if (sign.current) sign.current.rotation.y = Math.atan2(camera.position.x, camera.position.z);
  });

  const lighthouseAt: [number, number, number] = [-half - LIGHTHOUSE_OFFSET, 0, -half - LIGHTHOUSE_OFFSET];

  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position={[mid, 0.005, mid]}>
        <planeGeometry args={[side, side]} />
        <meshLambertMaterial ref={floor} color="#3b2a1d" emissive="#6b7fa6" emissiveIntensity={0.05} toneMapped={false} />
      </mesh>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[planks, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#24180f" toneMapped={false} />
      </lineSegments>
      {walls.map(([ax, az, bx, bz], i) => {
        const length = Math.hypot(bx - ax, bz - az);
        const horizontal = Math.abs(bz - az) < 1e-6;
        return (
          <mesh key={i} position={[(ax + bx) / 2, 0.2, (az + bz) / 2]}>
            <boxGeometry args={[horizontal ? length : 0.16, 0.4, horizontal ? 0.16 : length]} />
            <meshLambertMaterial color="#4a3528" toneMapped={false} />
          </mesh>
        );
      })}
      <instancedMesh ref={bulbs} args={[BULB, undefined, 256]} frustumCulled={false}>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      {[
        [x0, x0],
        [x1, x0],
        [x1, x1],
        [x0, x1],
      ].map(([x, z]) => (
        <mesh key={`${x}:${z}`} position={[x, 0.45, z]}>
          <boxGeometry args={[0.26, 0.9, 0.26]} />
          <meshLambertMaterial color="#5a4030" toneMapped={false} />
        </mesh>
      ))}
      <group ref={sign} position={[x0, 1.75, x0]}>
        <mesh position={[0, 0, 0]}>
          <planeGeometry args={[3.4, 0.85]} />
          <meshBasicMaterial map={texture} toneMapped={false} side={THREE.DoubleSide} />
        </mesh>
        <mesh position={[0, -0.85, 0]}>
          <boxGeometry args={[0.1, 0.9, 0.1]} />
          <meshLambertMaterial color="#5a4030" toneMapped={false} />
        </mesh>
      </group>
      <pointLight ref={lamp} position={[mid, 3.2, mid]} distance={20} decay={1} intensity={1.5} />
      <Lighthouse position={lighthouseAt} crowd={crowd} reduced={reduced} />
    </group>
  );
}
