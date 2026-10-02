import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { beamGeometry, createBeamMaterial } from "./beam";
import { createEdgeMaterial } from "./edgeMaterial";
import { figureGeometry, lighthouseGeometry } from "./facets";

interface Walker {
  x: number;
  z: number;
  speed: number;
  phase: number;
  stride: number;
  scale: number;
  yaw: number;
  lit: boolean;
}

const INK = new THREE.Color("#ece9e2");
const LAMP = new THREE.Color("#ffcf7a");
const NEAR = 3;
const FAR = -32;
const FOV = 24;
const EYE = new THREE.Vector3(0, 2.3, 13);
const LOOK = new THREE.Vector3(0, 1.0, -8);
const TOWER = new THREE.Vector3(10.5, 0, -27);
const TOWER_SCALE = 0.78;
const BEACON_BEAM = beamGeometry(2.4, 26);

function random(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function spawn(count: number, seed: number): Walker[] {
  const rand = random(seed);
  return Array.from({ length: count }, () => {
    const idle = rand() < 0.14;
    const dir = rand() < 0.5 ? -1 : 1;
    return {
      x: (rand() * 2 - 1) * 40,
      z: FAR + rand() * (NEAR - FAR),
      speed: idle ? 0 : dir * (0.55 + rand() * 0.8),
      phase: rand() * Math.PI * 2,
      stride: 5.2 + rand() * 1.8,
      scale: 0.9 + rand() * 0.22,
      yaw: idle ? (rand() - 0.5) * 1.8 : (dir * Math.PI) / 2 + (rand() - 0.5) * 0.24,
      lit: rand() < 0.08,
    };
  });
}

function Rig({ reduced }: { reduced: boolean }) {
  const camera = useThree((state) => state.camera);
  const gl = useThree((state) => state.gl);
  const section = useMemo(() => gl.domElement.closest<HTMLElement>("[data-chapter]"), [gl]);
  const pointer = useRef({ x: 0, y: 0 });
  const eased = useRef({ x: 0, y: 0, p: 0 });
  const look = useMemo(() => new THREE.Vector3(), []);

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      pointer.current.x = (event.clientX / window.innerWidth) * 2 - 1;
      pointer.current.y = (event.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);

  useFrame((_, delta) => {
    const p = Number(section?.style.getPropertyValue("--p") || 0);
    const k = reduced ? 1 : 1 - Math.exp(-Math.min(delta, 0.1) * 2.6);
    const e = eased.current;
    e.p += (p - e.p) * k;
    e.x += (pointer.current.x - e.x) * k;
    e.y += (pointer.current.y - e.y) * k;
    camera.position.set(EYE.x - 2.2 + e.p * 4.4 + e.x * 0.7, EYE.y - e.p * 0.4 - e.y * 0.25, EYE.z - e.p * 2.2);
    look.set(camera.position.x * 0.3, LOOK.y, LOOK.z);
    camera.lookAt(look);
  });

  return null;
}

function Floor() {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    const near = new THREE.Color("#3b3a37");
    const far = new THREE.Color("#070707");
    const tint = new THREE.Color();
    const front = NEAR + 9;
    const back = FAR - 16;
    const fade = (z: number) => tint.copy(near).lerp(far, Math.min(1, Math.max(0, (front - z) / (front - back))));
    const segment = (x0: number, z0: number, x1: number, z1: number) => {
      positions.push(x0, 0, z0, x1, 0, z1);
      const a = fade(z0);
      colors.push(a.r, a.g, a.b);
      const b = fade(z1);
      colors.push(b.r, b.g, b.b);
    };
    for (let x = -72; x <= 72; x += 4) segment(x, front, x, back);
    for (const z of [3.5, 0, -4, -9, -16, -25, -38]) segment(-140, z, 140, z);
    const result = new THREE.BufferGeometry();
    result.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    result.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return result;
  }, []);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <lineSegments geometry={geometry} frustumCulled={false}>
      <lineBasicMaterial vertexColors />
    </lineSegments>
  );
}

function Beacon({ reduced }: { reduced: boolean }) {
  const geometry = useMemo(() => lighthouseGeometry(), []);
  const material = useMemo(() => createEdgeMaterial({ edge: "#77746e", face: "#0c0c0c", ink: "#d9d5cc", headInk: 0.55, width: 1.05, rim: 0.05 }), []);
  const halo = useRef<THREE.Mesh>(null);
  const beam = useRef<THREE.Group>(null);
  const light = useMemo(() => createBeamMaterial({ color: "#ffcf7a", intensity: 0.16, falloff: 1.9, softness: 1.4 }), []);

  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
      light.dispose();
    },
    [geometry, material, light],
  );

  useFrame(({ clock }) => {
    const t = reduced ? 0 : clock.elapsedTime;
    if (halo.current) halo.current.scale.setScalar(1 + Math.sin(t * 1.6) * 0.12);
    if (beam.current) beam.current.rotation.y = t * 0.45;
  });

  return (
    <group position={TOWER} scale={TOWER_SCALE}>
      <mesh geometry={geometry} material={material} />
      <mesh position={[0, 7.4, 0]}>
        <octahedronGeometry args={[0.32]} />
        <meshBasicMaterial color="#ffcf7a" toneMapped={false} />
      </mesh>
      <mesh ref={halo} position={[0, 7.4, 0]}>
        <icosahedronGeometry args={[1.1, 0]} />
        <meshBasicMaterial color="#ffcf7a" transparent opacity={0.12} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      <group ref={beam} position={[0, 7.4, 0]}>
        {[0, Math.PI].map((angle) => (
          <mesh key={angle} geometry={BEACON_BEAM} material={light} rotation={[0, angle, -0.08]} renderOrder={4} frustumCulled={false} />
        ))}
      </group>
    </group>
  );
}

function Walkers({ count, seed, reduced }: { count: number; seed: number; reduced: boolean }) {
  const geometry = useMemo(() => figureGeometry(), []);
  const material = useMemo(() => createEdgeMaterial({ face: "#101010", headInk: 0.22, width: 1.2, rim: 0.14 }), []);
  const walkers = useMemo(() => spawn(count, seed), [count, seed]);
  const mesh = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tan = Math.tan(THREE.MathUtils.degToRad(FOV / 2));

  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );

  useEffect(() => {
    const target = mesh.current;
    if (!target) return;
    const color = new THREE.Color();
    walkers.forEach((w, i) => {
      const near = (w.z - FAR) / (NEAR - FAR);
      if (w.lit) color.copy(LAMP).multiplyScalar(0.55 + 0.45 * near);
      else color.copy(INK).multiplyScalar(0.1 + 0.9 * Math.pow(near, 1.35));
      target.setColorAt(i, color);
    });
    if (target.instanceColor) target.instanceColor.needsUpdate = true;
  }, [walkers]);

  useFrame((state, delta) => {
    const target = mesh.current;
    if (!target) return;
    const dt = reduced ? 0 : Math.min(delta, 0.05);
    const aspect = state.size.width / Math.max(1, state.size.height);
    const t = state.clock.elapsedTime;
    walkers.forEach((w, i) => {
      const half = (EYE.z - w.z) * tan * aspect + 4;
      w.x = ((((w.x + w.speed * dt + half) % (half * 2)) + half * 2) % (half * 2)) - half;
      w.phase += Math.abs(w.speed) * dt * w.stride;
      const walking = w.speed !== 0;
      const bob = walking ? Math.abs(Math.sin(w.phase)) * 0.055 : 0;
      const sway = walking ? Math.sin(w.phase) * 0.05 : reduced ? 0 : Math.sin(t * 0.7 + w.phase) * 0.02;
      dummy.position.set(w.x, bob, w.z);
      dummy.rotation.set(walking ? 0.07 : 0, w.yaw, sway, "YXZ");
      dummy.scale.setScalar(w.scale);
      dummy.updateMatrix();
      target.setMatrixAt(i, dummy.matrix);
    });
    target.instanceMatrix.needsUpdate = true;
  });

  return <instancedMesh ref={mesh} args={[geometry, material, count]} frustumCulled={false} />;
}

export default function CrowdScene({ count = 140, seed = 1994 }: { count?: number; seed?: number }) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(true);
  const reduced = useMemo(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches, []);

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    const sight = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: "160px 0px" });
    sight.observe(element);
    return () => sight.disconnect();
  }, []);

  return (
    <div ref={host} className="crowd" aria-hidden="true">
      <Canvas
        flat
        dpr={[1, 2]}
        frameloop={reduced ? "demand" : visible ? "always" : "never"}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        camera={{ fov: FOV, near: 0.1, far: 160, position: [EYE.x, EYE.y, EYE.z] }}
        style={{ pointerEvents: "none" }}
      >
        <Rig reduced={reduced} />
        <Floor />
        <Beacon reduced={reduced} />
        <Walkers count={count} seed={seed} reduced={reduced} />
      </Canvas>
    </div>
  );
}
