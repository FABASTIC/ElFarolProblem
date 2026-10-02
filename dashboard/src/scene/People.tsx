import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef, type MutableRefObject } from "react";
import * as THREE from "three";
import type { Frame } from "../model";
import { lensColor, lensKeyOf, type Lens } from "../people";

export const CAPACITY = 256;

export interface Motion {
  positions: Float32Array;
  ids: Int32Array;
  count: number;
  indexById: Map<number, number>;
}

export function createMotion(): Motion {
  return { positions: new Float32Array(CAPACITY * 3), ids: new Int32Array(CAPACITY), count: 0, indexById: new Map() };
}

interface PeopleProps {
  frame: Frame | null;
  frameKey: string;
  gridSize: number;
  lens: Lens;
  isolate: string | null;
  wealthSpan: number;
  selectedId: number | null;
  hoveredId: MutableRefObject<number | null>;
  motion: MutableRefObject<Motion>;
  reduced: boolean;
  onHover: (id: number | null) => void;
  onSelect: (id: number) => void;
}

const BODY_R = 0.25;
const BODY_LEN = 0.44;
const BODY = new THREE.CapsuleGeometry(BODY_R, BODY_LEN, 4, 10).translate(0, BODY_R + BODY_LEN / 2, 0);
const HEAD = new THREE.SphereGeometry(0.2, 14, 10).translate(0, BODY_LEN + BODY_R * 2 + 0.17, 0);
const GLOW = new THREE.CircleGeometry(0.56, 28).rotateX(-Math.PI / 2);
const RING = new THREE.RingGeometry(0.42, 0.5, 40).rotateX(-Math.PI / 2);
const BEAM = new THREE.CylinderGeometry(0.06, 0.06, 7, 10, 1, true).translate(0, 3.5, 0);
const HALO = new THREE.RingGeometry(0.5, 0.62, 48).rotateX(-Math.PI / 2);
const DECEPTIVE = new Set(["false_go", "false_stay"]);
const DIM = new THREE.Color("#161b24");
const HEAD_TONE = new THREE.Color("#e9e1d3");
const WHITE = new THREE.Color("#ffffff");
const COLOR_MS = 420;

function ease(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function jitter(id: number): number {
  const x = Math.sin(id * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

export default function People({
  frame,
  frameKey,
  gridSize,
  lens,
  isolate,
  wealthSpan,
  selectedId,
  hoveredId,
  motion,
  reduced,
  onHover,
  onSelect,
}: PeopleProps) {
  const bodies = useRef<THREE.InstancedMesh>(null);
  const heads = useRef<THREE.InstancedMesh>(null);
  const glows = useRef<THREE.InstancedMesh>(null);
  const rings = useRef<THREE.InstancedMesh>(null);
  const ringMaterial = useRef<THREE.MeshBasicMaterial>(null);
  const beam = useRef<THREE.Mesh>(null);
  const halo = useRef<THREE.Mesh>(null);
  const frameRef = useRef<Frame | null>(frame);
  const selectedRef = useRef<number | null>(selectedId);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tint = useMemo(() => new THREE.Color(), []);
  const state = useRef({
    from: new Float32Array(CAPACITY * 2),
    to: new Float32Array(CAPACITY * 2),
    start: new Float64Array(CAPACITY),
    duration: new Float32Array(CAPACITY),
    hop: new Float32Array(CAPACITY),
    colorFrom: new Float32Array(CAPACITY * 3),
    colorTo: new Float32Array(CAPACITY * 3),
    colorNow: new Float32Array(CAPACITY * 3),
    colorStart: 0,
    glow: new Float32Array(CAPACITY * 3),
    dim: new Float32Array(CAPACITY),
    dimTo: new Float32Array(CAPACITY),
    speakers: [] as number[],
    count: 0,
    preview: false,
  });

  selectedRef.current = selectedId;

  useLayoutEffect(() => {
    const sphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), gridSize);
    for (const mesh of [bodies.current, heads.current]) {
      if (!mesh) continue;
      mesh.boundingSphere = sphere;
      for (let i = 0; i < CAPACITY; i += 1) mesh.setColorAt(i, DIM);
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
    if (glows.current) {
      for (let i = 0; i < CAPACITY; i += 1) glows.current.setColorAt(i, DIM);
      if (glows.current.instanceColor) glows.current.instanceColor.needsUpdate = true;
    }
  }, [gridSize]);

  useEffect(() => {
    frameRef.current = frame;
    const s = state.current;
    const m = motion.current;
    const agents = frame?.agents ?? [];
    const count = Math.min(agents.length, CAPACITY);
    const half = gridSize / 2;
    const now = performance.now();
    const preview = !!frame?.preview;
    for (let i = 0; i < count; i += 1) {
      const agent = agents[i];
      const o = i * 2;
      const tx = agent.x - half + 0.5;
      const tz = agent.y - half + 0.5;
      const known = i < s.count && m.ids[i] === agent.id;
      const fx = known ? m.positions[i * 3] : tx;
      const fz = known ? m.positions[i * 3 + 2] : tz;
      s.from[o] = fx;
      s.from[o + 1] = fz;
      s.to[o] = tx;
      s.to[o + 1] = tz;
      const distance = Math.hypot(tx - fx, tz - fz);
      s.start[i] = now + (reduced || !known ? 0 : jitter(agent.id) * 320);
      s.duration[i] = reduced ? 1 : Math.min(1700, 420 + distance * 52);
      s.hop[i] = distance > 0.6 && !reduced ? Math.min(1.4, 0.3 + distance * 0.06) : 0;
      if (!known) {
        m.positions[i * 3] = tx;
        m.positions[i * 3 + 2] = tz;
      }
      m.ids[i] = agent.id;
      const glowing = !preview && DECEPTIVE.has(agent.strategy);
      tint.set(lensColor(agent, "strategy", 1));
      s.glow[i * 3] = glowing ? tint.r : 0;
      s.glow[i * 3 + 1] = glowing ? tint.g : 0;
      s.glow[i * 3 + 2] = glowing ? tint.b : 0;
    }
    m.indexById = new Map(agents.slice(0, count).map((a, i) => [a.id, i]));
    m.count = count;
    s.count = count;
    s.preview = preview;
    s.speakers = preview ? [] : agents.slice(0, count).flatMap((a, i) => (a.broadcast ? [i] : []));
  }, [frameKey, frame, gridSize, reduced, motion, tint]);

  useEffect(() => {
    const s = state.current;
    const agents = frame?.agents ?? [];
    const count = Math.min(agents.length, CAPACITY);
    s.colorFrom.set(s.colorNow);
    for (let i = 0; i < count; i += 1) {
      const agent = agents[i];
      tint.set(lensColor(agent, lens, wealthSpan, !!frame?.preview));
      s.colorTo[i * 3] = tint.r;
      s.colorTo[i * 3 + 1] = tint.g;
      s.colorTo[i * 3 + 2] = tint.b;
      s.dimTo[i] = isolate && !frame?.preview && lensKeyOf(agent, lens) !== isolate ? 1 : 0;
    }
    s.colorStart = performance.now();
  }, [frameKey, frame, lens, isolate, wealthSpan, tint]);

  useFrame((_, delta) => {
    const s = state.current;
    const m = motion.current;
    const bodyMesh = bodies.current;
    const headMesh = heads.current;
    const glowMesh = glows.current;
    const ringMesh = rings.current;
    if (!bodyMesh || !headMesh || !glowMesh || !ringMesh) return;
    const now = performance.now();
    const colorT = ease(Math.min(1, (now - s.colorStart) / COLOR_MS));
    const hovered = hoveredId.current;
    const selected = selectedRef.current;
    const fade = Math.min(1, delta * 6);
    let selectedIndex = -1;
    for (let i = 0; i < s.count; i += 1) {
      const o2 = i * 2;
      const o3 = i * 3;
      const t = s.duration[i] <= 1 ? 1 : Math.max(0, Math.min(1, (now - s.start[i]) / s.duration[i]));
      const e = ease(t);
      const x = s.from[o2] + (s.to[o2] - s.from[o2]) * e;
      const z = s.from[o2 + 1] + (s.to[o2 + 1] - s.from[o2 + 1]) * e;
      const id = m.ids[i];
      const bob = reduced ? 0 : s.preview ? 0.05 * Math.sin(now * 0.0021 + id * 1.7) : 0.022 * Math.sin(now * 0.004 + id);
      const y = s.hop[i] * Math.sin(Math.PI * e) + bob;
      m.positions[o3] = x;
      m.positions[o3 + 1] = y;
      m.positions[o3 + 2] = z;
      s.dim[i] += (s.dimTo[i] - s.dim[i]) * fade;
      for (let k = 0; k < 3; k += 1) {
        s.colorNow[o3 + k] = s.colorFrom[o3 + k] + (s.colorTo[o3 + k] - s.colorFrom[o3 + k]) * colorT;
      }
      const focus = id === hovered || id === selected;
      if (id === selected) selectedIndex = i;
      const scale = focus ? 1.22 : 1;
      dummy.position.set(x, y, z);
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      bodyMesh.setMatrixAt(i, dummy.matrix);
      headMesh.setMatrixAt(i, dummy.matrix);
      tint.setRGB(s.colorNow[o3], s.colorNow[o3 + 1], s.colorNow[o3 + 2]);
      if (focus) tint.lerp(WHITE, 0.28);
      tint.lerp(DIM, s.dim[i] * 0.82);
      bodyMesh.setColorAt(i, tint);
      tint.copy(HEAD_TONE).lerp(DIM, s.dim[i] * 0.82);
      headMesh.setColorAt(i, tint);
      dummy.position.set(x, 0.015, z);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      glowMesh.setMatrixAt(i, dummy.matrix);
      const pulse = reduced ? 1 : 0.75 + 0.25 * Math.sin(now * 0.005 + id);
      const glow = colorT * pulse * (1 - s.dim[i]);
      tint.setRGB(s.glow[o3] * glow, s.glow[o3 + 1] * glow, s.glow[o3 + 2] * glow);
      glowMesh.setColorAt(i, tint);
    }
    bodyMesh.count = s.count;
    headMesh.count = s.count;
    glowMesh.count = s.count;
    for (const mesh of [bodyMesh, headMesh, glowMesh]) {
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }

    const cycle = reduced ? 0.5 : ((now / 2200) % 1 + 1) % 1;
    if (s.speakers.length) {
      const scale = 1 + 2.2 * ease(cycle);
      s.speakers.forEach((index, slot) => {
        const o = index * 3;
        dummy.position.set(m.positions[o], 0.04, m.positions[o + 2]);
        dummy.scale.set(scale, 1, scale);
        dummy.updateMatrix();
        ringMesh.setMatrixAt(slot, dummy.matrix);
      });
      ringMesh.count = Math.min(s.speakers.length, CAPACITY);
      ringMesh.instanceMatrix.needsUpdate = true;
      if (ringMaterial.current) ringMaterial.current.opacity = 0.55 * (1 - cycle);
    } else {
      ringMesh.count = 0;
    }

    if (beam.current && halo.current) {
      const visible = selectedIndex >= 0;
      beam.current.visible = visible;
      halo.current.visible = visible;
      if (visible) {
        const o = selectedIndex * 3;
        beam.current.position.set(m.positions[o], 0, m.positions[o + 2]);
        halo.current.position.set(m.positions[o], 0.03, m.positions[o + 2]);
        const spin = reduced ? 1 : 1 + 0.08 * Math.sin(now * 0.006);
        halo.current.scale.set(spin, 1, spin);
      }
    }
  });

  const handleMove = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    const index = event.instanceId;
    const agent = index != null ? frameRef.current?.agents[index] : undefined;
    const id = agent && !frameRef.current?.preview ? agent.id : null;
    if (hoveredId.current !== id) {
      hoveredId.current = id;
      onHover(id);
    }
  };

  const handleOut = () => {
    if (hoveredId.current !== null) {
      hoveredId.current = null;
      onHover(null);
    }
  };

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    if (event.delta > 6) return;
    event.stopPropagation();
    const index = event.instanceId;
    const agent = index != null ? frameRef.current?.agents[index] : undefined;
    if (agent && !frameRef.current?.preview) onSelect(agent.id);
  };

  return (
    <group>
      <instancedMesh ref={glows} args={[GLOW, undefined, CAPACITY]} frustumCulled={false} renderOrder={1}>
        <meshBasicMaterial transparent opacity={0.55} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </instancedMesh>
      <instancedMesh
        ref={bodies}
        args={[BODY, undefined, CAPACITY]}
        frustumCulled={false}
        onPointerMove={handleMove}
        onPointerOut={handleOut}
        onClick={handleClick}
      >
        <meshLambertMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={heads} args={[HEAD, undefined, CAPACITY]} frustumCulled={false}>
        <meshLambertMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={rings} args={[RING, undefined, CAPACITY]} frustumCulled={false} renderOrder={2}>
        <meshBasicMaterial ref={ringMaterial} color="#f2ead8" transparent opacity={0} depthWrite={false} toneMapped={false} />
      </instancedMesh>
      <mesh ref={beam} geometry={BEAM} visible={false} renderOrder={3}>
        <meshBasicMaterial color="#f6f1e4" transparent opacity={0.22} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      <mesh ref={halo} geometry={HALO} visible={false} renderOrder={3}>
        <meshBasicMaterial color="#f6f1e4" transparent opacity={0.9} depthWrite={false} toneMapped={false} />
      </mesh>
    </group>
  );
}
