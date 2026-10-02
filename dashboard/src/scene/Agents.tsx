import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { STRATEGY_COLOR, type AgentState, type Frame } from "../model";

export interface AgentHover {
  agent: AgentState;
  clientX: number;
  clientY: number;
}

interface AgentsProps {
  frame: Frame | null;
  frameKey: string;
  gridSize: number;
  heightRange: [number, number];
  onHover: (hover: AgentHover | null) => void;
}

const CAPACITY = 256;
const RING_CAPACITY = 128;
const MOVE_MS = 680;
const PULSE_MS = 1500;
const MIN_HEIGHT = 0.18;
const MAX_HEIGHT = 6.2;

const PILLAR = new THREE.BoxGeometry(0.62, 1, 0.62).translate(0, 0.5, 0);
const GLOW = new THREE.PlaneGeometry(0.96, 0.96).rotateX(-Math.PI / 2);
const RING = new THREE.RingGeometry(0.44, 0.52, 48).rotateX(-Math.PI / 2);
const DECEPTIVE = new Set(["false_go", "false_stay"]);
const WHITE = new THREE.Color(1, 1, 1);

function ease(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export default function Agents({ frame, frameKey, gridSize, heightRange, onHover }: AgentsProps) {
  const pillars = useRef<THREE.InstancedMesh>(null);
  const glows = useRef<THREE.InstancedMesh>(null);
  const rings = useRef<THREE.InstancedMesh>(null);
  const ringMaterial = useRef<THREE.MeshBasicMaterial>(null);
  const hovered = useRef(-1);
  const frameRef = useRef<Frame | null>(frame);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tint = useMemo(() => new THREE.Color(), []);
  const motion = useRef({
    from: new Float32Array(CAPACITY * 3),
    to: new Float32Array(CAPACITY * 3),
    now: new Float32Array(CAPACITY * 3),
    colorFrom: new Float32Array(CAPACITY * 3),
    colorTo: new Float32Array(CAPACITY * 3),
    colorNow: new Float32Array(CAPACITY * 3),
    glowTo: new Float32Array(CAPACITY * 3),
    count: 0,
    start: 0,
    pulseStart: -Infinity,
    speakers: [] as number[],
  });

  useLayoutEffect(() => {
    const black = new THREE.Color(0, 0, 0);
    for (let i = 0; i < CAPACITY; i += 1) {
      pillars.current?.setColorAt(i, black);
      glows.current?.setColorAt(i, black);
    }
    if (pillars.current?.instanceColor) pillars.current.instanceColor.needsUpdate = true;
    if (glows.current?.instanceColor) glows.current.instanceColor.needsUpdate = true;
  }, []);

  useEffect(() => {
    frameRef.current = frame;
    const m = motion.current;
    const agents = frame?.agents ?? [];
    const count = Math.min(agents.length, CAPACITY);
    const half = gridSize / 2;
    const [lo, hi] = heightRange;
    const span = Math.max(1e-6, hi - lo);
    m.from.set(m.now);
    m.colorFrom.set(m.colorNow);
    for (let i = 0; i < count; i += 1) {
      const agent = agents[i];
      const o = i * 3;
      const height = MIN_HEIGHT + Math.max(0, (agent.cumulative - lo) / span) * (MAX_HEIGHT - MIN_HEIGHT);
      m.to[o] = agent.x - half + 0.5;
      m.to[o + 1] = agent.y - half + 0.5;
      m.to[o + 2] = Number.isFinite(height) ? height : MIN_HEIGHT;
      tint.set(STRATEGY_COLOR[agent.strategy]);
      m.colorTo[o] = tint.r;
      m.colorTo[o + 1] = tint.g;
      m.colorTo[o + 2] = tint.b;
      const glowing = DECEPTIVE.has(agent.strategy);
      m.glowTo[o] = glowing ? tint.r : 0;
      m.glowTo[o + 1] = glowing ? tint.g : 0;
      m.glowTo[o + 2] = glowing ? tint.b : 0;
      if (i >= m.count) {
        m.from[o] = m.to[o];
        m.from[o + 1] = m.to[o + 1];
        m.from[o + 2] = MIN_HEIGHT;
        m.colorFrom[o] = m.colorTo[o];
        m.colorFrom[o + 1] = m.colorTo[o + 1];
        m.colorFrom[o + 2] = m.colorTo[o + 2];
      }
    }
    m.count = count;
    m.start = performance.now();
    m.speakers = agents.slice(0, count).flatMap((agent, i) => (agent.broadcast ? [i] : [])).slice(0, RING_CAPACITY);
    m.pulseStart = m.speakers.length ? performance.now() : -Infinity;
  }, [frameKey, frame, gridSize, heightRange, tint]);

  useFrame(() => {
    const m = motion.current;
    const pillarMesh = pillars.current;
    const glowMesh = glows.current;
    const ringMesh = rings.current;
    if (!pillarMesh || !glowMesh || !ringMesh) return;
    const now = performance.now();
    const t = ease(Math.min(1, (now - m.start) / MOVE_MS));
    for (let i = 0; i < m.count; i += 1) {
      const o = i * 3;
      for (let k = 0; k < 3; k += 1) {
        m.now[o + k] = m.from[o + k] + (m.to[o + k] - m.from[o + k]) * t;
        m.colorNow[o + k] = m.colorFrom[o + k] + (m.colorTo[o + k] - m.colorFrom[o + k]) * t;
      }
      const focus = i === hovered.current;
      dummy.position.set(m.now[o], 0, m.now[o + 1]);
      dummy.scale.set(focus ? 1.35 : 1, m.now[o + 2], focus ? 1.35 : 1);
      dummy.updateMatrix();
      pillarMesh.setMatrixAt(i, dummy.matrix);
      tint.setRGB(m.colorNow[o], m.colorNow[o + 1], m.colorNow[o + 2]);
      if (focus) tint.lerp(WHITE, 0.35);
      pillarMesh.setColorAt(i, tint);
      dummy.position.set(m.now[o], 0.015, m.now[o + 1]);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      glowMesh.setMatrixAt(i, dummy.matrix);
      tint.setRGB(m.glowTo[o] * t, m.glowTo[o + 1] * t, m.glowTo[o + 2] * t);
      glowMesh.setColorAt(i, tint);
    }
    pillarMesh.count = m.count;
    glowMesh.count = m.count;
    pillarMesh.instanceMatrix.needsUpdate = true;
    glowMesh.instanceMatrix.needsUpdate = true;
    if (pillarMesh.instanceColor) pillarMesh.instanceColor.needsUpdate = true;
    if (glowMesh.instanceColor) glowMesh.instanceColor.needsUpdate = true;

    const pulse = (now - m.pulseStart) / PULSE_MS;
    if (pulse >= 0 && pulse < 1 && m.speakers.length) {
      const scale = 1 + 2.6 * ease(pulse);
      m.speakers.forEach((index, slot) => {
        const o = index * 3;
        dummy.position.set(m.now[o], 0.04, m.now[o + 1]);
        dummy.scale.set(scale, 1, scale);
        dummy.updateMatrix();
        ringMesh.setMatrixAt(slot, dummy.matrix);
      });
      ringMesh.count = m.speakers.length;
      ringMesh.instanceMatrix.needsUpdate = true;
      if (ringMaterial.current) ringMaterial.current.opacity = 0.75 * (1 - pulse);
    } else {
      ringMesh.count = 0;
    }
  });

  const handleMove = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    const id = event.instanceId;
    const agent = id != null ? frameRef.current?.agents[id] : undefined;
    hovered.current = id ?? -1;
    onHover(agent ? { agent, clientX: event.nativeEvent.clientX, clientY: event.nativeEvent.clientY } : null);
  };

  const handleOut = () => {
    hovered.current = -1;
    onHover(null);
  };

  return (
    <group>
      <instancedMesh ref={glows} args={[GLOW, undefined, CAPACITY]} frustumCulled={false} renderOrder={1}>
        <meshBasicMaterial transparent opacity={0.5} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </instancedMesh>
      <instancedMesh
        ref={pillars}
        args={[PILLAR, undefined, CAPACITY]}
        frustumCulled={false}
        onPointerMove={handleMove}
        onPointerOut={handleOut}
      >
        <meshLambertMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={rings} args={[RING, undefined, RING_CAPACITY]} frustumCulled={false} renderOrder={2}>
        <meshBasicMaterial ref={ringMaterial} color="#ebebeb" transparent opacity={0} depthWrite={false} toneMapped={false} />
      </instancedMesh>
    </group>
  );
}
