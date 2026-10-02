import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";

interface GroundProps {
  gridSize: number;
  barMin: number;
  barMax: number;
}

export const BORDER = 8;
export const LIGHTHOUSE_OFFSET = 3.4;

const TILE = new THREE.PlaneGeometry(0.94, 0.94).rotateX(-Math.PI / 2);
const FOLIAGE = new THREE.ConeGeometry(0.62, 1.5, 7).translate(0, 1.05, 0);
const TRUNK = new THREE.CylinderGeometry(0.07, 0.09, 0.4, 6).translate(0, 0.2, 0);
const HOUSE = new THREE.BoxGeometry(1.7, 0.95, 1.4).translate(0, 0.475, 0);
const ROOF = new THREE.ConeGeometry(1.32, 0.75, 4).rotateY(Math.PI / 4).scale(1, 1, 0.84).translate(0, 1.32, 0);
const WINDOW = new THREE.PlaneGeometry(0.28, 0.26);
const POST = new THREE.CylinderGeometry(0.045, 0.06, 1.7, 6).translate(0, 0.85, 0);
const BULB = new THREE.SphereGeometry(0.13, 10, 8).translate(0, 1.78, 0);
const POOL = new THREE.CircleGeometry(1.6, 28).rotateX(-Math.PI / 2);

function hash(x: number, y: number, salt = 0): number {
  const v = Math.sin(x * 127.1 + y * 311.7 + salt * 74.7) * 43758.5453;
  return v - Math.floor(v);
}

interface Placement {
  x: number;
  z: number;
  s: number;
  r: number;
  c: number;
}

function useInstances(ref: React.RefObject<THREE.InstancedMesh | null>, items: Placement[], palette: string[], lift = 0) {
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    items.forEach((item, i) => {
      dummy.position.set(item.x, lift, item.z);
      dummy.rotation.set(0, item.r, 0);
      dummy.scale.set(item.s, item.s, item.s);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, color.set(palette[item.c % palette.length]));
    });
    mesh.count = items.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [ref, items, palette, lift]);
}

const GRASS = ["#1b2925", "#1e2d27", "#182620", "#203029"];
const ROAD = ["#2e323d", "#323641", "#2b2f39"];
const PLAZA = ["#3c3835", "#403b38", "#383431"];
const LEAVES = ["#1d3a2b", "#214031", "#183326", "#264535"];
const BARK = ["#3a2a1f"];
const WALLS = ["#2a2e39", "#2e3240", "#272b35"];
const ROOFS = ["#5b3a35", "#4d3a44", "#3f4252", "#5a4632"];
const LIT = ["#ffcf7a", "#ffd994", "#f6b85c"];
const POSTS = ["#3a3f4b"];
const BULBS = ["#ffe2a8"];

export default function Ground({ gridSize, barMin, barMax }: GroundProps) {
  const half = gridSize / 2;
  const center = (barMin + barMax) / 2;
  const roadA = Math.floor(center) - 1;
  const roadB = Math.floor(center);
  const tiles = useRef<THREE.InstancedMesh>(null);
  const leaves = useRef<THREE.InstancedMesh>(null);
  const trunks = useRef<THREE.InstancedMesh>(null);
  const houses = useRef<THREE.InstancedMesh>(null);
  const roofs = useRef<THREE.InstancedMesh>(null);
  const windows = useRef<THREE.InstancedMesh>(null);
  const posts = useRef<THREE.InstancedMesh>(null);
  const bulbs = useRef<THREE.InstancedMesh>(null);
  const pools = useRef<THREE.InstancedMesh>(null);

  const layout = useMemo(() => {
    const isRoad = (x: number, y: number) => x === roadA || x === roadB || y === roadA || y === roadB;
    const inBar = (x: number, y: number) => x >= barMin && x < barMax && y >= barMin && y < barMax;
    const inPlaza = (x: number, y: number) => x >= barMin - 2 && x < barMax + 2 && y >= barMin - 2 && y < barMax + 2;
    const grass: Placement[] = [];
    const roads: Placement[] = [];
    const plaza: Placement[] = [];
    for (let x = -BORDER; x < gridSize + BORDER; x += 1) {
      for (let y = -BORDER; y < gridSize + BORDER; y += 1) {
        const inside = x >= 0 && y >= 0 && x < gridSize && y < gridSize;
        if (inside && inBar(x, y)) continue;
        const item = { x: x - half + 0.5, z: y - half + 0.5, s: 1, r: 0, c: Math.floor(hash(x, y) * 4) };
        if (inside && inPlaza(x, y)) plaza.push(item);
        else if (isRoad(x, y) && (inside || x === roadA || x === roadB || y === roadA || y === roadB)) roads.push(item);
        else if (inside) grass.push(item);
      }
    }
    const trees: Placement[] = [];
    const homes: Placement[] = [];
    const panes: Placement[] = [];
    const nearRoad = (x: number, y: number, pad: number) => Math.abs(x - center) < pad || Math.abs(y - center) < pad;
    for (let side = 0; side < 4; side += 1) {
      for (let k = 3; k < gridSize - 2; k += 6) {
        if (Math.abs(k + 1 - center) < 4) continue;
        const depth = 2.6 + hash(k, side, 3) * 1.2;
        const along = k + 1 + (hash(k, side, 5) - 0.5) * 1.5;
        let x = along;
        let y = -depth;
        let facing = 0;
        if (side === 1) {
          x = gridSize + depth;
          y = along;
          facing = -Math.PI / 2;
        } else if (side === 2) {
          x = along;
          y = gridSize + depth;
          facing = Math.PI;
        } else if (side === 3) {
          x = -depth;
          y = along;
          facing = Math.PI / 2;
        }
        const wx = x - half;
        const wz = y - half;
        const scale = 0.92 + hash(k, side, 7) * 0.22;
        homes.push({ x: wx, z: wz, s: scale, r: facing, c: Math.floor(hash(k, side, 9) * 4) });
        const nx = Math.sin(facing);
        const nz = Math.cos(facing);
        const tx = Math.cos(facing);
        const tz = -Math.sin(facing);
        [-0.42, 0.42].forEach((offset, w) => {
          if (hash(k, side, 11 + w) < 0.3) return;
          const reach = 0.7 * scale + 0.02;
          panes.push({ x: wx + nx * reach + tx * offset * scale, z: wz + nz * reach + tz * offset * scale, s: 1, r: facing, c: w });
        });
      }
    }
    for (let x = -BORDER; x < gridSize + BORDER; x += 1) {
      for (let y = -BORDER; y < gridSize + BORDER; y += 1) {
        const inside = x >= -1 && y >= -1 && x <= gridSize && y <= gridSize;
        if (inside || nearRoad(x + 0.5, y + 0.5, 2.6)) continue;
        if (hash(x, y, 13) > 0.34) continue;
        const wx = x - half + 0.5 + (hash(x, y, 17) - 0.5) * 0.6;
        const wz = y - half + 0.5 + (hash(x, y, 19) - 0.5) * 0.6;
        if (homes.some((h) => Math.abs(h.x - wx) < 1.8 && Math.abs(h.z - wz) < 1.8)) continue;
        if (Math.hypot(wx + half + LIGHTHOUSE_OFFSET, wz + half + LIGHTHOUSE_OFFSET) < 3.4) continue;
        trees.push({ x: wx, z: wz, s: 0.75 + hash(x, y, 23) * 0.6, r: hash(x, y, 29) * Math.PI, c: Math.floor(hash(x, y, 31) * 4) });
      }
    }
    const lamps: Placement[] = [];
    const plazaLo = barMin - 2 - half;
    const plazaHi = barMax + 2 - half;
    [
      [plazaLo, plazaLo],
      [plazaHi, plazaLo],
      [plazaLo, plazaHi],
      [plazaHi, plazaHi],
    ].forEach(([x, z]) => lamps.push({ x, z, s: 1, r: 0, c: 0 }));
    const gate = center - half;
    [
      [gate - 1.5, -half],
      [gate + 1.5, -half],
      [gate - 1.5, half],
      [gate + 1.5, half],
      [-half, gate - 1.5],
      [-half, gate + 1.5],
      [half, gate - 1.5],
      [half, gate + 1.5],
    ].forEach(([x, z]) => lamps.push({ x, z, s: 0.9, r: 0, c: 0 }));
    return { grass, roads, plaza, trees, homes, panes, lamps };
  }, [gridSize, barMin, barMax, half, center, roadA, roadB]);

  const allTiles = useMemo(
    () => [
      ...layout.grass,
      ...layout.roads.map((t) => ({ ...t, c: 4 + (t.c % 3) })),
      ...layout.plaza.map((t) => ({ ...t, c: 7 + (t.c % 3) })),
    ],
    [layout],
  );
  const tilePalette = useMemo(() => [...GRASS, ...ROAD, ...PLAZA], []);

  useInstances(tiles, allTiles, tilePalette, 0.001);
  useInstances(leaves, layout.trees, LEAVES);
  useInstances(trunks, layout.trees, BARK);
  useInstances(houses, layout.homes, WALLS);
  useInstances(roofs, layout.homes, ROOFS);
  useInstances(windows, layout.panes, LIT, 0.48);
  useInstances(posts, layout.lamps, POSTS);
  useInstances(bulbs, layout.lamps, BULBS);
  useInstances(pools, layout.lamps, BULBS, 0.012);

  const outer = gridSize + BORDER * 2 + 6;
  const tileCount = (gridSize + BORDER * 2) ** 2;

  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.03, 0]}>
        <planeGeometry args={[outer, outer]} />
        <meshBasicMaterial color="#0b0f13" toneMapped={false} />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.02, 0]}>
        <planeGeometry args={[gridSize, gridSize]} />
        <meshBasicMaterial color="#0e1316" toneMapped={false} />
      </mesh>
      <instancedMesh ref={tiles} args={[TILE, undefined, tileCount]} frustumCulled={false}>
        <meshLambertMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={trunks} args={[TRUNK, undefined, 2048]} frustumCulled={false}>
        <meshLambertMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={leaves} args={[FOLIAGE, undefined, 2048]} frustumCulled={false}>
        <meshLambertMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={houses} args={[HOUSE, undefined, 128]} frustumCulled={false}>
        <meshLambertMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={roofs} args={[ROOF, undefined, 128]} frustumCulled={false}>
        <meshLambertMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={windows} args={[WINDOW, undefined, 256]} frustumCulled={false}>
        <meshBasicMaterial toneMapped={false} side={THREE.DoubleSide} />
      </instancedMesh>
      <instancedMesh ref={posts} args={[POST, undefined, 32]} frustumCulled={false}>
        <meshLambertMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={bulbs} args={[BULB, undefined, 32]} frustumCulled={false}>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={pools} args={[POOL, undefined, 32]} frustumCulled={false} renderOrder={1}>
        <meshBasicMaterial transparent opacity={0.07} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </instancedMesh>
    </group>
  );
}
