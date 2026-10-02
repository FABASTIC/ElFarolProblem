import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { createEdgeMaterial } from "./edgeMaterial";
import { houseGeometry, postGeometry, treeGeometry } from "./facets";

interface GroundProps {
  gridSize: number;
  barMin: number;
  barMax: number;
}

export const BORDER = 8;
export const LIGHTHOUSE_OFFSET = 3.4;

const TILE = new THREE.PlaneGeometry(0.94, 0.94).rotateX(-Math.PI / 2);
const TREE = treeGeometry();
const HOUSE = houseGeometry();
const POST = postGeometry();
const WINDOW = new THREE.PlaneGeometry(0.28, 0.26);
const BULB = new THREE.OctahedronGeometry(0.12).translate(0, 1.8, 0);
const POOL = new THREE.CircleGeometry(1.6, 6).rotateX(-Math.PI / 2);

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

const GRASS = ["#0a0a0a", "#0b0b0b", "#090909", "#0c0c0c"];
const ROAD = ["#101010", "#111111", "#0f0f0f"];
const PLAZA = ["#151515", "#161616", "#141414"];
const TREE_EDGES = ["#2c2c2c", "#333333", "#282828", "#393939"];
const HOUSE_EDGES = ["#3a3a3a", "#444444", "#363636", "#4a4a4a"];
const LIT = ["#ffcf7a", "#ffd994", "#f6b85c"];
const POST_EDGES = ["#575757"];
const BULBS = ["#ffe2a8"];

export default function Ground({ gridSize, barMin, barMax }: GroundProps) {
  const half = gridSize / 2;
  const center = (barMin + barMax) / 2;
  const roadA = Math.floor(center) - 1;
  const roadB = Math.floor(center);
  const tiles = useRef<THREE.InstancedMesh>(null);
  const trees = useRef<THREE.InstancedMesh>(null);
  const houses = useRef<THREE.InstancedMesh>(null);
  const windows = useRef<THREE.InstancedMesh>(null);
  const posts = useRef<THREE.InstancedMesh>(null);
  const bulbs = useRef<THREE.InstancedMesh>(null);
  const pools = useRef<THREE.InstancedMesh>(null);
  const edges = useMemo(() => createEdgeMaterial({ face: "#0c0c0c", headInk: 0.3, width: 1.0, rim: 0.05, ink: "#8a8780" }), []);

  useEffect(() => () => edges.dispose(), [edges]);

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
  useInstances(trees, layout.trees, TREE_EDGES);
  useInstances(houses, layout.homes, HOUSE_EDGES);
  useInstances(windows, layout.panes, LIT, 0.48);
  useInstances(posts, layout.lamps, POST_EDGES);
  useInstances(bulbs, layout.lamps, BULBS);
  useInstances(pools, layout.lamps, BULBS, 0.012);

  const outer = gridSize + BORDER * 2 + 6;
  const tileCount = (gridSize + BORDER * 2) ** 2;

  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.03, 0]}>
        <planeGeometry args={[outer, outer]} />
        <meshBasicMaterial color="#070707" toneMapped={false} />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.02, 0]}>
        <planeGeometry args={[gridSize, gridSize]} />
        <meshBasicMaterial color="#1b1b1b" toneMapped={false} />
      </mesh>
      <instancedMesh ref={tiles} args={[TILE, undefined, tileCount]} frustumCulled={false}>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={trees} args={[TREE, edges, 2048]} frustumCulled={false} />
      <instancedMesh ref={houses} args={[HOUSE, edges, 128]} frustumCulled={false} />
      <instancedMesh ref={windows} args={[WINDOW, undefined, 256]} frustumCulled={false}>
        <meshBasicMaterial toneMapped={false} side={THREE.DoubleSide} />
      </instancedMesh>
      <instancedMesh ref={posts} args={[POST, edges, 32]} frustumCulled={false} />
      <instancedMesh ref={bulbs} args={[BULB, undefined, 32]} frustumCulled={false}>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={pools} args={[POOL, undefined, 32]} frustumCulled={false} renderOrder={1}>
        <meshBasicMaterial transparent opacity={0.07} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </instancedMesh>
    </group>
  );
}
