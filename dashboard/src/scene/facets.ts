import * as THREE from "three";

export type V3 = [number, number, number];

export interface Part {
  faces: V3[][];
  tone?: number;
}

interface FrustumOptions {
  depth?: number;
  twist?: number;
  top?: boolean;
  bottom?: boolean;
  x?: number;
  z?: number;
}

function ring(sides: number, y: number, r: number, depth: number, twist: number, x: number, z: number): V3[] {
  return Array.from({ length: sides }, (_, i) => {
    const a = twist + (i / sides) * Math.PI * 2;
    return [x + Math.cos(a) * r, y, z + Math.sin(a) * r * depth] as V3;
  });
}

export function frustum(sides: number, y0: number, r0: number, y1: number, r1: number, options: FrustumOptions = {}): V3[][] {
  const { depth = 1, twist = 0, top = true, bottom = false, x = 0, z = 0 } = options;
  const lower = ring(sides, y0, r0, depth, twist, x, z);
  const upper = ring(sides, y1, r1, depth, twist, x, z);
  const faces: V3[][] = [];
  for (let i = 0; i < sides; i += 1) {
    const j = (i + 1) % sides;
    faces.push([lower[i], upper[i], upper[j], lower[j]]);
  }
  if (top) faces.push([...upper].reverse());
  if (bottom) faces.push([...lower]);
  return faces;
}

export function pyramid(sides: number, y0: number, r: number, apexY: number, options: Omit<FrustumOptions, "top"> = {}): V3[][] {
  const { depth = 1, twist = 0, bottom = false, x = 0, z = 0 } = options;
  const base = ring(sides, y0, r, depth, twist, x, z);
  const apex: V3 = [x, apexY, z];
  const faces: V3[][] = [];
  for (let i = 0; i < sides; i += 1) faces.push([base[i], apex, base[(i + 1) % sides]]);
  if (bottom) faces.push([...base]);
  return faces;
}

export function bipyramid(sides: number, cy: number, r: number, up: number, down: number, options: { depth?: number; twist?: number; x?: number; z?: number } = {}): V3[][] {
  const { depth = 1, twist = 0, x = 0, z = 0 } = options;
  const waist = ring(sides, cy, r, depth, twist, x, z);
  const north: V3 = [x, cy + up, z];
  const south: V3 = [x, cy - down, z];
  const faces: V3[][] = [];
  for (let i = 0; i < sides; i += 1) {
    const j = (i + 1) % sides;
    faces.push([waist[i], north, waist[j]]);
    faces.push([waist[j], south, waist[i]]);
  }
  return faces;
}

export function cuboid(cx: number, y0: number, cz: number, w: number, h: number, d: number, bottom = false): V3[][] {
  const x0 = cx - w / 2;
  const x1 = cx + w / 2;
  const z0 = cz - d / 2;
  const z1 = cz + d / 2;
  const y1 = y0 + h;
  const faces: V3[][] = [
    [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
    [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
    [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
    [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
  ];
  if (bottom) faces.push([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]);
  return faces;
}

export function facetedGeometry(parts: Part[]): THREE.BufferGeometry {
  const positions: number[] = [];
  const normals: number[] = [];
  const bary: number[] = [];
  const tones: number[] = [];
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const n = new THREE.Vector3();
  for (const part of parts) {
    const tone = part.tone ?? 0;
    for (const face of part.faces) {
      for (let i = 1; i < face.length - 1; i += 1) {
        const tri = [face[0], face[i], face[i + 1]];
        a.fromArray(tri[0]);
        b.fromArray(tri[1]);
        c.fromArray(tri[2]);
        n.subVectors(c, b).cross(a.clone().sub(b)).normalize();
        const hideFirst = i > 1 ? 1 : 0;
        const hideLast = i + 1 < face.length - 1 ? 1 : 0;
        const corners: V3[] = [
          [1, hideLast, hideFirst],
          [0, 1 + hideLast, hideFirst],
          [0, hideLast, 1 + hideFirst],
        ];
        tri.forEach((p, k) => {
          positions.push(p[0], p[1], p[2]);
          normals.push(n.x, n.y, n.z);
          bary.push(corners[k][0], corners[k][1], corners[k][2]);
          tones.push(tone);
        });
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute("barycentric", new THREE.Float32BufferAttribute(bary, 3));
  geometry.setAttribute("tone", new THREE.Float32BufferAttribute(tones, 1));
  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();
  return geometry;
}

export const FIGURE_HEIGHT = 1.3;

export function figureGeometry(): THREE.BufferGeometry {
  return facetedGeometry([
    { faces: frustum(4, 0, 0.15, 0.4, 0.2, { depth: 0.74, twist: Math.PI / 4 }), tone: 0 },
    { faces: frustum(6, 0.45, 0.19, 0.9, 0.31, { depth: 0.6 }), tone: 0 },
    { faces: bipyramid(4, 1.1, 0.155, 0.2, 0.15, { depth: 0.9, twist: Math.PI / 4 }), tone: 1 },
  ]);
}

export function treeGeometry(): THREE.BufferGeometry {
  return facetedGeometry([
    { faces: frustum(4, 0, 0.07, 0.42, 0.06, { twist: Math.PI / 4 }), tone: 0 },
    { faces: frustum(6, 0.38, 0.5, 1.0, 0.36), tone: 0 },
    { faces: pyramid(6, 1.0, 0.36, 1.9), tone: 0 },
  ]);
}

export function houseGeometry(): THREE.BufferGeometry {
  return facetedGeometry([
    { faces: cuboid(0, 0, 0, 1.7, 0.95, 1.4), tone: 0 },
    { faces: pyramid(4, 0.95, 1.15, 1.7, { twist: Math.PI / 4, depth: 0.84 }), tone: 1 },
  ]);
}

export function postGeometry(): THREE.BufferGeometry {
  return facetedGeometry([{ faces: frustum(4, 0, 0.06, 1.7, 0.045, { twist: Math.PI / 4 }), tone: 0 }]);
}

export function lighthouseGeometry(): THREE.BufferGeometry {
  return facetedGeometry([
    { faces: frustum(8, 0, 1.9, 0.6, 1.7, { twist: Math.PI / 8 }), tone: 0 },
    { faces: frustum(8, 0.6, 1.05, 2.8, 0.93, { twist: Math.PI / 8 }), tone: 0 },
    { faces: frustum(8, 2.8, 0.93, 4.8, 0.8, { twist: Math.PI / 8 }), tone: 1 },
    { faces: frustum(8, 4.8, 0.8, 6.8, 0.68, { twist: Math.PI / 8 }), tone: 0 },
    { faces: frustum(8, 6.8, 1.0, 6.95, 1.0, { twist: Math.PI / 8, bottom: true }), tone: 1 },
    { faces: pyramid(8, 7.85, 0.75, 8.6, { twist: Math.PI / 8, bottom: true }), tone: 1 },
  ]);
}
