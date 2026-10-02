import { useMemo } from "react";

interface FloorProps {
  gridSize: number;
  barMin: number;
  barMax: number;
}

function segments(gridSize: number, every: number, include: (i: number) => boolean): Float32Array {
  const half = gridSize / 2;
  const points: number[] = [];
  for (let i = 0; i <= gridSize; i += every) {
    if (!include(i)) continue;
    const p = i - half;
    points.push(-half, 0, p, half, 0, p, p, 0, -half, p, 0, half);
  }
  return new Float32Array(points);
}

function loop(x0: number, z0: number, x1: number, z1: number, y: number): Float32Array {
  return new Float32Array([x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1]);
}

function brackets(x0: number, z0: number, x1: number, z1: number, y: number, arm: number): Float32Array {
  return new Float32Array([
    x0, y, z0, x0 + arm, y, z0, x0, y, z0, x0, y, z0 + arm,
    x1, y, z0, x1 - arm, y, z0, x1, y, z0, x1, y, z0 + arm,
    x1, y, z1, x1 - arm, y, z1, x1, y, z1, x1, y, z1 - arm,
    x0, y, z1, x0 + arm, y, z1, x0, y, z1, x0, y, z1 - arm,
  ]);
}

export default function Floor({ gridSize, barMin, barMax }: FloorProps) {
  const half = gridSize / 2;
  const minor = useMemo(() => segments(gridSize, 1, (i) => i % 5 !== 0), [gridSize]);
  const major = useMemo(() => segments(gridSize, 1, (i) => i % 5 === 0), [gridSize]);
  const x0 = barMin - half;
  const x1 = barMax - half;
  const side = barMax - barMin;
  const center = (x0 + x1) / 2;
  const barLoop = useMemo(() => loop(x0, x0, x1, x1, 0.03), [x0, x1]);
  const edgeLoop = useMemo(() => loop(-half, -half, half, half, 0.02), [half]);
  const corners = useMemo(() => brackets(-half - 1, -half - 1, half + 1, half + 1, 0.02, 2.4), [half]);

  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.02, 0]}>
        <planeGeometry args={[gridSize, gridSize]} />
        <meshBasicMaterial color="#0e0e11" transparent opacity={0.94} toneMapped={false} />
      </mesh>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[minor, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#19191e" toneMapped={false} />
      </lineSegments>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[major, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#2a2a31" toneMapped={false} />
      </lineSegments>
      <lineLoop>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[edgeLoop, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#3a3a42" toneMapped={false} />
      </lineLoop>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[corners, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#87878e" toneMapped={false} />
      </lineSegments>
      <mesh rotation-x={-Math.PI / 2} position={[center, 0.01, center]}>
        <planeGeometry args={[side, side]} />
        <meshBasicMaterial color="#ffd700" transparent opacity={0.06} depthWrite={false} toneMapped={false} />
      </mesh>
      <lineLoop>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[barLoop, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#ffd700" transparent opacity={0.85} toneMapped={false} />
      </lineLoop>
    </group>
  );
}
