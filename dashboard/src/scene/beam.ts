import * as THREE from "three";

const VERTEX = `
varying float vAlong;
varying vec3 vNormalView;
varying vec3 vViewDir;

void main() {
  vAlong = 1.0 - uv.y;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vNormalView = normalize(normalMatrix * normal);
  vViewDir = isOrthographic ? vec3(0.0, 0.0, 1.0) : normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}
`;

const FRAGMENT = `
uniform vec3 uColor;
uniform float uIntensity;
uniform float uFalloff;
uniform float uSoftness;
varying float vAlong;
varying vec3 vNormalView;
varying vec3 vViewDir;

void main() {
  float along = clamp(vAlong, 0.0, 1.0);
  float body = pow(1.0 - along, uFalloff);
  float throat = smoothstep(0.0, 0.06, along);
  float facing = abs(dot(normalize(vNormalView), normalize(vViewDir)));
  float edge = pow(facing, uSoftness);
  float alpha = uIntensity * body * throat * edge;
  gl_FragColor = vec4(uColor, alpha);
  #include <colorspace_fragment>
}
`;

export interface BeamOptions {
  color?: string;
  intensity?: number;
  falloff?: number;
  softness?: number;
}

export function createBeamMaterial(options: BeamOptions = {}): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uColor: { value: new THREE.Color(options.color ?? "#ffcf7a") },
      uIntensity: { value: options.intensity ?? 0.1 },
      uFalloff: { value: options.falloff ?? 1.7 },
      uSoftness: { value: options.softness ?? 1.6 },
    },
  });
}

export function beamGeometry(radius: number, length: number, radial = 40): THREE.BufferGeometry {
  return new THREE.ConeGeometry(radius, length, radial, 12, true).translate(0, -length / 2, 0).rotateZ(Math.PI / 2);
}
