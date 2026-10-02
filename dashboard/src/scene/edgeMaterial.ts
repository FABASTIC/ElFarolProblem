import * as THREE from "three";

export interface EdgeMaterialOptions {
  edge?: string;
  face?: string;
  ink?: string;
  headInk?: number;
  width?: number;
  rim?: number;
}

const VERTEX = `
attribute vec3 barycentric;
attribute float tone;
uniform vec3 uEdge;
varying vec3 vBary;
varying vec3 vView;
varying vec3 vEdge;
varying float vTone;

void main() {
  vBary = barycentric;
  vTone = tone;
  vec4 local = vec4(position, 1.0);
  #ifdef USE_INSTANCING
  local = instanceMatrix * local;
  #endif
  #ifdef USE_INSTANCING_COLOR
  vEdge = instanceColor;
  #else
  vEdge = uEdge;
  #endif
  vec4 mv = modelViewMatrix * local;
  vView = mv.xyz;
  gl_Position = projectionMatrix * mv;
}
`;

const FRAGMENT = `
uniform vec3 uFace;
uniform vec3 uInk;
uniform vec3 uLight;
uniform float uHeadInk;
uniform float uWidth;
uniform float uRim;
varying vec3 vBary;
varying vec3 vView;
varying vec3 vEdge;
varying float vTone;

void main() {
  vec3 n = normalize(cross(dFdx(vView), dFdy(vView)));
  if (n.z < 0.0) n = -n;
  float lit = 0.3 + 0.7 * max(dot(n, normalize(uLight)), 0.0);
  float rim = pow(1.0 - clamp(n.z, 0.0, 1.0), 2.4);
  vec3 edgeColor = mix(vEdge, uInk, clamp(vTone * uHeadInk, 0.0, 1.0));
  vec3 face = uFace * lit + edgeColor * rim * uRim;
  vec3 spread = fwidth(vBary) * uWidth;
  vec3 ramp = smoothstep(spread * 0.35, spread, vBary);
  float edge = 1.0 - min(min(ramp.x, ramp.y), ramp.z);
  gl_FragColor = vec4(mix(face, edgeColor, edge), 1.0);
  #include <colorspace_fragment>
}
`;

export function createEdgeMaterial(options: EdgeMaterialOptions = {}): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    side: THREE.DoubleSide,
    uniforms: {
      uEdge: { value: new THREE.Color(options.edge ?? "#ece9e2") },
      uFace: { value: new THREE.Color(options.face ?? "#141414") },
      uInk: { value: new THREE.Color(options.ink ?? "#f4f1ea") },
      uLight: { value: new THREE.Vector3(-0.45, 0.75, 0.5) },
      uHeadInk: { value: options.headInk ?? 0 },
      uWidth: { value: options.width ?? 1.15 },
      uRim: { value: options.rim ?? 0.1 },
    },
  });
}
