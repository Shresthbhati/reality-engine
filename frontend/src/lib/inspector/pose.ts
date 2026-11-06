/**
 * Entity pose helpers for correction commits. A real WorldIR entity transform serialises as
 * {source_frame, target_frame, matrix (4x4, row-major), uncertainty}; the correction API accepts only
 * {position: {x,y,z}, rotation: {w,x,y,z}} (unit quaternion). Pure: no `@/` imports.
 */

export interface Quat { w: number; x: number; y: number; z: number }
export interface Vec3 { x: number; y: number; z: number }

const IDENTITY: Quat = { w: 1, x: 0, y: 0, z: 0 };

function isMat(m: unknown): m is number[][] {
  return (
    Array.isArray(m) && m.length >= 3 &&
    m.slice(0, 3).every((r) => Array.isArray(r) && r.length >= 3 && r.slice(0, 3).every((v) => typeof v === "number" && Number.isFinite(v)))
  );
}

/** Normalised quaternion from the upper-left 3x3 of a row-major matrix (Shepperd's method). Identity if unusable. */
export function matrixToQuaternion(matrix: unknown): Quat {
  if (!isMat(matrix)) return { ...IDENTITY };
  const [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = matrix.map((r) => r.slice(0, 3)) as number[][];
  const tr = m00 + m11 + m22;
  let w: number, x: number, y: number, z: number;
  if (tr > 0) {
    const s = 2 * Math.sqrt(tr + 1);
    w = s / 4; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    w = (m21 - m12) / s; x = s / 4; y = (m01 + m10) / s; z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    w = (m02 - m20) / s; x = (m01 + m10) / s; y = s / 4; z = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = s / 4;
  }
  const n = Math.hypot(w, x, y, z);
  if (!Number.isFinite(n) || n === 0) return { ...IDENTITY };
  return { w: w / n, x: x / n, y: y / n, z: z / n };
}

/** Current rotation of an entity transform in either shape (already-quaternion, or matrix); identity if absent. */
export function rotationOf(transform: unknown): Quat {
  const t = (transform ?? {}) as { rotation?: Partial<Quat>; matrix?: unknown };
  const r = t.rotation;
  if (r && [r.w, r.x, r.y, r.z].every((v) => typeof v === "number" && Number.isFinite(v))) {
    const n = Math.hypot(r.w!, r.x!, r.y!, r.z!);
    if (n > 0) return { w: r.w! / n, x: r.x! / n, y: r.y! / n, z: r.z! / n };
  }
  return matrixToQuaternion(t.matrix);
}

/** Current position (explicit position, else the matrix translation column), or undefined. */
export function positionOf(transform: unknown): Vec3 | undefined {
  const t = (transform ?? {}) as { position?: Partial<Vec3>; matrix?: unknown };
  const p = t.position;
  if (p && [p.x, p.y, p.z].every((v) => typeof v === "number")) return p as Vec3;
  const m = t.matrix;
  if (Array.isArray(m) && m.length >= 3 && [0, 1, 2].every((i) => Array.isArray(m[i]) && typeof m[i][3] === "number")) {
    return { x: m[0][3], y: m[1][3], z: m[2][3] };
  }
  return undefined;
}

/** The exact payload the correction API accepts for a move: {position, rotation}. */
export function buildTransformPayload(position: Vec3, currentTransform: unknown): { position: Vec3; rotation: Quat } {
  return { position: { x: position.x, y: position.y, z: position.z }, rotation: rotationOf(currentTransform) };
}
