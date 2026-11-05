/** Correction transform payload: exactly {position, rotation(unit quaternion)} derived from a WorldIR matrix. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { matrixToQuaternion, buildTransformPayload, positionOf } from "../../src/lib/inspector/pose.ts";

const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("identity matrix -> identity quaternion", () => {
  const q = matrixToQuaternion([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
  assert.deepEqual(q, { w: 1, x: 0, y: 0, z: 0 });
});

test("90 degrees about Z -> (cos45, 0, 0, sin45), unit norm", () => {
  const q = matrixToQuaternion([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
  const h = Math.SQRT1_2;
  close(q.w, h); close(q.x, 0); close(q.y, 0); close(q.z, h);
  close(Math.hypot(q.w, q.x, q.y, q.z), 1);
});

test("180 degrees about X exercises the non-trace branch", () => {
  const q = matrixToQuaternion([[1, 0, 0], [0, -1, 0], [0, 0, -1]]);
  close(Math.abs(q.x), 1); close(q.w, 0); close(Math.hypot(q.w, q.x, q.y, q.z), 1);
});

test("missing or malformed matrix -> identity", () => {
  assert.deepEqual(matrixToQuaternion(undefined), { w: 1, x: 0, y: 0, z: 0 });
  assert.deepEqual(matrixToQuaternion([[1, 2]]), { w: 1, x: 0, y: 0, z: 0 });
});

test("payload has exactly position+rotation keys and drops matrix/frames", () => {
  const t = { source_frame: "a", target_frame: "b", uncertainty: 0.1,
    matrix: [[0, -1, 0, 5], [1, 0, 0, 6], [0, 0, 1, 7], [0, 0, 0, 1]] };
  const p = buildTransformPayload({ x: 1, y: 2, z: 3 }, t);
  assert.deepEqual(Object.keys(p).sort(), ["position", "rotation"]);
  assert.deepEqual(p.position, { x: 1, y: 2, z: 3 });
  assert.deepEqual(Object.keys(p.rotation).sort(), ["w", "x", "y", "z"]);
  assert.deepEqual(buildTransformPayload({ x: 0, y: 0, z: 0 }, undefined).rotation, { w: 1, x: 0, y: 0, z: 0 });
  assert.deepEqual(positionOf(t), { x: 5, y: 6, z: 7 });
});
