/**
 * Which points the viewer draws, and how the capture-quality verdict is parsed. Pure logic only (no three.js).
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  DECIMATED_CAP,
  choosePointDisplay,
  decimatePoints,
  isUnplaced,
  parseCaptureQuality,
  parseStructurePoints,
} from "../../src/lib/viewport/structure-points-model.ts";

test("structure points are shown by default; the full cloud only when asked for", () => {
  assert.equal(choosePointDisplay({ structureCount: 12, fullCloud: false, totalPoints: 56715 }), "structure");
  assert.equal(choosePointDisplay({ structureCount: 12, fullCloud: true, totalPoints: 56715 }), "full");
  // nothing to show in "full" when there is no cloud: the structure points still are
  assert.equal(choosePointDisplay({ structureCount: 12, fullCloud: true, totalPoints: 0 }), "structure");
});

test("without structure points the cloud is decimated, never drawn whole by default", () => {
  assert.equal(choosePointDisplay({ structureCount: 0, fullCloud: false, totalPoints: 56715 }), "decimated");
  assert.equal(choosePointDisplay({ structureCount: 0, fullCloud: true, totalPoints: 56715 }), "full");
  assert.equal(choosePointDisplay({ structureCount: 0, fullCloud: false, totalPoints: 0 }), "none");
});

test("decimatePoints caps the cloud at ~2,000 points with an even stride and keeps whole xyz triples", () => {
  const n = 56715;
  const src = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) src.set([i, i + 0.25, i + 0.5], i * 3);
  const out = decimatePoints(src);
  assert.ok(out.length % 3 === 0);
  assert.ok(out.length / 3 <= DECIMATED_CAP, `kept ${out.length / 3}`);
  assert.ok(out.length / 3 > DECIMATED_CAP / 2, "not thinned far below the cap");
  // each kept point is an original point (x, x+0.25, x+0.5), starting at the first
  assert.deepEqual([out[0], out[1], out[2]], [0, 0.25, 0.5]);
  for (let i = 0; i < out.length; i += 3) assert.equal(out[i + 2] - out[i], 0.5);
  // small clouds are returned untouched
  const small = new Float32Array(30);
  assert.equal(decimatePoints(small), small);
});

test("parseStructurePoints keeps valid points and drops malformed ones without throwing", () => {
  const pts = parseStructurePoints([
    { id: "c1", kind: "corner", position: [0, 0, 0], entity_ids: ["wall-1", "wall-2"], label: "NE corner" },
    { id: "o1", kind: "opening_corner", position: [1, 2, 3], entity_ids: ["door-1"] },
    { kind: "weird", position: [4, 5, 6] }, // unknown kind -> dimension, id generated, no entity ids
    { id: "bad1", kind: "corner", position: [1, 2] }, // too short
    { id: "bad2", kind: "corner", position: [1, "x", 3] }, // not numeric
    { id: "bad3", kind: "corner", position: [1, NaN, 3] }, // not finite
    null,
    "nope",
  ]);
  assert.deepEqual(pts.map((p) => p.id), ["c1", "o1", "sp-2"]);
  assert.equal(pts[0].label, "NE corner");
  assert.equal(pts[1].label, undefined);
  assert.equal(pts[2].kind, "dimension");
  assert.deepEqual(pts[2].entity_ids, []);
  assert.deepEqual(pts[1].position, [1, 2, 3]);
  for (const absent of [undefined, null, {}, "x", 5]) assert.deepEqual(parseStructurePoints(absent), []);
});

test("parseCaptureQuality reads the verdict; an absent or unrecognised one is null", () => {
  const q = parseCaptureQuality({
    state: "insufficient",
    n_input: 10,
    registered: 0,
    points: 92,
    duplicates_dropped: 10,
    reasons: ["0 of 10 photos registered"],
    advice: ["Retake with overlap", "Walk around the room"],
    message: "The photos could not be placed in one 3D space.",
  })!;
  assert.equal(q.state, "insufficient");
  assert.equal(q.registered, 0);
  assert.equal(q.duplicates_dropped, 10);
  assert.deepEqual(q.advice, ["Retake with overlap", "Walk around the room"]);
  assert.equal(isUnplaced(q), true);
  assert.equal(isUnplaced(parseCaptureQuality({ state: "partial", message: "m" })), false);
  assert.equal(isUnplaced(parseCaptureQuality({ state: "sufficient" })), false);
  assert.equal(isUnplaced(null), false);
  for (const absent of [undefined, null, {}, { state: "great" }, { message: "no state" }, "x", 3]) {
    assert.equal(parseCaptureQuality(absent), null);
  }
  // missing / wrong-typed fields default, never throw
  const sparse = parseCaptureQuality({ state: "partial", advice: ["ok", 7, ""], reasons: "nope", registered: -3, points: "many" })!;
  assert.deepEqual(sparse.advice, ["ok"]);
  assert.deepEqual(sparse.reasons, []);
  assert.equal(sparse.registered, 0);
  assert.equal(sparse.points, 0);
  assert.equal(sparse.message, "");
});
