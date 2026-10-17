/**
 * Auto-frame decision: frame once per world when there is geometry but no point cloud.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { shouldAutoFrame } from "../../src/lib/viewport/autoframe.ts";

test("procedural world with meshes frames on first build", () => {
  assert.equal(shouldAutoFrame(false, 12, null, "w1"), true);
});

test("a later rebuild of the same world does not re-frame", () => {
  assert.equal(shouldAutoFrame(false, 12, "w1", "w1"), false);
});

test("a different world frames again", () => {
  assert.equal(shouldAutoFrame(false, 12, "w1", "w2"), true);
});

test("a world with a point cloud keeps its cloud-based framing", () => {
  assert.equal(shouldAutoFrame(true, 12, null, "w1"), false);
});

test("no meshes means nothing to frame", () => {
  assert.equal(shouldAutoFrame(false, 0, null, "w1"), false);
});
