/**
 * The "oversized entity" rule: large entities are treated as depth noise only when a point cloud exists.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { isOversizedEntity } from "../../src/lib/viewport/oversized.ts";

test("a procedural world (no cloud) never hides large entities", () => {
  assert.equal(isOversizedEntity(20, false, 1.0), false);
});

test("with a cloud, an entity far larger than the robust extent is oversized", () => {
  assert.equal(isOversizedEntity(20, true, 1), true);
});

test("with a cloud, a modest entity is not oversized", () => {
  assert.equal(isOversizedEntity(3, true, 1), false);
});

test("an unknown robust extent (0) never flags anything", () => {
  assert.equal(isOversizedEntity(20, true, 0), false);
});
