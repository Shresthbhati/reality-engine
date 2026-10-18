/**
 * Map / plot view pure logic: footprint area/extent, georeference decision, unit rule.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  entityFootprints,
  formatAreaValue,
  isGeoreferenced,
  planExtent,
  resolveUnitContext,
  unionArea,
} from "../../src/lib/map/plot-model.ts";

const box = (id: string, a: [number, number], b: [number, number]) => ({
  id,
  geom: { id: `g-${id}`, type: "box", bounds_min: { x: a[0], y: 0, z: a[1] }, bounds_max: { x: b[0], y: 3, z: b[1] } },
});

function world(boxes: ReturnType<typeof box>[], metadata?: Record<string, unknown>) {
  return {
    id: "w",
    entities: Object.fromEntries(boxes.map((b) => [b.id, { id: b.id, type: "wall", geometry_ids: [`g-${b.id}`] }])),
    geometries: Object.fromEntries(boxes.map((b) => [`g-${b.id}`, b.geom])),
    metadata,
  } as never;
}

test("footprints project geometry bounds onto x/z and ignore y", () => {
  const fps = entityFootprints(world([box("a", [0, 0], [4, 2])]));
  assert.equal(fps.length, 1);
  assert.equal(fps[0].area, 8);
});

test("union area counts overlap once (2x2 + 2x2 overlapping by 1x2 = 6)", () => {
  const fps = entityFootprints(world([box("a", [0, 0], [2, 2]), box("b", [1, 0], [3, 2])]));
  assert.equal(unionArea(fps), 6);
  assert.equal(unionArea(fps) < fps.reduce((s, f) => s + f.area, 0), true);
});

test("extent spans footprints and cameras; null when empty", () => {
  const fps = entityFootprints(world([box("a", [0, 0], [4, 2])]));
  const ext = planExtent(fps, [{ position_m: [10, 1, -3], rotation_wxyz: [1, 0, 0, 0] }]);
  assert.deepEqual([ext?.width, ext?.depth], [10, 5]);
  assert.equal(planExtent([]), null);
});

test("entities without geometry are skipped, not invented", () => {
  assert.deepEqual(entityFootprints({ id: "w", entities: { e: { id: "e", type: "wall" } } } as never), []);
});

test("georeference decision needs a real in-range origin", () => {
  assert.equal(isGeoreferenced(28.6, 77.2), true);
  assert.equal(isGeoreferenced(null, 77.2), false);
  assert.equal(isGeoreferenced(0, 0), false);
  assert.equal(isGeoreferenced(91, 10), false);
  assert.equal(isGeoreferenced(NaN, 10), false);
});

test("unit rule: metres only with a positive scale that is not unscaled/nominal", () => {
  assert.equal(resolveUnitContext(world([], { scale: { state: "unscaled", meters_per_unit: 2 } })).unit, "model units");
  assert.equal(resolveUnitContext(world([], { scale: { state: "nominal" } })).unit, "model units");
  assert.equal(resolveUnitContext(world([])).unit, "model units");
  const m = resolveUnitContext(world([], { scale: { state: "scaled", meters_per_unit: 0.5 } }));
  assert.equal(m.unit, "m");
  assert.equal(formatAreaValue(8, m), "2.00 m²");
  assert.equal(formatAreaValue(8, resolveUnitContext(world([]))), "8.00 model units²");
});
