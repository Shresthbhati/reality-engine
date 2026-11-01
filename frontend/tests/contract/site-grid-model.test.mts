/** Site composer pure logic. Run with: npm run test:contract */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  centerM, positionFromPoint, positionForCell, nudge, nudgeStepM, normalizeDeg, rotateBy, clampScale,
  footprintRect, fallbackBoxM, contentBounds, fitView, toScreen, toMetres, gridLines,
  overlappingIds, overlapPartners, parseCell, parsePositiveM, snapToStep, cellAtPoint, defaultLabel,
  type Placement,
} from "../../src/lib/sites/site-grid-model.ts";

function mk(over: Partial<Placement> = {}): Placement {
  return {
    id: "p1", site_id: "s", world_id: "w", world_name: "W", label: "W",
    cell_x: 0, cell_z: 0, offset_x_m: 0, offset_z_m: 0, rotation_deg: 0, scale: 1,
    center_m: { x: 25, z: 25 },
    footprint: { width_m: 20, depth_m: 10, source: "worldir" },
    ...over,
  };
}

test("centerM follows the API contract, incl. negative cells and offsets", () => {
  assert.deepEqual(centerM({ cell_x: 2, cell_z: 1, offset_x_m: 3, offset_z_m: -4 }, 50), { x: 128, z: 71 });
  assert.deepEqual(centerM({ cell_x: -1, cell_z: -1, offset_x_m: 0, offset_z_m: 0 }, 10), { x: -5, z: -5 });
  assert.deepEqual(centerM(positionForCell(3, 4), 10), { x: 35, z: 45 });
});

test("positionFromPoint round-trips through centerM", () => {
  for (const pt of [{ x: 128, z: 71 }, { x: -7.5, z: -33 }, { x: 0, z: 0 }, { x: 99.99, z: 100 }]) {
    const pos = positionFromPoint(pt, 50);
    const back = centerM(pos, 50);
    assert.ok(Math.abs(back.x - pt.x) < 1e-3 && Math.abs(back.z - pt.z) < 1e-3, JSON.stringify([pt, pos, back]));
  }
  assert.equal(positionFromPoint({ x: 100, z: 0 }, 50).cell_x, 2);
});

test("nudge crosses cell borders and +z moves toward the viewer", () => {
  const p = positionForCell(0, 0); // centre (25,25)
  const down = nudge(p, 0, 30, 50); // z=55 -> cell 1
  assert.equal(down.cell_z, 1);
  assert.equal(centerM(down, 50).z, 55);
  const left = nudge(p, -30, 0, 50); // x=-5 -> cell -1
  assert.equal(left.cell_x, -1);
  assert.equal(nudgeStepM(50, false), 12.5);
  assert.equal(nudgeStepM(50, true), 2.5);
});

test("rotation wraps and scale is clamped", () => {
  assert.equal(normalizeDeg(-90), 270);
  assert.equal(normalizeDeg(450), 90);
  assert.equal(rotateBy(350, 15), 5);
  assert.equal(clampScale(0), 0.1);
  assert.equal(clampScale(999), 10);
  assert.equal(clampScale(NaN), 1);
  assert.equal(clampScale(2), 2);
});

test("known footprint scales; unknown falls back to a flagged box", () => {
  assert.deepEqual(footprintRect(mk({ scale: 2 }), 50), { width_m: 40, depth_m: 20, isFallback: false });
  const unk = footprintRect(mk({ footprint: { width_m: null, depth_m: null, source: "unknown" } }), 50);
  assert.equal(unk.isFallback, true);
  assert.equal(unk.width_m, fallbackBoxM(50));
  // a missing dimension is a fallback, never a zero-size box
  assert.equal(footprintRect(mk({ footprint: { width_m: 5, depth_m: null, source: "worldir" } }), 50).isFallback, true);
});

test("contentBounds: empty site spans a readable minimum; content is cell-aligned", () => {
  assert.deepEqual(contentBounds([], 50), { minX: 0, maxX: 150, minZ: 0, maxZ: 150 });
  const b = contentBounds([mk({ center_m: { x: 125, z: 25 } })], 50);
  assert.equal(b.minX, 100);
  assert.equal(b.maxX, 250); // widened to the 3-cell minimum
  assert.ok(b.minZ <= 15 && b.maxZ >= 35);
});

test("fitView keeps bounds inside the viewport; screen <-> metres round-trips", () => {
  const v = fitView({ minX: 0, maxX: 150, minZ: 0, maxZ: 150 }, 600, 400, 20);
  const tl = toScreen({ x: 0, z: 0 }, v);
  const br = toScreen({ x: 150, z: 150 }, v);
  assert.ok(tl.x >= 0 && tl.y >= 0 && br.x <= 600 && br.y <= 400);
  const back = toMetres(toScreen({ x: 33, z: 66 }, v), v);
  assert.ok(Math.abs(back.x - 33) < 1e-9 && Math.abs(back.z - 66) < 1e-9);
  const g = gridLines(v, 600, 400, 50);
  assert.ok(g.xs.includes(0) && g.xs.includes(150) && g.zs.includes(100));
});

test("overlap helpers use server pairs only", () => {
  const pairs: [string, string][] = [["a", "b"], ["c", "a"]];
  assert.deepEqual([...overlappingIds(pairs)].sort(), ["a", "b", "c"]);
  assert.deepEqual(overlapPartners("a", pairs).sort(), ["b", "c"]);
  assert.deepEqual(overlapPartners("z", pairs), []);
});

test("input parsing and misc helpers", () => {
  assert.equal(parseCell(" 3 "), 3);
  assert.equal(parseCell("-2"), -2);
  assert.equal(parseCell("1.5"), null);
  assert.equal(parseCell(""), null);
  assert.equal(parsePositiveM("0"), null);
  assert.equal(parsePositiveM("25"), 25);
  assert.equal(parsePositiveM(""), null);
  assert.equal(snapToStep(13, 5), 15);
  assert.deepEqual(cellAtPoint({ x: -1, z: 120 }, 50), { cellX: -1, cellZ: 2 });
  assert.equal(defaultLabel("  "), "Untitled world");
});
