/**
 * Capture-coverage model: SVG geometry fits the box, labels come from the API, hints never use compass words.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { coverageHeadline, mapGeometry, type CaptureCoverageDto } from "../../src/lib/api/capture-coverage-model.ts";

const dto: CaptureCoverageDto = {
  available: true,
  reason: null,
  walls: [
    { index: 1, label: "wall 1 (between corner A and B)", between: ["A", "B"], p0: [-3, -3], p1: [3, -3], status: "weak", covered_fraction: 0.4, multi_view_fraction: 0.1, hint: "Take 2 more photos of wall 1 (between corner A and B), standing on the opposite side of the room and overlapping each view with the last." },
    { index: 2, label: "wall 2 (between corner B and C)", between: ["B", "C"], p0: [3, -3], p1: [3, 3], status: "seen", covered_fraction: 1, multi_view_fraction: 1, hint: null },
    { index: 3, label: "wall 3 (between corner C and D)", between: ["C", "D"], p0: [3, 3], p1: [-3, 3], status: "not_seen", covered_fraction: 0, multi_view_fraction: 0, hint: "Take 3 more photos of wall 3 (between corner C and D), standing on the opposite side of the room and facing it squarely." },
  ],
  corners: [{ label: "A", xy: [-3, -3] }, { label: "B", xy: [3, -3] }, { label: "C", xy: [3, 3] }],
  cameras: [{ id: "e1", xy: [-2.5, 0], dir: [1, 0] }],
  hints: ["Take 2 more photos of wall 1 (between corner A and B)", "Take 3 more photos of wall 3 (between corner C and D)"],
  summary: { seen: 1, weak: 1, not_seen: 1 },
};

test("geometry fits inside the box and flips y so +plan-y is up on screen", () => {
  const g = mapGeometry(dto, 320, 260, 28)!;
  for (const w of g.walls) for (const v of [w.x1, w.x2]) assert.ok(v >= 28 - 1e-6 && v <= 320 - 28 + 1e-6);
  for (const w of g.walls) for (const v of [w.y1, w.y2]) assert.ok(v >= 28 - 1e-6 && v <= 260 - 28 + 1e-6);
  const wall3 = g.walls.find((w) => w.index === 3)!; // plan y = +3 is the top edge
  const wall1 = g.walls.find((w) => w.index === 1)!;
  assert.ok(wall3.y1 < wall1.y1);
  const cam = g.cameras[0];
  assert.ok(cam.tx > cam.x && cam.ty === cam.y); // faces +x
});

test("unavailable coverage yields no map and surfaces the API reason", () => {
  const off: CaptureCoverageDto = { available: false, reason: "no measured room footprint in the current model", walls: [], hints: [] };
  assert.equal(mapGeometry(off), null);
  assert.equal(coverageHeadline(off), "no measured room footprint in the current model");
});

test("headline counts come from the API summary and no compass directions appear", () => {
  assert.equal(coverageHeadline(dto), "1 of 3 walls seen; 1 not seen, 1 seen weakly.");
  const all = { ...dto, summary: { seen: 3, weak: 0, not_seen: 0 } };
  assert.equal(coverageHeadline(all), "All 3 walls were seen.");
  assert.doesNotMatch(dto.hints.join(" ").toLowerCase(), /north|south|east|west/);
});
