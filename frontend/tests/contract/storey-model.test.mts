/**
 * Contract tests for Multi-Storey Client Model (Epic E04-3).
 * Verifies storey elevation ordering, point containment, clip bounds, and elevation queries.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  sortStoreysByElevation,
  isPointInStorey,
  getStoreyForElevation,
  calculateStoreyClipBounds,
  formatStoreyBadge,
  type StoreyLevelItem,
} from "../../src/lib/studio/storey-model.ts";

const storeys: StoreyLevelItem[] = [
  {
    storey_id: "storey-1",
    name: "First Floor",
    level_index: 1,
    floor_elevation_m: 3.2,
    ceiling_elevation_m: 6.0,
    height_m: 2.8,
    room_ids: ["r-201"],
    entity_ids: ["e-201"],
  },
  {
    storey_id: "storey-0",
    name: "Ground Floor",
    level_index: 0,
    floor_elevation_m: 0.0,
    ceiling_elevation_m: 3.0,
    height_m: 3.0,
    room_ids: ["r-101", "r-102"],
    entity_ids: ["e-101"],
  },
  {
    storey_id: "storey-basement",
    name: "Basement",
    level_index: -1,
    floor_elevation_m: -2.8,
    ceiling_elevation_m: -0.2,
    height_m: 2.6,
    room_ids: ["r-001"],
    entity_ids: ["e-001"],
  },
];

test("sortStoreysByElevation orders storeys monotonically by floor elevation", () => {
  const sorted = sortStoreysByElevation(storeys);
  assert.equal(sorted[0].storey_id, "storey-basement");
  assert.equal(sorted[1].storey_id, "storey-0");
  assert.equal(sorted[2].storey_id, "storey-1");
});

test("isPointInStorey detects vertical containment with boundary tolerance", () => {
  const ground = storeys[1];
  assert.equal(isPointInStorey([1.0, 2.0, 1.5], ground), true);
  assert.equal(isPointInStorey([1.0, 2.0, -0.02], ground), true); // within tolerance
  assert.equal(isPointInStorey([1.0, 2.0, 3.5], ground), false);  // above ceiling
});

test("getStoreyForElevation resolves matching level or nearest storey", () => {
  const s0 = getStoreyForElevation(1.2, storeys);
  assert.equal(s0?.storey_id, "storey-0");

  const s1 = getStoreyForElevation(4.5, storeys);
  assert.equal(s1?.storey_id, "storey-1");

  const sB = getStoreyForElevation(-1.5, storeys);
  assert.equal(sB?.storey_id, "storey-basement");
});

test("calculateStoreyClipBounds adds safety padding to Z boundaries", () => {
  const ground = storeys[1];
  const bounds = calculateStoreyClipBounds(ground, 0.25);
  assert.equal(bounds.minZ, -0.25);
  assert.equal(bounds.maxZ, 3.25);
});

test("formatStoreyBadge formats user-friendly badge with sign prefix", () => {
  assert.equal(formatStoreyBadge(storeys[0]), "First Floor (+3.2m)");
  assert.equal(formatStoreyBadge(storeys[1]), "Ground Floor (0.0m)");
  assert.equal(formatStoreyBadge(storeys[2]), "Basement (-2.8m)");
});
