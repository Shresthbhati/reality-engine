/**
 * District builder pure logic: validation, POST body, live count note, edit patch.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  defaultDistrictDraft,
  districtCounts,
  districtEditFrom,
  districtNote,
  districtPatch,
  districtSummary,
  explicitFamilies,
  toDistrictBody,
  validateDistrict,
  validateDistrictEdit,
} from "../../src/lib/procedural/district-model.ts";
import { summarize } from "../../src/lib/procedural/generated-model.ts";

test("default draft is valid and builds a body with seeded families omitted", () => {
  const d = defaultDistrictDraft();
  assert.deepEqual(validateDistrict(d), []);
  const b = toDistrictBody({ ...d, name: " Town " });
  assert.equal(b.name, "Town");
  assert.equal(b.blocks_y, 3);
  assert.equal(b.families, undefined);
  assert.equal(b.cross_every, undefined);
  assert.deepEqual(b.detail, { max_full: 4 });
});

test("validation mirrors the grammar limits", () => {
  const d = defaultDistrictDraft();
  assert.ok(validateDistrict({ ...d, name: " " }).length);
  assert.ok(validateDistrict({ ...d, lot: 7 }).length);
  assert.ok(validateDistrict({ ...d, streetWidth: 61 }).length);
  assert.ok(validateDistrict({ ...d, blocksX: 40, blocksY: 40 })[0].includes("1000"));
  assert.ok(validateDistrict({ ...d, families: 13 }).length);
  assert.ok(validateDistrict({ ...d, maxFull: -1 }).length);
  assert.ok(validateDistrict({ ...d, seed: 1.5 }).length);
  assert.deepEqual(validateDistrict({ ...d, maxFull: null }), []);
});

test("spine forces two rows and sends cross_every; custom family count sends explicit families", () => {
  const b = toDistrictBody({ ...defaultDistrictDraft(), layout: "spine", blocksX: 8, blocksY: 9, families: 5 });
  assert.equal(b.blocks_y, 2);
  assert.equal(b.cross_every, 4);
  assert.equal(b.families?.length, 5);
  assert.deepEqual(explicitFamilies(5).map((f) => f.storeys), [1, 3, 5, 6, 8]);
  assert.equal(explicitFamilies(1)[0].storeys, 3);
});

test("counts: grid and spine match DistrictSpec.plan", () => {
  assert.deepEqual(districtCounts({ layout: "grid", blocksX: 3, blocksY: 2, crossEvery: 4 }), { parcels: 6, streets: 4 + 3 });
  // spine, 8 parcels per row, cross street every 4: groups = 2 -> 3 cross streets + main
  assert.deepEqual(districtCounts({ layout: "spine", blocksX: 8, blocksY: 2, crossEvery: 4 }), { parcels: 16, streets: 4 });
});

test("note states only derivable counts and is empty for invalid input", () => {
  const d = { ...defaultDistrictDraft(), blocksX: 2, blocksY: 2, maxFull: 1 };
  assert.match(districtNote(d), /^4 parcels .* and 6 named streets\. Up to 1 building fully detailed.*3 are one box each/);
  assert.equal(districtNote({ ...d, maxFull: null }).includes("Up to 4 buildings"), true);
  assert.equal(districtNote({ ...d, lot: 1 }), "");
});

test("edit form round-trips the stored detail and keeps focus; empty means null", () => {
  const f = districtEditFrom({ seed: 3, detail: { max_full: 2, full_within: null, focus: [10, 20] } });
  assert.deepEqual(districtPatch(f), { seed: 3, detail: { max_full: 2, full_within: null, focus: [10, 20] } });
  const g = districtEditFrom({ seed: 0, detail: { max_full: null, full_within: 50, focus: null } });
  assert.deepEqual(districtPatch(g), { seed: 0, detail: { max_full: null, full_within: 50, focus: null } });
  assert.ok(validateDistrictEdit({ ...f, maxFull: 1.5 }).length);
  assert.ok(validateDistrictEdit({ ...f, fullWithin: -2 }).length);
  assert.deepEqual(validateDistrictEdit(g), []);
});

test("list summary for a district item", () => {
  const p = { layout: "grid", blocks_x: 3, blocks_y: 3, lot_w: 24, lot_d: 24 };
  assert.equal(districtSummary(p), "grid, 3 x 3 parcels, lot 24 x 24 m");
  assert.equal(
    summarize({ entity_id: "t-district", kind: "district", name: "t", params: p, origin: [0, 0, 0], attach: null, depends_on: [] }),
    "grid, 3 x 3 parcels, lot 24 x 24 m",
  );
});
