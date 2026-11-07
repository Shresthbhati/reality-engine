/** Furniture from a layout world is drawn as a recognisable piece (seat, back, arms; table top and legs), not one slab. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { buildRoomShellParts } from "../../src/lib/viewport/room-shell-model.ts";

const world = JSON.parse(readFileSync(new URL("./fixtures/layout-world.json", import.meta.url), "utf8"));
const parts = (w: unknown, id: string) => buildRoomShellParts(w as never).filter((p) => p.entityId === id && p.kind === "furniture");
const bounds = (ps: Array<{ positions: number[] }>) => {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of ps) for (let i = 0; i < p.positions.length; i += 3) for (let k = 0; k < 3; k++) {
    lo[k] = Math.min(lo[k], p.positions[i + k]); hi[k] = Math.max(hi[k], p.positions[i + k]);
  }
  return { lo, hi };
};

test("a sofa is several parts (base, back, arms, cushions), not one box", () => {
  const ps = parts(world, "boot-furniture-01");
  assert.ok(ps.length >= 6, `parts: ${ps.length}`);
  assert.equal(new Set(ps.map((p) => p.color.join(","))).size >= 2, true, "cushions and frame must differ in tone");
});

test("the pieces stay inside the estimated footprint and height of the entity", () => {
  const f = world.entities["boot-furniture-01"];
  const d = f.custom_properties.dimensions;
  const { x, y, z } = f.transform.position;
  const { lo, hi } = bounds(parts(world, "boot-furniture-01"));
  const r = 0.5 * Math.hypot(d.length_m.value, d.width_m.value) + 0.01;
  assert.ok(lo[0] >= x - r && hi[0] <= x + r && lo[2] >= z - r && hi[2] <= z + r);
  assert.ok(Math.abs(lo[1] - y) < 0.02, "stands on the floor contact");
  assert.ok(hi[1] <= y + d.height_m.value + 0.01 && hi[1] >= y + 0.9 * d.height_m.value, "reaches the estimated height");
});

test("a table has a top and four legs; an unknown class stays one box", () => {
  const table = structuredClone(world);
  table.entities["boot-furniture-01"].custom_properties.category = "dining_table";
  table.entities["boot-furniture-01"].custom_properties.entity_class = "furniture.dining_table";
  assert.equal(parts(table, "boot-furniture-01").length, 5);
  const odd = structuredClone(world);
  odd.entities["boot-furniture-01"].custom_properties.category = "teapot";
  assert.equal(parts(odd, "boot-furniture-01").length, 1);
});

test("the modelled colour is flagged as not measured when no colour was measured", () => {
  for (const p of parts(world, "boot-furniture-01")) assert.equal(p.colorMeasured, false);
});
