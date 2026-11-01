/**
 * M2 viewer: an entity with an appearance atlas reference gets per-vertex atlas coordinates; entities without one
 * (or with a malformed one) keep the flat/striped fallback. Fixture mirrors SurfaceAppearance.to_property().
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { buildRoomShellParts, partMaterial, atlasProjector, atlasUrl } from "../../src/lib/viewport/room-shell-model.ts";

const SHA = "a".repeat(64);
const texture = (extra: Record<string, unknown> = {}) => ({
  color_rgb: [200, 190, 180], color_state: "measured", views_used: ["v1"], coverage: 0.5,
  texture: {
    uri: `artifact://${SHA}`, hash: SHA, media_type: "image/png", width: 8, height: 4,
    frame: { origin: [0, 0, 0], u_axis: [1, 0, 0], v_axis: [0, 0, 1], u_range_m: [0, 4], v_range_m: [0, 2], row0: "max v (top)" },
    ...extra,
  },
});

function world(appearance?: unknown) {
  return {
    id: "w",
    entities: {
      "floor-1": { id: "floor-1", type: "floor", custom_properties: { room_id: "r", slab: { up: [0, 0, 1], polygon_xyz_m: [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]] } } },
      "wall-1": {
        id: "wall-1", type: "wall",
        custom_properties: { room_id: "r", mean_color_rgb: [180, 170, 160], ...(appearance ? { appearance } : {}) },
      },
      "room-1": {
        id: "room-1", type: "room",
        custom_properties: {
          footprint_polygon_xyz_m: [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]], floor_height_m: 0,
          footprint_basis: { up: [0, 0, 1] }, footprint_edge_heights_m: [2, 2, 2, 2], boundary_element_ids: ["wall-1", "floor-1"],
        },
      },
    },
  } as never;
}

test("projector maps world positions into atlas coordinates", () => {
  const pr = atlasProjector(texture())!;
  assert.deepEqual(pr.project([0, 0, 0]), [0, 0]);
  assert.deepEqual(pr.project([4, 9, 2]), [1, 1]);
  assert.deepEqual(pr.project([2, 0, 1]), [0.5, 0.5]);
});

test("malformed or missing atlas references are rejected (fallback)", () => {
  assert.equal(atlasProjector(undefined), null);
  assert.equal(atlasProjector({ texture: null }), null);
  assert.equal(atlasProjector({ texture: { uri: "artifact://x" } }), null);
  const bad = texture();
  (bad.texture.frame as Record<string, unknown>).u_range_m = [1, 1];
  assert.equal(atlasProjector(bad), null);
});

test("wall parts of a textured entity carry uvs (2 per vertex); other entities do not", () => {
  const parts = buildRoomShellParts(world(texture()));
  const walls = parts.filter((p) => p.entityId === "wall-1");
  assert.ok(walls.length > 0, "fixture produced no wall parts");
  for (const p of walls) {
    assert.ok(p.texture, p.id);
    assert.equal(p.texture!.uvs.length, (p.positions.length / 3) * 2);
    assert.ok(p.texture!.uvs.every(Number.isFinite));
  }
  for (const p of parts.filter((q) => q.entityId !== "wall-1")) assert.equal(p.texture, undefined, p.id);
});

test("without an appearance the part keeps the flat colour and stripes exactly as before", () => {
  const parts = buildRoomShellParts(world());
  assert.ok(parts.every((p) => p.texture === undefined));
  assert.equal(partMaterial(parts.find((p) => p.kind === "floor")!, "solid").textured, false);
});

test("geometry is identical with and without texture (modes may only change material)", () => {
  const a = buildRoomShellParts(world());
  const b = buildRoomShellParts(world(texture()));
  assert.deepEqual(a.map((p) => [p.id, p.positions, p.indices]), b.map((p) => [p.id, p.positions, p.indices]));
});

test("textured part: textured in solid/both, never in confidence; no stripes where the atlas covers", () => {
  const wall = buildRoomShellParts(world(texture())).find((p) => p.entityId === "wall-1")!;
  for (const mode of ["solid", "both", "wireframe"] as const) {
    assert.equal(partMaterial(wall, mode).textured, true, mode);
    assert.equal(partMaterial(wall, mode).hatch, false, mode);
  }
  assert.equal(partMaterial(wall, "confidence").textured, false);
});

test("atlas url resolution", () => {
  assert.equal(atlasUrl(`artifact://${SHA}`, "w 1"), `/api/worlds/w%201/artifacts/${SHA}`);
  assert.equal(atlasUrl(`artifact://${SHA}`, undefined), null);
  assert.equal(atlasUrl("artifact://nothex", "w"), null);
  assert.equal(atlasUrl("https://x/y.png", "w"), "https://x/y.png");
  assert.equal(atlasUrl("javascript:alert(1)", "w"), null);
});
