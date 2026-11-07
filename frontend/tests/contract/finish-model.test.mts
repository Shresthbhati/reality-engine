/** Surface finishes: the viewer draws a design finish in the normal view only, keeps the survey colour, and the panel's arithmetic is exact. */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  applyAllLabel, currentFinish, finishSummary, hexToRgb, offersMaterials, rgbToHex, surfaceType,
} from "../../src/lib/design/finish-model.ts";
import { buildRoomShellParts, partMaterial } from "../../src/lib/viewport/room-shell-model.ts";

test("only walls, floors and ceilings take a finish; only floors take a material", () => {
  assert.equal(surfaceType({ type: "wall" }), "wall");
  assert.equal(surfaceType({ type: "floor" }), "floor");
  assert.equal(surfaceType({ type: "furniture" }), null);
  assert.equal(offersMaterials("floor"), true);
  assert.equal(offersMaterials("wall"), false);
  assert.equal(offersMaterials(null), false);
});

test("hex and rgb round trip, and bad hex is null, never black", () => {
  assert.equal(rgbToHex([178, 190, 168]), "#b2bea8");
  assert.deepEqual(hexToRgb("#B2BEA8"), [178, 190, 168]);
  assert.deepEqual(hexToRgb("b2bea8"), [178, 190, 168]);
  assert.equal(hexToRgb("#12"), null);
  assert.equal(hexToRgb("red"), null);
  assert.equal(rgbToHex([300, -5, 12.4]), "#ff000c");
});

test("the current finish is read defensively", () => {
  assert.equal(currentFinish({ custom_properties: {} }), null);
  assert.equal(currentFinish({ custom_properties: { design_finish: null } }), null);
  assert.equal(currentFinish({ custom_properties: { design_finish: { colour_rgb: [1, 2] } } }), null);
  assert.equal(currentFinish({ custom_properties: { design_finish: { colour_rgb: [1, 2, 300] } } }), null);
  const f = currentFinish({ custom_properties: { design_finish: { kind: "paint", name: "Sage", paint: "sage", colour_rgb: [178, 190, 168] } } });
  assert.equal(finishSummary(f), "Sage (paint)");
  assert.equal(finishSummary(null), "Survey colour (no finish applied)");
  assert.equal(finishSummary({ kind: "colour", name: "x", colour_rgb: [0, 0, 255] }), "Custom colour #0000ff");
  assert.equal(applyAllLabel("wall", 4), "Apply to all 4 walls");
  assert.equal(applyAllLabel("wall", 1), null);
});

// a room with one solid wall and a floor slab, both with a measured colour
const world = (wallFinish?: unknown, floorFinish?: unknown) => ({
  id: "w", entities: {
    "room-1": { id: "room-1", type: "room", custom_properties: { wall_color_rgb: [190, 180, 170] } },
    "wall-1": {
      id: "wall-1", type: "wall", semantic_labels: ["wall_slab"], confidence: 0.9, provenance: "RECONSTRUCTED",
      custom_properties: {
        room_id: "room-1", mean_color_rgb: [200, 190, 180],
        wall_slab: { start_xyz_m: [0, 0, 0], direction: [1, 0, 0], outward: [0, -1, 0], up: [0, 0, 1], length_m: 4, height_m: 2.6, thickness_m: 0.1, openings: [] },
        ...(wallFinish ? { design_finish: wallFinish } : {}),
      },
    },
    "floor-slab": {
      id: "floor-slab", type: "floor", confidence: 0.9, provenance: "RECONSTRUCTED",
      custom_properties: {
        room_id: "room-1", mean_color_rgb: [150, 110, 70],
        slab: { polygon_xyz_m: [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]], up: [0, 0, 1], thickness_m: 0.15, direction: -1 },
        ...(floorFinish ? { design_finish: floorFinish } : {}),
      },
    },
  },
  geometries: {}, metadata: {},
}) as never;

const navy = { kind: "paint", name: "Navy", colour_rgb: [40, 52, 84] };
const oak = { kind: "material", name: "Oak boards", material: "oak", colour_rgb: [196, 154, 108] };

test("a finish repaints that surface in the normal view only; the other surface and the survey colour are unchanged", () => {
  const plain = buildRoomShellParts(world());
  const painted = buildRoomShellParts(world(navy, oak));
  const of = (parts: ReturnType<typeof buildRoomShellParts>, id: string) => parts.filter((p) => p.entityId === id);
  assert.ok(of(plain, "wall-1").length > 0 && of(plain, "floor-slab").length > 0, "the test world must produce wall and floor parts");
  assert.ok(plain.every((p) => !p.designed));
  assert.ok(of(painted, "wall-1").every((p) => p.designed && p.color.join() === "40,52,84"));
  assert.ok(of(painted, "floor-slab").every((p) => p.designed && p.color.join() === "196,154,108"));
  const onlyWall = buildRoomShellParts(world(navy));
  assert.ok(of(onlyWall, "floor-slab").every((p) => !p.designed), "a finish on the wall does not touch the floor");
  const w = of(painted, "wall-1")[0];
  assert.deepEqual(partMaterial(w, "normal" as never), { color: [40, 52, 84], hatch: false, opacity: w.opacity, textured: false });
  // evidence / confidence view: the finish plays no part (same result as the unpainted wall)
  const w0 = of(plain, "wall-1")[0];
  assert.deepEqual(partMaterial(w, "confidence"), partMaterial({ ...w0, id: w.id }, "confidence"));
});
