import { test } from "node:test";
import assert from "node:assert/strict";
import {
  entityKind, displayName, kindLabel, primaryDimensionRows, dimensionTooltip, roomFacts, hostedOpeningIds,
  openingHostId, materialSwatches, colourName, countWorld, describeWorldCounts, scaleSentence, unitText, fmtValue,
} from "../../src/lib/inspector/entity-facts.ts";
import { buildDimensionEvidence } from "../../src/lib/viewport/inspector-evidence-model.ts";

const dim = (value: number, state = "ESTIMATED", basis: string | null = "derived", unit = "model_units") => ({ value, state, basis, unit, uncertainty_m: null });

test("kinds and names: ids never reach the primary title", () => {
  assert.equal(entityKind({ id: "o", type: "unknown", semantic_labels: ["furniture_candidate"] }), "furniture");
  assert.equal(entityKind({ id: "x", type: "stairs" }), "other");
  assert.equal(displayName({ id: "room-plane-000", type: "room", name: "room-plane-000" }), "Room");
  assert.equal(displayName({ id: "wall-1", type: "wall", name: "Wall 1" }), "Wall 1");
  assert.equal(displayName({ id: "o1", type: "unknown", name: "furniture candidate 1", semantic_labels: ["furniture_candidate"] }), "Furniture candidate 1");
  assert.equal(kindLabel({ id: "s", type: "stairs" }), "Stairs");
});

test("wall rows: plain labels, thickness default is flagged as assumed, not measured", () => {
  const e = { id: "w", type: "wall", custom_properties: { thickness_basis: "default_unmeasured", dimensions: {
    length_m: dim(4.9334, "RECONSTRUCTED"), height_m: dim(2.61, "RECONSTRUCTED"), thickness_m: dim(0.1, "ESTIMATED", "labelled default: no pipeline stage measures wall/ceiling thickness"),
  } } };
  const rows = primaryDimensionRows(e, buildDimensionEvidence(e.custom_properties));
  assert.deepEqual(rows.map((r) => r.label), ["Length", "Height", "Thickness"]);
  assert.equal(rows[2].assumedDefault, true);
  assert.equal(rows[0].assumedDefault, false);
  assert.match(dimensionTooltip(rows[2]), /assumed default/);
  assert.match(dimensionTooltip(rows[0]), /^Reconstructed\. Basis: derived/);
  assert.match(dimensionTooltip(rows[0]), /Uncertainty: not recorded/);
  assert.match(dimensionTooltip(rows[0]), /model units, not metres/);
});

test("room rows use layout width/length, one ceiling height; missing dims are omitted", () => {
  const e = { id: "r", type: "room", custom_properties: { dimensions: {
    layout_dimension_u_m: dim(3.98), layout_dimension_v_m: dim(4.93), ceiling_height_max_m: dim(2.61), height_m: dim(2.59, "RECONSTRUCTED"), floor_area_m2: dim(19.64, "RECONSTRUCTED", "x", "model_units2"),
  } } };
  const rows = primaryDimensionRows(e, buildDimensionEvidence(e.custom_properties));
  assert.deepEqual(rows.map((r) => r.label), ["Width", "Length", "Ceiling height", "Floor area"]);
  assert.equal(rows[2].value, 2.61);
  assert.deepEqual(primaryDimensionRows({ id: "w", type: "window" }, []), []);
});

test("furniture footprint comes from the plan box and carries no state", () => {
  const e = { id: "o", type: "unknown", semantic_labels: ["furniture_candidate"], custom_properties: { plan_obb_size_m: [0.8, 1.2], dimensions: { height_m: dim(0.63) } } };
  const rows = primaryDimensionRows(e, buildDimensionEvidence(e.custom_properties));
  assert.deepEqual(rows.map((r) => r.label), ["Width", "Depth", "Height"]);
  assert.equal(rows[0].state, null);
  assert.equal(rows[0].correctable, false);
});

test("room facts: missing room type is 'Not identified'; volume only when area and height exist", () => {
  const dims = buildDimensionEvidence({ dimensions: { floor_area_m2: dim(20, "RECONSTRUCTED", "b", "model_units2"), ceiling_height_max_m: dim(2.5) } });
  const f = roomFacts({ id: "r", type: "room" }, dims, null);
  assert.deepEqual(f.map((x) => x.label), ["Room type", "Area (reconstructed)", "Volume (estimated)"]);
  assert.equal(f[0].value, "Not identified");
  assert.equal(f[2].value, "50 units³");
  assert.equal(roomFacts({ id: "r", type: "room", custom_properties: { room_type: "kitchen" } }, [], "Level 1")[1].value, "Kitchen");
  assert.equal(roomFacts({ id: "r", type: "room" }, [], null).length, 1);
});

test("wall hosts its openings; opening finds its wall", () => {
  const wall = { id: "w3", type: "wall", relationships: [], custom_properties: {} };
  const win = { id: "win", type: "window", relationships: [{ target_id: "w3", kind: "part_of" }, { target_id: "struct", kind: "part_of" }] };
  const door = { id: "d", type: "door", custom_properties: { host_wall_id: "w1" } };
  const w1 = { id: "w1", type: "wall", custom_properties: { wall_slab: { openings: [{ entity_id: "d" }, { entity_id: "ghost" }] } } };
  const ents = { w3: wall, win, d: door, w1 };
  assert.deepEqual(hostedOpeningIds(wall, ents), ["win"]);
  assert.deepEqual(hostedOpeningIds(w1, ents), ["d"]);
  assert.equal(openingHostId(win, ents), "w3");
  assert.equal(openingHostId(door, ents), "w1");
  const plane = { id: "struct-plane-001", type: "wall", custom_properties: { slab_wall_id: "w3" } };
  const win2 = { id: "win2", type: "window", relationships: [{ target_id: "struct-plane-001", kind: "part_of" }, { target_id: "w3", kind: "part_of" }] };
  assert.equal(openingHostId(win2, { ...ents, "struct-plane-001": plane, win2 }), "w3");
});

test("materials: only recorded colours, labelled as estimates", () => {
  const room = { id: "r", type: "room", custom_properties: { wall_color_rgb: [141, 118, 103], floor_color_rgb: [149, 109, 70] } };
  const s = materialSwatches(room);
  assert.deepEqual(s.map((x) => x.title), ["Surface finish (estimated)", "Floor material (estimated)"]);
  assert.deepEqual(materialSwatches({ id: "w", type: "wall", custom_properties: {} }), []);
  assert.equal(materialSwatches({ id: "w", type: "wall", custom_properties: { mean_color_rgb: [1, 2] } }).length, 0);
  assert.equal(colourName([245, 245, 240]), "Off-white");
  assert.equal(colourName([120, 120, 120]), "Grey");
});

test("world counts skip evidence planes already solved into shells; scale is in plain words", () => {
  const ents = {
    r: { id: "r", type: "room" },
    "wall-1": { id: "wall-1", type: "wall" },
    "struct-plane-001": { id: "struct-plane-001", type: "wall", custom_properties: { slab_wall_id: "wall-1" } },
    "struct-plane-009": { id: "struct-plane-009", type: "wall" },
    o: { id: "o", type: "unknown", semantic_labels: ["furniture_candidate"] },
  };
  const c = countWorld(ents);
  assert.deepEqual([c.rooms, c.walls, c.furniture, c.total], [1, 2, 1, 4]);
  assert.equal(describeWorldCounts(c), "1 room, 2 walls, 1 piece of furniture");
  assert.equal(describeWorldCounts(countWorld({})), "Nothing recognised yet");
  assert.match(scaleSentence({ kind: "relative" }), /not metres/);
  assert.match(scaleSentence({ kind: "unknown" }), /not recorded/);
  assert.equal(unitText("model_units2"), "units²");
  assert.equal(fmtValue(2.61, "model_units"), "2.61 units");
  assert.equal(fmtValue(0.1, "m"), "0.10 m");
});

test("uncertaintyText shows the stated +/- inline, in the row's own unit, and says nothing when none was recorded", async () => {
  const { uncertaintyText } = await import("../../src/lib/inspector/entity-facts.ts");
  const row = (u: number | null, unit: string | null) => ({ key: "k", label: "L", value: 3.4, unit, state: "ESTIMATED", basis: "b", uncertaintyM: u, assumedDefault: false, correctable: true }) as never;
  assert.equal(uncertaintyText(row(0.51, "m")), "±0.51 m");
  assert.equal(uncertaintyText(row(0.51, "model_units")), "±0.51 units");
  assert.equal(uncertaintyText(row(null, "m")), "");
  assert.equal(uncertaintyText(row(0, "m")), "");
});
