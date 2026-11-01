/**
 * Results summary is derived from a WorldIR fixture (shape written by perception/architecture/room_semantics.py):
 * every number matches the IR, and a relative world never shows metres.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { buildRoomSummary } from "../../src/lib/reconstruct/summary-model.ts";
import { parseCaptureQuality } from "../../src/lib/viewport/structure-points-model.ts";

const ent = (id: string, type: string, extra: Record<string, unknown> = {}) => ({ id, type, ...extra });

function world(roomProps: Record<string, unknown> = {}, scale?: Record<string, unknown>) {
  return {
    id: "w",
    metadata: scale ? { scale } : {},
    entities: {
      "room-001": ent("room-001", "room", {
        custom_properties: {
          // L-shaped footprint: 6 x 5 bounding box, 24 area, ceiling 2.5
          footprint_polygon_m: [[0, 0], [6, 0], [6, 3], [3, 3], [3, 5], [0, 5]],
          floor_area_m2: 24,
          ceiling_height_min_m: 2.5,
          ...roomProps,
        },
      }),
      "room-002": ent("room-002", "room", { custom_properties: { superseded_by: "room-001", footprint_polygon_m: [[0, 0], [1, 0], [1, 1]] } }),
      "wall-1": ent("wall-1", "wall", { provenance: "OBSERVED" }),
      "wall-2": ent("wall-2", "wall", { provenance: "OBSERVED" }),
      "wall-3": ent("wall-3", "wall", { provenance: "INFERRED" }),
      "wall-4": ent("wall-4", "wall", { provenance: "UNOBSERVED" }),
      "door-1": ent("door-1", "door"),
      "window-1": ent("window-1", "window"),
      "window-2": ent("window-2", "window"),
      "object-001": ent("object-001", "unknown", { semantic_labels: ["furniture_candidate"] }),
      "object-002": ent("object-002", "unknown", { semantic_labels: ["furniture_candidate"] }),
      "unknown-1": ent("unknown-1", "unknown", { semantic_labels: ["noise"] }),
    },
  } as never;
}

test("counts and dimensions match the WorldIR; superseded room fragments are ignored", () => {
  const s = buildRoomSummary(world(), { scale: { state: "metric", scale_basis: "known_length", dimension_state: "MEASURED" } })!;
  assert.deepEqual(s.counts, { walls: 4, doors: 1, windows: 2, furniture: 2 });
  assert.equal(s.roomId, "room-001");
  const by = Object.fromEntries(s.dimensions.map((d) => [d.key, d]));
  assert.equal(by.length.value, 6);
  assert.equal(by.width.value, 5);
  assert.equal(by.height.value, 2.5);
  assert.equal(by.area.value, 24);
  assert.equal(by.length.unit, "m");
  assert.equal(by.area.unit, "m²");
  assert.equal(by.length.state, "MEASURED");
  assert.equal(s.unitNote, null);
});

test("a relative world never shows metres and carries no measurement-state badge", () => {
  const s = buildRoomSummary(world(), { scale: { state: "relative" } })!;
  assert.equal(s.metric, false);
  assert.ok(s.dimensions.every((d) => d.unit.startsWith("model units")), "no metre unit on a relative world");
  assert.ok(s.dimensions.every((d) => d.state === null));
  assert.match(s.unitNote ?? "", /not metres/);
  // unscaled values are still the IR's own numbers
  assert.equal(s.dimensions.find((d) => d.key === "length")!.value, 6);
});

test("missing scale is treated as not-metres, never as metric", () => {
  const s = buildRoomSummary(world(), {})!;
  assert.equal(s.metric, false);
  assert.equal(s.scaleKind, "unknown");
  assert.match(s.unitNote ?? "", /not in metres/);
});

test("an accepted prior is ESTIMATED; the per-entity dimension-state shape wins when present", () => {
  const est = buildRoomSummary(world(), { scale: { state: "metric", scale_basis: "prior_accepted_by_user", dimension_state: "ESTIMATED" } })!;
  assert.ok(est.dimensions.every((d) => d.state === "ESTIMATED"));
  const shaped = buildRoomSummary(
    world({ dimensions: { length: { value: 6.02, unit: "m", state: "MEASURED" }, height: { value: 2.48, unit: "m", state: "INFERRED" } } }),
    { scale: { state: "metric", dimension_state: "ESTIMATED" } },
  )!;
  const by = Object.fromEntries(shaped.dimensions.map((d) => [d.key, d]));
  assert.equal(by.length.value, 6.02);
  assert.equal(by.length.state, "MEASURED");
  assert.equal(by.height.value, 2.48);
  assert.equal(by.height.state, "INFERRED");
  assert.equal(by.area.state, "ESTIMATED");
});

test("unobserved walls produce a coverage warning and a capture hint; an API hint overrides the derived one", () => {
  const s = buildRoomSummary(world(), { scale: { state: "relative" }, imagesUsed: 8, imagesRegistered: 6 })!;
  assert.equal(s.unobservedWalls, 1);
  assert.deepEqual(s.warnings, ["1 wall was not seen in your photos.", "2 of 8 photos could not be placed in the model."]);
  assert.match(s.captureHint ?? "", /unseen or weakly seen wall/);
  const api = buildRoomSummary(world(), { coverage: { unobserved_walls: 2, hint: "Capture the north wall." } })!;
  assert.equal(api.unobservedWalls, 2);
  assert.equal(api.captureHint, "Capture the north wall.");
  const weak = buildRoomSummary(world(), { coverage: { unobserved_walls: 0, weak_walls: 2 } })!;
  assert.deepEqual(weak.warnings, ["2 walls were seen only weakly (from one angle or at a grazing view)."]);
  assert.ok(weak.captureHint);
});

const cq = (state: string, extra: Record<string, unknown> = {}) =>
  parseCaptureQuality({ state, message: `msg-${state}`, advice: ["first advice", "second advice"], reasons: ["why"], ...extra });

test("an insufficient capture verdict replaces the generic wording and withholds counts from unplaced boxes", () => {
  const s = buildRoomSummary(world(), {
    scale: { state: "relative" },
    imagesUsed: 10,
    imagesRegistered: 0,
    coverage: { unobserved_walls: 3, weak_walls: 1, hint: "Capture the north wall." },
    captureQuality: cq("insufficient"),
  })!;
  assert.equal(s.countsReliable, false);
  assert.equal(s.captureQuality?.message, "msg-insufficient");
  assert.equal(s.captureHint, "first advice", "first advice item, not the generic or wall-derived hint");
  assert.equal(s.unobservedWalls, 0);
  assert.deepEqual(s.warnings, [], "no 'N of M photos could not be placed' and no wall claims from unplaced boxes");
});

test("a partial verdict supplies the capture hint and drops the generic unplaced line; walls still count", () => {
  const s = buildRoomSummary(world(), { scale: { state: "relative" }, imagesUsed: 8, imagesRegistered: 6, captureQuality: cq("partial") })!;
  assert.equal(s.countsReliable, true);
  assert.ok(!s.warnings.some((w) => /could not be placed/.test(w)));
  assert.equal(s.captureQuality?.state, "partial");
  // a wall-specific hint still wins over a generic advice item when walls are unobserved...
  assert.match(s.captureHint ?? "", /unseen or weakly seen wall/);
  // ...and the advice is used when nothing wall-specific applies
  const noWalls = buildRoomSummary({ id: "w", entities: { "room-001": ent("room-001", "room", { custom_properties: {} }) } } as never, {
    imagesUsed: 8,
    imagesRegistered: 6,
    captureQuality: cq("partial"),
  })!;
  assert.equal(noWalls.captureHint, "first advice");
});

test("without a capture-quality verdict the old unplaced wording is kept", () => {
  const s = buildRoomSummary(world(), { scale: { state: "relative" }, imagesUsed: 8, imagesRegistered: 6, captureQuality: null })!;
  assert.equal(s.countsReliable, true);
  assert.equal(s.captureQuality, null);
  assert.ok(s.warnings.includes("2 of 8 photos could not be placed in the model."));
  const plain = buildRoomSummary({ id: "w", entities: { "room-001": ent("room-001", "room", { custom_properties: {} }) } } as never, { imagesUsed: 8, imagesRegistered: 6 })!;
  assert.match(plain.captureHint ?? "", /Retake the unplaced views/);
});

test("no room, no footprint: values are null (not available), never defaulted", () => {
  const none = buildRoomSummary({ id: "w", entities: { "wall-1": ent("wall-1", "wall") } } as never, { scale: { state: "metric" } })!;
  assert.deepEqual(none.dimensions, []);
  assert.equal(none.roomId, null);
  const bare = buildRoomSummary({ id: "w", entities: { "room-001": ent("room-001", "room", { custom_properties: {} }) } } as never, { scale: { state: "metric" } })!;
  assert.ok(bare.dimensions.every((d) => d.value === null && d.state === null));
  assert.equal(buildRoomSummary(null), null);
});
