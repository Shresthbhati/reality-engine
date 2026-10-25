/**
 * Generated panel / Build dialog pure logic: opening validation, stored params -> form,
 * honest result line, dependency text.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  cityFormFrom,
  corridorFormFrom,
  dependencyLines,
  duplicateNameMessage,
  parseDependents,
  resultLine,
  roomFormFrom,
  summarize,
  validateOpening,
  wallLength,
  type GeneratedItem,
  type OpeningDraft,
} from "../../src/lib/procedural/generated-model.ts";

const room = { width: 4, depth: 3 };
const door = (over: Partial<OpeningDraft> = {}): OpeningDraft => ({
  wall: "south", kind: "door", lateral_offset: 0.5, width: 0.9, bottom: 0, top: 2.1, ...over,
});

test("wall length: north/south use width, east/west use depth", () => {
  assert.equal(wallLength("north", room), 4);
  assert.equal(wallLength("south", room), 4);
  assert.equal(wallLength("east", room), 3);
  assert.equal(wallLength("west", room), 3);
});

test("a valid opening has no messages; boundary fit is allowed", () => {
  assert.deepEqual(validateOpening(door(), room), []);
  assert.deepEqual(validateOpening(door({ lateral_offset: 3.1, width: 0.9 }), room), []);
});

test("offset + width beyond the wall is rejected, using the right wall length", () => {
  const msgs = validateOpening(door({ wall: "east", lateral_offset: 2.5, width: 0.9 }), room);
  assert.equal(msgs.length, 1);
  assert.match(msgs[0], /exceeds the east wall length \(3 m\)/);
  assert.deepEqual(validateOpening(door({ wall: "north", lateral_offset: 2.5, width: 0.9 }), room), []);
});

test("top must be above bottom and bottom >= 0; width > 0; offset >= 0", () => {
  assert.match(validateOpening(door({ top: 0.5, bottom: 0.5 }), room).join(" "), /Top must be above bottom/);
  assert.match(validateOpening(door({ bottom: -0.1 }), room).join(" "), /below the floor/);
  assert.match(validateOpening(door({ width: 0 }), room).join(" "), /Width must be greater than 0/);
  assert.match(validateOpening(door({ lateral_offset: -1 }), room).join(" "), /negative/);
});

test("non-finite input yields one message, not NaN arithmetic", () => {
  assert.deepEqual(validateOpening(door({ width: NaN }), room), ["Enter numbers for every field."]);
});

const roomItem: GeneratedItem = {
  entity_id: "A-room", kind: "room", name: "A", origin: [1, 2, 0], attach: null, depends_on: [],
  params: {
    width: 4, depth: 3, height: 2.4, seed: 7,
    openings: [{ wall: "south", kind: "door", lateral_offset: 0.5, width: 0.9, bottom: 0, top: 2.1 }],
  },
};
const corridorItem: GeneratedItem = {
  entity_id: "Hall-corridor", kind: "corridor", name: "Hall", origin: [0, 0, 0],
  attach: { to: "A-room", side: "east" }, depends_on: ["A-room", "B-room"],
  params: { length: 6, width: 2, height: 2.6, axis: "x", connects: ["B-room", "A-room"] },
};

test("room params prefill the form exactly as stored", () => {
  const f = roomFormFrom(roomItem);
  assert.deepEqual([f.width, f.depth, f.height, f.seed, f.x, f.y], [4, 3, 2.4, 7, 1, 2]);
  assert.equal(f.openings.length, 1);
  assert.equal(f.openings[0].wall, "south");
});

test("corridor form drops the attach target from connects (server re-adds it)", () => {
  const f = corridorFormFrom(corridorItem);
  assert.deepEqual(f.connects, ["B-room"]);
  assert.deepEqual(f.attach, { to: "A-room", side: "east" });
  assert.equal(f.axis, "x");
});

test("city form maps stored params", () => {
  const f = cityFormFrom({
    entity_id: "C-city", kind: "city", name: "C", origin: [0, 0, 0], attach: null, depends_on: [],
    params: { blocks_x: 3, blocks_y: 2, lot_size: 20, street_width: 8, min_height: 6, max_height: 30, seed: 1 },
  });
  assert.equal(f.blocks_y, 2);
  assert.equal(f.max_height, 30);
});

test("result line states exactly the counts from the response", () => {
  assert.equal(resultLine({ changed: 3, unchanged: 12, removed: 0 }), "3 changed - 12 unchanged - 0 removed");
});

test("summaries and dependency lines are derived from stored data only", () => {
  assert.equal(summarize(roomItem), "4 x 3 x 2.4 m");
  const items = [roomItem, corridorItem, { ...roomItem, entity_id: "B-room", name: "B" }];
  assert.deepEqual(dependencyLines(corridorItem, items), ["Attached to A (east side)", "Connects B"]);
  assert.deepEqual(dependencyLines(roomItem, items), ["Used by Hall"]);
});

test("duplicate-name message and dependents parsing", () => {
  assert.equal(duplicateNameMessage("room", "A"), "A room named A already exists - edit it in Generated");
  assert.deepEqual(
    parseDependents("room 'A' has dependents: Hall, Walk; delete them first or pass ?cascade=true"),
    ["Hall", "Walk"],
  );
  assert.deepEqual(parseDependents("something else"), []);
});
