/** The Rooms review panel's wording and grouping: plain language, nothing applied by looking, every action says what it will do. */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  changeHeadline, changeStateLabel, confidenceBand, groupChanges, hasProposals, joinSentence, joinConfirmation, revisitConfirmation,
  roomName, lostConnectionText,
} from "../../src/lib/rooms/rooms-model.ts";

test("room prefixes read as room numbers", () => {
  assert.equal(roomName("boot"), "Room 1");
  assert.equal(roomName("boot1"), "Room 1");
  assert.equal(roomName("boot3"), "Room 3");
  assert.equal(roomName("weird"), "weird");
});

test("confidence is banded in words, with the number kept for the tooltip", () => {
  assert.deepEqual(confidenceBand(0.93), { label: "High", tone: "good" });
  assert.equal(confidenceBand(0.7).label, "Medium");
  assert.equal(confidenceBand(0.2).label, "Low");
});

test("a door join is described by what was measured, never as a fact", () => {
  const s = joinSentence({ door_a: "boot1-opening-01", door_b: "boot2-opening-01", room_a: "boot1", room_b: "boot2",
    width_diff_m: 0.04, height_diff_m: 0, confidence: 0.9, state: "proposed", note: "" });
  assert.match(s, /Room 1/);
  assert.match(s, /Room 2/);
  assert.match(s, /could be the same opening/);
  assert.match(s, /4 cm/);
});

test("confirmation texts say exactly what will change and what will not", () => {
  assert.match(joinConfirmation("boot1", "boot2"), /Room 2 will be moved/);
  assert.match(joinConfirmation("boot1", "boot2"), /Nothing else changes/);
  const r = revisitConfirmation("boot2", "boot3");
  assert.match(r, /Room 3 replaces Room 2/);
  assert.match(r, /earlier version stays/);
});

test("change states use plain words and 'not seen' is never called removed", () => {
  assert.equal(changeStateLabel("matched"), "Same as before");
  assert.equal(changeStateLabel("conflicting"), "Different: check this");
  assert.match(changeStateLabel("uncertain"), /Not seen this time/);
  assert.doesNotMatch(changeStateLabel("uncertain"), /removed|gone|deleted/i);
});

test("the headline lists only the states that occur, worst first", () => {
  assert.equal(changeHeadline({ matched: 12, new: 2, conflicting: 1 }), "1 different, 2 new, 12 same");
  assert.equal(changeHeadline({}), "Nothing to compare");
});

test("changes are grouped by state in review order", () => {
  const g = groupChanges([
    { kind: "wall", state: "matched", old: "a", new: "b", label: null },
    { kind: "opening", state: "conflicting", old: "c", new: "d", label: "door" },
    { kind: "furniture", state: "new", old: null, new: "e", label: "sofa" },
  ]);
  assert.deepEqual(g.map((x) => x.state), ["conflicting", "new", "matched"]);
});

test("proposals are empty when there is nothing to show", () => {
  assert.equal(hasProposals({ door_joins: [], revisits: [] }), false);
  assert.equal(hasProposals({ door_joins: [{} as never], revisits: [] }), true);
});

test("a lost connection explains itself", () => {
  assert.match(lostConnectionText({ old_door: "boot2-opening-01", other_door: "boot1-opening-01", reason: "no door of the new room matches the old door's size" }),
    /Room 1/);
});
