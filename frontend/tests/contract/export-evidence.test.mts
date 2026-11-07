import test from "node:test";
import assert from "node:assert/strict";
import { evidenceHeadline, orderedElements, stateLabel } from "../../src/lib/export/evidence-model.ts";

const rows = [
  { id: "w1", name: "Wall 1", type: "wall", state: "inferred", confidence: 0.35, note: null },
  { id: "f1", name: "Floor", type: "floor", state: "measured", confidence: 0.9, note: null },
  { id: "s1", name: "Sofa 1", type: "furniture", state: "estimated", confidence: null, note: "class-prior size" },
];

test("headline counts every state and never claims a measurement that is not there", () => {
  assert.equal(evidenceHeadline({ total: 3, by_state: { inferred: 1, measured: 1, estimated: 1 } }), "3 elements: 1 measured, 1 estimated, 1 inferred");
  assert.equal(evidenceHeadline({ total: 2, by_state: { inferred: 2 } }), "2 elements: 2 inferred, none measured");
  assert.equal(evidenceHeadline({ total: 0, by_state: {} }), "No elements were exported");
});

test("elements are ordered weakest evidence first so uncertainty is not buried", () => {
  assert.deepEqual(orderedElements(rows).map((r) => r.id), ["s1", "w1", "f1"]);
});

test("state labels are plain language", () => {
  assert.equal(stateLabel("measured"), "Measured");
  assert.equal(stateLabel("inferred"), "Inferred");
  assert.equal(stateLabel("estimated"), "Estimated");
  assert.equal(stateLabel("something-else"), "Unknown");
});
