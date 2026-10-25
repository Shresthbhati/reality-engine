/**
 * How a version was built, and the ten change categories, in the words the Studio shows.
 *
 * Pure logic (src/lib/api/reconstruction-decision.ts) driven with the shapes the status API sends. The rule these
 * pin: the Studio states only what the engine reported -- an unreported field is null, never a sentence that sounds
 * like a measurement -- and every change category is shown, with a zero where nothing changed.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { CHANGE_KINDS, changeCategoryCounts, describeBuild } from "../../src/lib/api/reconstruction-decision.ts";

test("all ten change categories are listed in a fixed order, zero included", () => {
  assert.deepEqual(
    [...CHANGE_KINDS],
    ["preserved", "refined", "extended", "reduced", "split", "merge", "regrouped", "ambiguous", "removed", "new"],
  );
  const rows = changeCategoryCounts({ preserved: 12, extended: 2, new: 1 });
  assert.equal(rows.length, 10);
  assert.deepEqual(rows.map((r) => r.count), [12, 0, 2, 0, 0, 0, 0, 0, 0, 1]);
  assert.equal(rows.find((r) => r.kind === "reduced")?.label, "Partly reproduced");
});

test("a missing or null count map reads all zeros, not an empty list", () => {
  for (const input of [undefined, null, {}]) {
    const rows = changeCategoryCounts(input);
    assert.equal(rows.length, 10);
    assert.ok(rows.every((r) => r.count === 0));
  }
});

test("an incremental build says it extended the model, why, and that the frame was kept", () => {
  const b = describeBuild({
    verdict: "ACCEPT",
    dense: { state: "sparse", detail: "dense not justified by the measured evidence" },
    strategy: {
      mode: "incremental",
      reason: "new photos overlap the existing model",
      frame: "preserved",
      prior_images: 6,
      new_images: ["a", "b"],
      incremental_registered: 8,
      full_registered: 8,
      unregistered: [],
      arbitration: {
        choice: "incremental", deciding: "frame", why: "it keeps the established coordinate frame",
        incremental: { measures: { frame_changed: false, frame_shift: { rotation_deg: 0.02, scale_change: 0, translation_rel: 0, changed: false } } },
        full: { measures: { frame_changed: true, frame_shift: { rotation_deg: 35, scale_change: 0.27, translation_rel: 0.85, changed: true } } },
      },
    },
  });
  assert.equal(b.method, "Added the new photos to the existing model");
  assert.equal(b.why, "it keeps the established coordinate frame", "the arbitration's deciding fact wins");
  assert.match(b.frame ?? "", /frame kept/);
  assert.equal(b.comparison, "Extending placed 8 photos; re-solving everything placed 8");
  assert.match(b.decision ?? "", /^Accepted:/);
  assert.equal(b.dense, "Sparse only: dense not justified by the measured evidence");
});

test("a full re-solve says the frame was re-solved, with the strategy's own reason when nothing was arbitrated", () => {
  const b = describeBuild({
    verdict: "ACCEPT_WITH_UNCERTAINTY",
    dense: { state: "dense", detail: "84,104 dense points" },
    strategy: {
      mode: "full", reason: "the first model", frame: "re-solved", prior_images: 0, new_images: [], unregistered: [],
    },
  });
  assert.equal(b.method, "Re-solved all photos together");
  assert.equal(b.why, "the first model");
  assert.match(b.frame ?? "", /re-solved/);
  assert.equal(b.comparison, null, "no candidate comparison was reported");
  assert.match(b.decision ?? "", /open questions/);
  assert.equal(b.dense, "Dense surface built: 84,104 dense points");
});

test("nothing reported means nothing said", () => {
  const b = describeBuild({ strategy: null, verdict: null, dense: undefined });
  assert.deepEqual(b, { method: null, why: null, frame: null, comparison: null, decision: null, dense: null });
});

test("a single placed photo is not pluralised", () => {
  const b = describeBuild({
    verdict: null, dense: undefined,
    strategy: { mode: "incremental", reason: "", frame: "preserved", prior_images: 1, new_images: [],
      incremental_registered: 1, full_registered: 3, unregistered: [] },
  });
  assert.equal(b.comparison, "Extending placed 1 photo; re-solving everything placed 3");
  assert.equal(b.why, null, "an empty reason is not a reason");
});

test("the measured frame shift of the chosen candidate beats the strategy's mode label", () => {
  // an incremental build is labelled 'preserved', but the world's visible frame rotated 7.6 degrees: say so
  const b = describeBuild({
    verdict: "ACCEPT", dense: undefined,
    strategy: {
      mode: "incremental", reason: "", frame: "preserved", prior_images: 6, new_images: [], unregistered: [],
      arbitration: {
        choice: "incremental", deciding: "established surfaces lost", why: "decided by established surfaces lost",
        incremental: { measures: { frame_changed: true, frame_shift: { rotation_deg: 7.5687, scale_change: 0.00084, translation_rel: 0.00314, changed: true } } },
      },
    },
  });
  assert.match(b.frame ?? "", /^Coordinate frame changed \(rotated 7\.6°\)/);
  assert.doesNotMatch(b.frame ?? "", /rescaled|shifted/, "negligible scale and translation are not reported");
  assert.doesNotMatch(b.frame ?? "", /frame kept/);
});

test("with no measurement the strategy label is used, and an unmeasurable shift does not claim a change", () => {
  const base = { reason: "", prior_images: 1, new_images: [], unregistered: [] };
  assert.match(describeBuild({ verdict: null, dense: undefined, strategy: { ...base, mode: "incremental", frame: "preserved" } }).frame ?? "", /kept/);
  assert.match(describeBuild({ verdict: null, dense: undefined, strategy: { ...base, mode: "full", frame: "re-solved" } }).frame ?? "", /re-solved/);
  const unknown = describeBuild({
    verdict: null, dense: undefined,
    strategy: { ...base, mode: "incremental", frame: "preserved",
      arbitration: { choice: "incremental", deciding: "tie", why: "x", incremental: { measures: { frame_changed: null, frame_shift: null } } } },
  });
  assert.match(unknown.frame ?? "", /kept/, "no measurement -> the mode label, not an invented shift");
});
