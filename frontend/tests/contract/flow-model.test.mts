/**
 * Guided flow: every step state and reason comes from the world status; stage progress is the job's real stage.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { buildFlowSteps, ctaTarget, flowNav, stageLabel, type FlowStatusInput } from "../../src/lib/reconstruct/flow-model.ts";

const base: FlowStatusInput = { state: "READY_TO_INSPECT", in_progress: false, has_model: true, evidence_summary: { count: 8 }, model: { images_used: 8, images_registered: 7, model_state: "PARTIAL", scale: { state: "relative" } } };
const states = (s: FlowStatusInput | null) => buildFlowSteps(s).map((x) => x.state);

test("no world yet: only step 1 needs action, the rest are unavailable with a reason", () => {
  const steps = buildFlowSteps(null);
  assert.deepEqual(steps.map((s) => s.state), ["action", "unavailable", "unavailable", "unavailable", "unavailable"]);
  assert.ok(steps.every((s) => s.reason.length > 0));
  assert.match(steps[0].reason, /60%/);
  assert.equal(steps.filter((s) => s.current).length, 1);
  assert.equal(steps[0].current, true);
});

test("running: step 2 is active with the job's real stage, later steps wait", () => {
  const steps = buildFlowSteps({ state: "RECONSTRUCTING", state_label: "Reconstructing", in_progress: true, has_model: false, evidence_summary: { count: 5 }, job: { id: "j", status: "running", stage: "analyzing_evidence" } });
  assert.deepEqual(steps.map((s) => s.state), ["done", "active", "unavailable", "unavailable", "unavailable"]);
  assert.match(steps[1].reason, /^Analysing the photos/);
  assert.equal(steps[1].current, true);
  assert.equal(stageLabel("some_new_stage"), "Some new stage");
  assert.equal(stageLabel(null), null);
});

test("relative model: scale needs action with the call to action; metric clears it", () => {
  assert.deepEqual(states(base), ["done", "done", "action", "action", "action"]);
  const steps = buildFlowSteps(base);
  assert.equal(steps[2].reason, "Dimensions are not in metres. Enter one known distance.");
  assert.equal(steps[2].current, true);
  assert.match(steps[1].reason, /7 of 8 photos placed \(partial\)/);
  assert.match(steps[4].reason, /model units/);
  const metric = buildFlowSteps({ ...base, model: { ...base.model!, scale: { state: "metric", scale_basis: "known_length", dimension_state: "MEASURED" } } });
  assert.equal(metric[2].state, "done");
  assert.equal(metric[3].current, true);
  assert.doesNotMatch(metric[4].reason, /model units/);
  const est = buildFlowSteps({ ...base, model: { ...base.model!, scale: { state: "metric", scale_basis: "prior_accepted_by_user", dimension_state: "ESTIMATED" } } });
  assert.equal(est[2].state, "done");
  assert.match(est[2].reason, /ESTIMATED/);
});

test("failure and cancellation surface the real message with a retry action", () => {
  const f = buildFlowSteps({ state: "FAILED", in_progress: false, has_model: false, evidence_summary: { count: 3 }, failure: { message: "Too few overlapping photos" } });
  assert.equal(f[1].state, "action");
  assert.equal(f[1].reason, "Too few overlapping photos");
  assert.equal(f[1].cta?.kind, "retry");
  assert.equal(f[2].state, "unavailable");
  const c = buildFlowSteps({ state: "CANCELLED", in_progress: false, has_model: false, evidence_summary: { count: 3 } });
  assert.match(c[1].reason, /cancelled/);
});

test("photos but no model and no failure: step 2 asks for more photos using the API guidance", () => {
  const s = buildFlowSteps({ state: "NEEDS_MORE_EVIDENCE", in_progress: false, has_model: false, evidence_summary: { count: 2 }, guidance: [{ message: "Add photos that overlap." }] });
  assert.equal(s[1].reason, "Add photos that overlap.");
  assert.equal(s[1].state, "action");
});

test("CTA targets: never a route without a world; scale deep-links to the scale step", () => {
  assert.deepEqual(ctaTarget("open_world", null), { type: "photos" });
  assert.deepEqual(ctaTarget("set_scale", undefined), { type: "photos" });
  assert.deepEqual(ctaTarget("add_photos", "w1"), { type: "photos" });
  assert.deepEqual(ctaTarget("retry", "w1"), { type: "retry" });
  assert.deepEqual(ctaTarget("open_world", "w 1"), { type: "route", href: "/worlds/w%201" });
  assert.deepEqual(ctaTarget("set_scale", "w1"), { type: "route", href: "/worlds/w1?step=scale" });
});

test("back/next nav: none without a world, next only once a model exists, both point at the viewer", () => {
  assert.deepEqual(flowNav(null, true), { back: null, next: null });
  assert.equal(flowNav("w1", false).next, null);
  const n = flowNav("w1", true);
  assert.equal(n.back?.href, "/worlds/w1");
  assert.equal(n.next?.href, "/worlds/w1");
});
