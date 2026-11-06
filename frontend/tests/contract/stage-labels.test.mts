/**
 * Processing states must match the backend (plan U.5): every stage `apps/api/jobs.py` can set on a job has an explicit
 * plain-language label in the flow model, so the UI never falls back to a humanised raw name for a stage that really exists,
 * and the running step shows that label and nothing invented (no percentage).
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { STAGE_LABEL, buildFlowSteps, stageLabel, type FlowStatusInput } from "../../src/lib/reconstruct/flow-model.ts";

const jobsPy = readFileSync(new URL("../../../apps/api/jobs.py", import.meta.url), "utf8");
const backendStages = [...new Set([...jobsPy.matchAll(/job\.stage\s*=\s*"([a-z_]+)"/g)].map((m) => m[1]))];

test("the backend's stage names were found (guards the regex against a refactor)", () => {
  assert.ok(backendStages.length >= 5, `only found: ${backendStages.join(", ")}`);
  assert.ok(backendStages.includes("committing_version"));
});

test("every stage the backend can report has an explicit label", () => {
  const missing = backendStages.filter((s) => !(s in STAGE_LABEL));
  assert.deepEqual(missing, [], `backend stages without a label: ${missing.join(", ")}`);
});

test("labels are plain words: no underscores, no raw stage names", () => {
  for (const [stage, label] of Object.entries(STAGE_LABEL)) {
    assert.ok(!label.includes("_"), `${stage}: ${label}`);
    assert.notEqual(label.toLowerCase(), stage);
  }
});

function running(stage: string | null): FlowStatusInput {
  return {
    state: "PROCESSING", state_label: "Processing", in_progress: true, has_model: false,
    evidence_summary: { count: 6 }, job: { id: "job_x", status: "running", stage },
  };
}

test("the running step names the job's real stage and never shows a percentage", () => {
  for (const stage of backendStages) {
    const step = buildFlowSteps(running(stage)).find((s) => s.id === "reconstruct")!;
    assert.equal(step.state, "active");
    assert.ok(step.reason.startsWith(stageLabel(stage)!), `${stage}: ${step.reason}`);
    assert.ok(!/\d\s*%/.test(step.reason), `${stage} reason contains a percentage: ${step.reason}`);
  }
});

test("a running job with no stage says only that it is working, not a guessed stage", () => {
  const step = buildFlowSteps(running(null)).find((s) => s.id === "reconstruct")!;
  assert.equal(step.reason, "Processing…");
});

test("an unknown future stage is shown humanised, as-is, rather than hidden or mapped to a guess", () => {
  const step = buildFlowSteps(running("some_new_stage")).find((s) => s.id === "reconstruct")!;
  assert.ok(step.reason.startsWith("Some new stage…"));
});
