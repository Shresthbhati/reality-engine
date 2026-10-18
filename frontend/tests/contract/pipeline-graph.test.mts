/**
 * Pipeline graph view-model: stage -> node status mapping and the honest summary headline.
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  NOT_RECORDED,
  PIPELINE_DEFS,
  buildPipelineNodes,
  buildPipelineSummary,
  formatDuration,
  jobDurationSeconds,
} from "../../src/lib/api/pipeline-graph.ts";

const statuses = (job: Parameters<typeof buildPipelineNodes>[0]["job"]) =>
  buildPipelineNodes({ job }).map((n) => n.status);
const job = (status: string, stage: string | null, payload: object | null = null) => ({
  id: "j",
  status,
  stage,
  payload,
});

test("no job: every node is not-run and says no run was recorded", () => {
  const nodes = buildPipelineNodes({ job: null });
  assert.ok(nodes.every((n) => n.status === "not-run"));
  assert.equal(nodes[0].detail, "No run recorded");
});

test("running: earlier succeeded, current running, later queued", () => {
  assert.deepEqual(statuses(job("running", "analyzing_evidence")), [
    "succeeded", "succeeded", "running", "queued", "queued",
  ]);
  // the three reconstruct variants all land on the same node
  for (const s of ["building_rough_model", "reconstructing", "refining"]) {
    assert.equal(buildPipelineNodes({ job: job("running", s) })[3].status, "running");
  }
});

test("running with an unknown stage marks nothing running", () => {
  for (const stage of [null, "something_new"]) {
    const st = statuses(job("running", stage));
    assert.ok(!st.includes("running") && !st.includes("succeeded"));
  }
});

test("failed is pinned to the stage it stopped at; later stages did not run", () => {
  assert.deepEqual(statuses(job("failed", "reconstructing")), [
    "succeeded", "succeeded", "succeeded", "failed", "not-run",
  ]);
  assert.ok(!statuses(job("failed", "reconstructing")).includes("partial"));
});

test("failed with no recorded stage claims nothing succeeded", () => {
  const nodes = buildPipelineNodes({ job: job("failed", null) });
  assert.ok(nodes.every((n) => n.status === "not-run"));
  assert.match(nodes[0].detail ?? "", /not recorded/);
});

test("cancelled is not failed; unknown job status is never optimistic", () => {
  assert.equal(statuses(job("cancelled", "resolving_evidence"))[1], "cancelled");
  assert.equal(statuses(job("exploded", "resolving_evidence"))[1], "failed");
});

test("partial marks only the reconstruct node", () => {
  assert.deepEqual(statuses(job("partial", null)), [
    "succeeded", "succeeded", "succeeded", "partial", "succeeded",
  ]);
});

test("succeeded: all succeeded; an unadopted candidate skips the commit node", () => {
  assert.ok(statuses(job("succeeded", null, { adopted: true })).every((s) => s === "succeeded"));
  const nodes = buildPipelineNodes({ job: job("succeeded", null, { adopted: false, unchanged: true }) });
  assert.equal(nodes[4].status, "skipped");
  assert.match(nodes[4].detail ?? "", /Nothing new/);
});

test("queued job: nothing has started", () => {
  assert.ok(statuses(job("queued", null)).every((s) => s === "queued"));
});

test("facts come from the report; absent values read Not recorded, never zero", () => {
  const bare = buildPipelineNodes({ job: job("succeeded", null) })[3].facts;
  assert.ok(bare.every((f) => f.value === NOT_RECORDED));
  const full = buildPipelineNodes({
    job: job("succeeded", null),
    stages: { reconstruction: { backend: "colmap", cameras_registered: 5, cameras_input: 6, points: 100 }, depth: { status: "ran" } },
  })[3].facts;
  assert.equal(full.find((f) => f.label === "Photos placed")?.value, "5 of 6");
  assert.equal(full.find((f) => f.label === "Depth")?.value, "ran");
  assert.equal(PIPELINE_DEFS.length, 5);
});

test("duration only from two recorded timestamps", () => {
  assert.equal(jobDurationSeconds({ started_at: null, completed_at: "2026-01-01T00:00:10Z" }), null);
  assert.equal(jobDurationSeconds({ started_at: "2026-01-01T00:00:00Z", completed_at: "2026-01-01T00:01:05Z" }), 65);
  assert.equal(formatDuration(null), NOT_RECORDED);
  assert.equal(formatDuration(65), "1 min 5 s");
});

test("headline says 'World constructed' only for a truly ready state", () => {
  const head = (state: string) => buildPipelineSummary({ state }, null).headline;
  assert.equal(head("READY_TO_INSPECT"), "World constructed");
  for (const s of ["PARTIALLY_COMPLETE", "FAILED", "CANCELLED", "NEEDS_MORE_EVIDENCE", "ANALYZING", "weird"]) {
    assert.notEqual(head(s), "World constructed");
  }
  assert.equal(buildPipelineSummary({ state: "FAILED" }, null).tone, "error");
});

test("summary rows are measured or Not recorded", () => {
  const s = buildPipelineSummary(
    {
      state: "PARTIALLY_COMPLETE",
      model: { images_used: 4, images_registered: 3, scale: { state: "unknown" }, uncertainties: [1, 2], dense: { state: "sparse" } },
      versions: [{ label: "V2", is_current: true }],
      failure: null,
    },
    12,
  );
  const row = (l: string) => s.rows.find((r) => r.label === l)?.value;
  assert.equal(row("Version"), "V2");
  assert.equal(row("Photos placed"), "3 of 4");
  assert.equal(row("Entities"), "12");
  assert.equal(row("Warnings"), "2 uncertainties");
  assert.equal(buildPipelineSummary({ state: "FAILED" }, undefined).rows.find((r) => r.label === "Entities")?.value, NOT_RECORDED);
});
