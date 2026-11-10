/**
 * Job stages client (P114): parseJobStages shapes the P105 endpoint body and never invents a stage status.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseJobStages } from "../../src/lib/api/job-stages.ts";

test("parseJobStages maps a valid body to stages", () => {
  const stages = parseJobStages({
    items: [
      { name: "sfm", version: "1", status: "succeeded", attempt: 1, metrics: {}, warnings: [], error: null, started_at: "2026-01-01T00:00:00", ended_at: "2026-01-01T00:00:02", duration_s: 2 },
      { name: "mesh", version: "1", status: "failed", attempt: 2, metrics: {}, warnings: ["thin"], error: "boom", started_at: null, ended_at: null, duration_s: null },
    ],
  });
  assert.equal(stages.length, 2);
  assert.deepEqual(stages[0], { name: "sfm", status: "succeeded", attempt: 1, durationS: 2, warnings: [], error: null });
  assert.deepEqual(stages[1], { name: "mesh", status: "failed", attempt: 2, durationS: null, warnings: ["thin"], error: "boom" });
});

test("parseJobStages throws when items is not an array", () => {
  assert.throws(() => parseJobStages({}), /job stages: unexpected response shape/);
  assert.throws(() => parseJobStages(null), /job stages: unexpected response shape/);
});

test("parseJobStages maps unknown status strings to unknown", () => {
  const [s] = parseJobStages({ items: [{ name: "x", status: "weird", attempt: 1 }] });
  assert.equal(s.status, "unknown");
});

test("parseJobStages returns [] for empty items", () => {
  assert.deepEqual(parseJobStages({ items: [] }), []);
});
