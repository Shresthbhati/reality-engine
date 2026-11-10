/**
 * One job-status parser (T007): unknown/garbage statuses must never read as terminal or usable.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { isTerminal, isUsableResult, toJobStatus } from "../../src/lib/api/job-status.ts";

test("toJobStatus lower-cases and matches the known literals", () => {
  assert.equal(toJobStatus("Succeeded"), "succeeded");
  for (const s of ["queued", "running", "partial", "succeeded", "failed", "cancelled"]) {
    assert.equal(toJobStatus(s), s);
  }
});

test("toJobStatus returns unknown for anything else", () => {
  for (const raw of ["done", "SUCCESS", "", undefined, null, 1, {}]) {
    assert.equal(toJobStatus(raw), "unknown");
  }
});

test("isTerminal is false for queued, running and unknown", () => {
  for (const s of ["partial", "succeeded", "failed", "cancelled"] as const) assert.equal(isTerminal(s), true);
  for (const s of ["queued", "running", "unknown"] as const) assert.equal(isTerminal(s), false);
});

test("isUsableResult is true only for succeeded and partial", () => {
  assert.equal(isUsableResult("succeeded"), true);
  assert.equal(isUsableResult("partial"), true);
  for (const s of ["queued", "running", "failed", "cancelled", "unknown"] as const) {
    assert.equal(isUsableResult(s), false);
  }
});
