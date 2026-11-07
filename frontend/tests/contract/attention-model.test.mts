/**
 * Home "Needs attention" (plan U.4): derived only from real jobs, latest job per entity decides.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { attentionItems, readableFailure, STALE_AFTER_DAYS } from "../../src/lib/dashboard/attention-model.ts";
import type { JobDto } from "../../src/lib/api/types.ts";

const NOW = new Date("2026-11-07T12:00:00Z");
const iso = (daysAgo: number) => new Date(NOW.getTime() - daysAgo * 86_400_000).toISOString();

function job(over: Partial<JobDto> & Pick<JobDto, "id" | "status">): JobDto {
  return {
    type: "RECONSTRUCT_SESSION", entity_type: "session", entity_id: "s1", stage: null, attempts: 1, error: null,
    worker_id: null, created_at: iso(1), started_at: null, completed_at: iso(1), payload: null, ...over,
  };
}

test("no jobs -> nothing needs attention", () => {
  assert.deepEqual(attentionItems(null, NOW), []);
  assert.deepEqual(attentionItems([], NOW), []);
});

test("succeeded and cancelled jobs never appear", () => {
  const items = attentionItems([
    job({ id: "a", status: "succeeded", entity_id: "s1" }),
    job({ id: "b", status: "cancelled", entity_id: "s2" }),
  ], NOW);
  assert.deepEqual(items, []);
});

test("a failed job carries its own error and links to its session", () => {
  const [it] = attentionItems([job({ id: "j1", status: "failed", error: "World has 0 usable photo evidence" })], NOW);
  assert.equal(it.kind, "failed");
  assert.equal(it.title, "Reconstruction failed");
  assert.equal(it.detail, "World has 0 usable photo evidence");
  assert.equal(it.href, "/sessions/s1");
});

test("a finished job's world_id wins as the link; degradation reasons are shown (max two + count)", () => {
  const [it] = attentionItems([
    job({ id: "j1", status: "partial", payload: { world_id: "w 9", degraded: ["a", "b", "c", "d"] } }),
  ], NOW);
  assert.equal(it.kind, "partial");
  assert.equal(it.title, "Reconstruction finished with problems");
  assert.equal(it.detail, "a; b (+2 more)");
  assert.equal(it.href, "/worlds/w%209");
});

test("a later successful run supersedes an earlier failure on the same entity", () => {
  const items = attentionItems([
    job({ id: "new", status: "succeeded", created_at: iso(0.5) }),
    job({ id: "old", status: "failed", error: "x", created_at: iso(2) }),
  ], NOW);
  assert.deepEqual(items, []);
});

test("a later failure after an earlier success is shown", () => {
  const items = attentionItems([
    job({ id: "old", status: "succeeded", created_at: iso(3) }),
    job({ id: "new", status: "failed", error: "boom", created_at: iso(0.2) }),
  ], NOW);
  assert.deepEqual(items.map((i) => i.id), ["new"]);
});

test("different entities do not supersede each other", () => {
  const items = attentionItems([
    job({ id: "ok", status: "succeeded", entity_id: "s1" }),
    job({ id: "bad", status: "failed", entity_id: "s2", error: "e" }),
  ], NOW);
  assert.deepEqual(items.map((i) => i.id), ["bad"]);
});

test("ordering: failed, then partial, then running; newest first inside a group", () => {
  const items = attentionItems([
    job({ id: "run", status: "running", entity_id: "s1", stage: "dense_fusion", created_at: iso(0.1), completed_at: null }),
    job({ id: "part", status: "partial", entity_id: "s2", created_at: iso(1) }),
    job({ id: "f-old", status: "failed", entity_id: "s3", error: "e", created_at: iso(5), completed_at: iso(5) }),
    job({ id: "f-new", status: "failed", entity_id: "s4", error: "e", created_at: iso(2), completed_at: iso(2) }),
  ], NOW);
  assert.deepEqual(items.map((i) => i.id), ["f-new", "f-old", "part", "run"]);
  assert.equal(items.find((i) => i.id === "run")?.detail, "Stage: dense_fusion");
  assert.equal(items.find((i) => i.id === "run")?.title, "Reconstruction in progress");
});

test("queued jobs show as in progress", () => {
  const [it] = attentionItems([job({ id: "q", status: "queued", completed_at: null })], NOW);
  assert.equal(it.kind, "running");
});

test(`failures older than ${STALE_AFTER_DAYS} days are history, not a to-do; running jobs never go stale`, () => {
  const items = attentionItems([
    job({ id: "old-fail", status: "failed", entity_id: "s1", error: "e", created_at: iso(30), completed_at: iso(30) }),
    job({ id: "old-run", status: "running", entity_id: "s2", created_at: iso(30), started_at: iso(30), completed_at: null }),
  ], NOW);
  assert.deepEqual(items.map((i) => i.id), ["old-run"]);
});

test("evidence-processing failures link to the photo; unknown job types still show with their raw name", () => {
  const items = attentionItems([
    job({ id: "p", type: "PROCESS_EVIDENCE", entity_type: "evidence", entity_id: "ev_1", status: "failed", error: "bad file" }),
    job({ id: "x", type: "SOMETHING_NEW", entity_type: "thing", entity_id: "t1", status: "failed", error: "e" }),
  ], NOW);
  const p = items.find((i) => i.id === "p")!;
  assert.equal(p.title, "Photo processing failed");
  assert.equal(p.href, "/evidence/ev_1");
  const x = items.find((i) => i.id === "x")!;
  assert.equal(x.title, "SOMETHING_NEW failed");
  assert.equal(x.href, null);
});

test("the list is capped", () => {
  const jobs = Array.from({ length: 10 }, (_, i) => job({ id: `j${i}`, status: "failed", entity_id: `s${i}`, error: "e" }));
  assert.equal(attentionItems(jobs, NOW, 4).length, 4);
  assert.equal(attentionItems(jobs, NOW).length, 6);
});

test("a raw traceback is reduced to its last line; no file paths reach the screen", () => {
  const tb = [
    "Traceback (most recent call last):",
    String.raw`  File "C:\Users\x\repo\apps\api\jobs.py", line 1367, in run`,
    "    outcome = await h()",
    "RuntimeError: no usable photos",
  ].join("\n");
  assert.equal(readableFailure(tb), "no usable photos");
  const [it] = attentionItems([job({ id: "t", status: "failed", error: tb })], NOW);
  assert.equal(it.detail, "no usable photos");
  assert.ok(!/Users|\.py|Traceback/.test(it.detail ?? ""));
});

test("readableFailure keeps plain messages, handles empty, and strips only a leading exception class", () => {
  assert.equal(readableFailure("World has 0 usable photo evidence"), "World has 0 usable photo evidence");
  assert.equal(readableFailure(null), null);
  assert.equal(readableFailure("   "), null);
  assert.equal(readableFailure(["Traceback (most recent call last):", "  File x", "ValueError: bad: thing"].join("\n")), "bad: thing");
});
