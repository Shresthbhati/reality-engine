/**
 * Mobile capture logic: upload-queue state machine + the per-photo / per-world
 * view-models. Every label must trace to a backend field; nothing is invented.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  EMPTY_QUEUE,
  nextToUpload,
  outcomeFromResponse,
  queueReducer,
  summarize,
  type QueueAction,
  type QueueState,
} from "../../src/lib/mobile/upload-queue.ts";
import {
  parseEvidenceQuality,
  photoResult,
  resultLabel,
  worldHeadline,
  type EvidenceStatusLike,
  type WorldStatusLike,
} from "../../src/lib/mobile/capture-result.ts";

const run = (actions: QueueAction[], from: QueueState = EMPTY_QUEUE) => actions.reduce(queueReducer, from);
const add = (...ids: string[]): QueueAction => ({ type: "add", items: ids.map((id) => ({ id, name: `${id}.jpg`, size: 1000 })) });

test("queue: happy path queued -> uploading -> accepted, progress is monotonic", () => {
  let s = run([add("a"), { type: "start", id: "a" }]);
  assert.equal(s.items[0].status, "uploading");
  assert.equal(s.items[0].attempts, 1);
  s = run([{ type: "progress", id: "a", loaded: 600, total: 1000 }], s);
  assert.equal(s.items[0].progress, 0.6);
  s = run([{ type: "progress", id: "a", loaded: 200, total: 1000 }], s); // late event
  assert.equal(s.items[0].progress, 0.6);
  s = run([{ type: "progress", id: "a", loaded: 5000, total: 1000 }], s); // clamped
  assert.equal(s.items[0].progress, 1);
  s = run([{ type: "accepted", id: "a", evidenceId: "ev-1" }], s);
  assert.equal(s.items[0].status, "accepted");
  assert.equal(s.items[0].evidenceId, "ev-1");
});

test("queue: invalid transitions are ignored (no accepting a queued photo, no progress when not uploading)", () => {
  const queued = run([add("a")]);
  assert.equal(run([{ type: "accepted", id: "a", evidenceId: "x" }], queued), queued);
  assert.equal(run([{ type: "progress", id: "a", loaded: 1, total: 2 }], queued), queued);
  assert.equal(run([{ type: "retry", id: "a" }], queued), queued);
  assert.equal(run([{ type: "start", id: "missing" }], queued), queued);
  assert.equal(run([add("a")], queued), queued, "duplicate id is not added twice");
});

test("queue: failed -> retry re-queues, clears the error and counts the attempt", () => {
  let s = run([add("a"), { type: "start", id: "a" }, { type: "failed", id: "a", error: "API unreachable", retryable: true }]);
  assert.equal(s.items[0].status, "failed");
  assert.equal(s.items[0].error, "API unreachable");
  s = run([{ type: "retry", id: "a" }], s);
  assert.deepEqual([s.items[0].status, s.items[0].error, s.items[0].progress], ["queued", null, 0]);
  s = run([{ type: "start", id: "a" }], s);
  assert.equal(s.items[0].attempts, 2);
});

test("queue: an uploading photo cannot be removed; others can", () => {
  const s = run([add("a", "b"), { type: "start", id: "a" }]);
  assert.equal(run([{ type: "remove", id: "a" }], s), s);
  assert.deepEqual(run([{ type: "remove", id: "b" }], s).items.map((i) => i.id), ["a"]);
});

test("queue: one upload at a time, in order", () => {
  let s = run([add("a", "b", "c")]);
  assert.equal(nextToUpload(s, null)?.id, "a");
  s = run([{ type: "start", id: "a" }], s);
  assert.equal(nextToUpload(s, null), null, "a is uploading");
  s = run([{ type: "accepted", id: "a", evidenceId: "e" }], s);
  assert.equal(nextToUpload(s, "w1")?.id, "b");
});

test("queue: a failed FIRST photo holds the queue until a world exists (never creates a second world)", () => {
  let s = run([add("a", "b"), { type: "start", id: "a" }, { type: "failed", id: "a", error: "x", retryable: true }]);
  assert.equal(nextToUpload(s, null), null, "b must wait: no world id yet");
  assert.equal(nextToUpload(s, "w1")?.id, "b", "once a world exists a failure does not block the rest");
  s = run([{ type: "remove", id: "a" }], s);
  assert.equal(nextToUpload(s, null)?.id, "b", "removing the failed photo releases the queue");
});

test("queue: summary counts every status", () => {
  const s = run([
    add("a", "b", "c", "d", "e"),
    { type: "start", id: "a" },
    { type: "accepted", id: "a", evidenceId: "e1" },
    { type: "start", id: "b" },
    { type: "rejected", id: "b", reason: "not an image" },
    { type: "start", id: "c" },
    { type: "failed", id: "c", error: "boom", retryable: false },
    { type: "start", id: "d" },
  ]);
  assert.deepEqual(summarize(s), { total: 5, queued: 1, uploading: 1, accepted: 1, rejected: 1, failed: 1, pending: 2 });
});

test("outcomeFromResponse: accepted evidence, rejected reason, or an explicit fallback", () => {
  assert.deepEqual(outcomeFromResponse({ evidence: [{ id: "ev-9" }], rejected: [] }), { kind: "accepted", evidenceId: "ev-9" });
  assert.deepEqual(outcomeFromResponse({ evidence: [], rejected: [{ name: "x.jpg", reason: "unsupported file type" }] }), {
    kind: "rejected",
    reason: "unsupported file type",
  });
  assert.equal(outcomeFromResponse({ evidence: [], rejected: [] }).kind, "rejected");
});

const ev = (over: Partial<EvidenceStatusLike> = {}): EvidenceStatusLike => ({
  id: "ev-1",
  evidence_class: "photograph",
  used_for_geometry: true,
  registered: null,
  contribution: null,
  in_current_model: false,
  registration: null,
  ...over,
});
const accepted = { status: "accepted", progress: 1, error: null, rejectedReason: null } as const;

test("photoResult: pre-acceptance states come from the queue only", () => {
  assert.equal(photoResult({ status: "queued", progress: 0, error: null, rejectedReason: null }, undefined).label, "Queued");
  assert.equal(photoResult({ status: "uploading", progress: 0.456, error: null, rejectedReason: null }, undefined).label, "Uploading 46%");
  assert.equal(photoResult({ status: "uploading", progress: 1, error: null, rejectedReason: null }, undefined).label, "Sent");
  const failed = photoResult({ status: "failed", progress: 0.2, error: "timeout", rejectedReason: null }, undefined);
  assert.deepEqual([failed.tone, failed.label, failed.detail], ["bad", "Upload failed", "timeout"]);
  const rej = photoResult({ status: "rejected", progress: 1, error: null, rejectedReason: "corrupt image" }, undefined);
  assert.deepEqual([rej.tone, rej.detail], ["bad", "corrupt image"]);
});

test("photoResult: accepted photo reports only what the backend measured", () => {
  assert.equal(photoResult(accepted, undefined).label, "Accepted");
  assert.equal(photoResult(accepted, ev()).tone, "working", "not analysed yet");
  const placed = photoResult(accepted, ev({ registered: true, contribution: "new_view" }));
  assert.deepEqual([placed.tone, placed.label, placed.detail], ["good", "Placed in the model", "Adds a new viewpoint"]);
  const waiting = photoResult(accepted, ev({ registration: { state: "waiting", attempts: 2, ever_registered: false } }));
  assert.deepEqual([waiting.tone, waiting.label], ["warn", "Waiting to be placed"]);
  assert.match(waiting.detail ?? "", /tried 2 times/);
  const unplaced = photoResult(accepted, ev({ registered: false, contribution: "disconnected" }));
  assert.deepEqual([unplaced.tone, unplaced.label], ["warn", "Not placed"]);
  const ctx = photoResult(accepted, ev({ used_for_geometry: false, evidence_class: "floor_plan" }));
  assert.deepEqual([ctx.tone, ctx.label], ["neutral", "Kept as context"]);
  assert.match(ctx.detail ?? "", /floor_plan/);
});

const status = (over: Partial<WorldStatusLike> = {}): WorldStatusLike => ({
  state: "READY_TO_INSPECT",
  state_label: "Ready to inspect",
  in_progress: false,
  has_model: true,
  model: { model_state: "PARTIAL", images_registered: 3, images_used: 5 },
  failure: null,
  last_run: { adopted: true },
  guidance: [],
  ...over,
});
const copy = (m: "ROUGH" | "PARTIAL" | "REFINED" | null) => ({ label: m ?? "none", detail: `detail-${m}` });

test("worldHeadline: ready model states the backend model state and placed count, never a percentage", () => {
  const h = worldHeadline(status({ guidance: [{ kind: "k", message: "Photograph the left wall.", basis: "b" }] }), copy);
  assert.deepEqual([h.tone, h.headline, h.placed], ["good", "Ready to inspect", "3 of 5 photos placed"]);
  assert.equal(h.detail, "PARTIAL. detail-PARTIAL");
  assert.equal(h.guidance, "Photograph the left wall.", "guidance is the backend text, verbatim");
  assert.doesNotMatch(JSON.stringify(h), /%/);
});

test("worldHeadline: no guidance from the backend means no guidance on the phone", () => {
  assert.equal(worldHeadline(status(), copy).guidance, null);
});

test("worldHeadline: failure kinds are distinguished and the previous model is mentioned honestly", () => {
  const failed = worldHeadline(status({ state: "FAILED", failure: { kind: "error", message: "COLMAP crashed" } }), copy);
  assert.deepEqual([failed.tone, failed.headline], ["bad", "Could not finish"]);
  assert.match(failed.detail ?? "", /COLMAP crashed.*previous model/);
  const none = worldHeadline(status({ has_model: false, model: null, failure: { kind: "error", message: null } }), copy);
  assert.doesNotMatch(none.detail ?? "", /previous model/);
  assert.equal(worldHeadline(status({ failure: { kind: "cancelled", message: null } }), copy).tone, "neutral");
  assert.equal(worldHeadline(status({ failure: { kind: "insufficient_evidence", message: "no usable photos" } }), copy).tone, "warn");
});

test("worldHeadline: in-progress, not-adopted and unchanged runs are not reported as success", () => {
  assert.equal(worldHeadline(status({ in_progress: true, state_label: "Reconstructing" }), copy).tone, "working");
  const kept = worldHeadline(status({ last_run: { adopted: false } }), copy);
  assert.deepEqual([kept.tone, kept.headline], ["warn", "Kept your current model"]);
  assert.equal(worldHeadline(status({ last_run: { adopted: true, unchanged: true } }), copy).headline, "Already up to date");
  const empty = worldHeadline(status({ has_model: false, model: null, state: "CAPTURING", state_label: "Capturing" }), copy);
  assert.equal(empty.placed, null);
});

test("parseEvidenceQuality: accepts array or {items}, ignores malformed rows, never invents a score", () => {
  const m = parseEvidenceQuality({
    items: [
      { evidence_id: "a", result: "REDUNDANT", score: 0.31 },
      { id: "b", verdict: "useful" },
      { id: "c" },
      { result: "useful" },
      null,
      { id: "d", result: "ok", score: "high" },
    ],
  });
  assert.deepEqual([...m.keys()], ["a", "b", "d"]);
  assert.deepEqual(m.get("a"), { result: "REDUNDANT", score: 0.31 });
  assert.equal(m.get("b")?.score, null);
  assert.equal(m.get("d")?.score, null);
  assert.equal(parseEvidenceQuality(undefined).size, 0);
  assert.equal(parseEvidenceQuality([{ id: "z", result: "x", score: 1 }]).size, 1);
  assert.equal(resultLabel("needs_more-views"), "Needs more views");
});
