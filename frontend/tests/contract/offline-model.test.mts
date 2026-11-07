/** T-5.3 (pure part): which photos are kept for a restart, which come back, and when a failed upload is retried by itself. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { autoRetryDelayMs, idsToRetry, keepInStore, restoredMessage, restorePlan } from "../../src/lib/mobile/offline-model.ts";
import { queueReducer, EMPTY_QUEUE, type QueueItem, type QueueState } from "../../src/lib/mobile/upload-queue.ts";

const item = (over: Partial<QueueItem>): QueueItem => ({
  id: "a", name: "a.jpg", size: 10, status: "queued", progress: 0, attempts: 0, error: null, retryable: true, evidenceId: null, rejectedReason: null, ...over,
});

test("a photo that still needs the network is kept for a restart; a finished one is not", () => {
  for (const s of ["queued", "uploading"] as const) assert.equal(keepInStore(item({ status: s })), true, s);
  assert.equal(keepInStore(item({ status: "failed", retryable: true })), true);
  assert.equal(keepInStore(item({ status: "failed", retryable: false })), false);
  assert.equal(keepInStore(item({ status: "accepted" })), false);
  assert.equal(keepInStore(item({ status: "rejected" })), false);
});

test("restoring adds the stored photos the queue does not already hold, as fresh queued items", () => {
  const stored = [{ id: "a", name: "a.jpg", size: 10 }, { id: "b", name: "b.jpg", size: 20 }];
  const have: QueueState = queueReducer(EMPTY_QUEUE, { type: "add", items: [{ id: "a", name: "a.jpg", size: 10 }] });
  const plan = restorePlan(stored, have);
  assert.deepEqual(plan.add.map((p) => p.id), ["b"]);
  const after = queueReducer(have, { type: "add", items: plan.add });
  assert.deepEqual(after.items.map((i) => [i.id, i.status]), [["a", "queued"], ["b", "queued"]]);
});

test("the restored message names how many photos came back", () => {
  assert.equal(restoredMessage(0), null);
  assert.match(restoredMessage(1) as string, /1 photo from before/);
  assert.match(restoredMessage(3) as string, /3 photos from before/);
});

test("only retryable failures are retried when the connection returns", () => {
  const q: QueueState = { items: [item({ id: "x", status: "failed", retryable: true }), item({ id: "y", status: "failed", retryable: false }), item({ id: "z", status: "accepted" })] };
  assert.deepEqual(idsToRetry(q), ["x"]);
});

test("automatic retry backs off, stops after a few tries, and never retries an unretryable photo", () => {
  assert.equal(autoRetryDelayMs(item({ status: "failed", attempts: 1 })), 2000);
  assert.equal(autoRetryDelayMs(item({ status: "failed", attempts: 2 })), 4000);
  assert.equal(autoRetryDelayMs(item({ status: "failed", attempts: 5 })), 30_000);
  assert.equal(autoRetryDelayMs(item({ status: "failed", attempts: 6 })), null);
  assert.equal(autoRetryDelayMs(item({ status: "failed", attempts: 1, retryable: false })), null);
  assert.equal(autoRetryDelayMs(item({ status: "queued", attempts: 1 })), null);
});
