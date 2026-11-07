/** T-5.4: one calm hint after a batch, from the real coverage API; never invented, never repeated, never nagging. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { pickHint } from "../../src/lib/mobile/coverage-hint.ts";
import type { CaptureCoverageDto } from "../../src/lib/api/capture-coverage-model.ts";

const dto = (over: Partial<CaptureCoverageDto>): CaptureCoverageDto => ({
  available: true, reason: null, walls: [{ index: 1, label: "wall 1", between: ["A", "B"], p0: [0, 0], p1: [1, 0], status: "seen", covered_fraction: 1, multi_view_fraction: 1, hint: null }],
  hints: [], summary: { seen: 1, weak: 0, not_seen: 0 }, ...over,
});
const none = { shown: [] as string[], dismissedKey: null as string | null };

test("no coverage yet means no hint (nothing is invented before a model exists)", () => {
  assert.equal(pickHint(null, none), null);
  assert.equal(pickHint(dto({ available: false, reason: "no model yet" }), none), null);
});

test("the first backend hint is shown, once", () => {
  const d = dto({ hints: ["Take 2 photos of wall 3 (between corner C and D)", "Another"], summary: { seen: 2, weak: 0, not_seen: 1 } });
  const h = pickHint(d, none);
  assert.equal(h?.text, "Take 2 photos of wall 3 (between corner C and D)");
  assert.equal(h?.tone, "info");
  assert.equal(pickHint(d, { shown: [h!.key], dismissedKey: null })?.text, "Another");
});

test("a hint already shown is not repeated and the same hint is silent after it is dismissed", () => {
  const d = dto({ hints: ["Take 2 photos of wall 3"], summary: { seen: 2, weak: 0, not_seen: 1 } });
  const h = pickHint(d, none)!;
  assert.equal(pickHint(d, { shown: [h.key], dismissedKey: null }), null);
  assert.equal(pickHint(d, { shown: [], dismissedKey: h.key }), null);
});

test("when every wall is seen the hint says so, once, in a good tone", () => {
  const h = pickHint(dto({}), none);
  assert.equal(h?.tone, "good");
  assert.match(h!.text, /every wall/i);
  assert.equal(pickHint(dto({}), { shown: [h!.key], dismissedKey: null }), null);
});

test("an early estimate says so", () => {
  const h = pickHint(dto({ hints: ["Take 2 photos of wall 3"], provisional: true, summary: { seen: 2, weak: 0, not_seen: 1 } }), none);
  assert.match(h!.text, /early estimate/i);
});

test("weak walls with no backend hint still get a plain suggestion built from the real status", () => {
  const d = dto({ summary: { seen: 1, weak: 2, not_seen: 0 } });
  const h = pickHint(d, none);
  assert.match(h!.text, /2 walls/);
});
