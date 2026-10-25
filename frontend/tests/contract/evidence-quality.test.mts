/**
 * Evidence Quality Firewall view model: counters, filters, tones and the honest "not measured" wording.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  barPercent,
  checkTone,
  contributionSummary,
  countResults,
  filterItems,
  formatBytes,
  percentOf,
  resultOf,
  scoreTone,
} from "../../src/lib/api/evidence-quality-model.ts";

const items = [
  { name: "IMG_1.jpg", result: "accepted" as const },
  { name: "IMG_2.jpg", result: "accepted_with_warning" as const },
  { name: "map.png", result: "needs_review" as const },
  { name: "IMG_4.jpg", result: "rejected" as const },
  { name: "raw.jpg", result: null },
];

test("counters partition the list: every item lands in exactly one bucket", () => {
  const c = countResults(items);
  assert.deepEqual(c, { accepted: 1, accepted_with_warning: 1, needs_review: 1, rejected: 1, not_inspected: 1, total: 5 });
  assert.equal(c.accepted + c.accepted_with_warning + c.needs_review + c.rejected + c.not_inspected, c.total);
});

test("an uninspected item is 'not_inspected', never silently accepted", () => {
  assert.equal(resultOf({ result: null }), "not_inspected");
});

test("tab and search filters combine; search is case-insensitive", () => {
  assert.equal(filterItems(items, "all", "").length, 5);
  assert.deepEqual(filterItems(items, "rejected", "").map((i) => i.name), ["IMG_4.jpg"]);
  assert.deepEqual(filterItems(items, "all", "  MAP ").map((i) => i.name), ["map.png"]);
  assert.deepEqual(filterItems(items, "not_inspected", "").map((i) => i.name), ["raw.jpg"]);
  assert.equal(filterItems(items, "accepted", "map").length, 0);
});

test("percentages are whole numbers and never NaN for an empty list", () => {
  assert.equal(percentOf(1, 3), "33%");
  assert.equal(percentOf(0, 0), "0%");
});

test("tones follow the documented score bands and check statuses", () => {
  assert.equal(scoreTone(100), "success");
  assert.equal(scoreTone(85), "success");
  assert.equal(scoreTone(70), "warning");
  assert.equal(scoreTone(60), "warning");
  assert.equal(scoreTone(59), "error");
  assert.equal(scoreTone(null), "neutral");
  assert.equal(checkTone("pass"), "success");
  assert.equal(checkTone("warn"), "warning");
  assert.equal(checkTone("fail"), "error");
});

test("contribution is described from measured counts, or says exactly why it was not measured", () => {
  assert.equal(
    contributionSummary({ available: true, label: "new_view", verified_matches: 42 }),
    "New viewpoint (42 verified feature matches)",
  );
  assert.equal(contributionSummary({ available: true, label: "first", verified_matches: 0 }), "First image (nothing earlier to compare)");
  assert.equal(
    contributionSummary({ available: false, reason: "No model has been built for this world yet" }),
    "Not measured: No model has been built for this world yet",
  );
});

test("size and bar helpers clamp instead of overflowing", () => {
  assert.equal(formatBytes(null), "Unknown");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatBytes(3 * 1024 * 1024), "3.0 MB");
  assert.equal(barPercent(150, 100), 100);
  assert.equal(barPercent(-5, 100), 0);
  assert.equal(barPercent(25, 100), 25);
  assert.equal(barPercent(5, 0), 0);
});
