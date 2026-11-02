/**
 * Home cards, build steps and the photo check, as the user reads them.
 *
 * Rules pinned: a card claims only what the API reported (no quality chip without a model, no "not real size" without
 * one); a queued or unknown-stage job is "waiting", never a step it did not report; problems are plain words.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  buildOutcome,
  buildStep,
  cardView,
  fileProblems,
  formatElapsed,
  relativeTime,
  entityLabel,
  statusChip,
  suggestName,
  traceMessage,
  verdictMessage,
} from "../../src/lib/places.ts";

const NOW = Date.parse("2026-10-13T12:00:00Z");

test("buildStep maps only reported stages and falls back to waiting", () => {
  assert.equal(buildStep({ status: "queued", stage: null }), "waiting");
  assert.equal(buildStep({ status: "running", stage: "analyzing_evidence" }), "reading");
  assert.equal(buildStep({ status: "running", stage: "reconstructing" }), "building");
  assert.equal(buildStep({ status: "running", stage: "refining" }), "building");
  assert.equal(buildStep({ status: "running", stage: "committing_version" }), "saving");
  assert.equal(buildStep({ status: "running", stage: "some_new_stage" }), "reading");
  assert.equal(buildStep({ status: "succeeded", stage: "done" }), "waiting");
});

test("buildOutcome", () => {
  assert.equal(buildOutcome({ status: "running" }), "building");
  assert.equal(buildOutcome({ status: "queued" }), "building");
  assert.equal(buildOutcome({ status: "succeeded" }), "ready");
  assert.equal(buildOutcome({ status: "partial" }), "partial");
  assert.equal(buildOutcome({ status: "failed" }), "failed");
  assert.equal(buildOutcome({ status: "cancelled" }), "cancelled");
});

test("a card without a model shows no quality or size chip", () => {
  const v = cardView({ name: "A", photo_count: 4, quality_band: "unknown", scale_state: "relative" }, NOW);
  assert.equal(v.quality, null);
  assert.equal(v.notRealSize, false);
  assert.equal(v.photoLine, "4 photos");
});

test("a card with a model shows photos placed, quality band and relative scale", () => {
  const v = cardView(
    { name: "A", photo_count: 10, placed_count: 7, quality_band: "fair", scale_state: "relative", current_version_id: "v-1", updated_at: "2026-10-13T11:46:00Z" },
    NOW,
  );
  assert.equal(v.photoLine, "7 of 10 photos placed");
  assert.deepEqual(v.quality, { band: "fair", label: "Fair" });
  assert.equal(v.notRealSize, true);
  assert.equal(v.updated, "14 min ago");
  assert.equal(cardView({ name: "A", current_version_id: "v", scale_state: "metric", quality_band: "good", photo_count: 1, placed_count: 1 }, NOW).notRealSize, false);
});

test("a building card shows the real step and a rounded percent", () => {
  const v = cardView({ name: "A", photo_count: 3, active_job: { status: "running", stage: "reconstructing", progress: 25 } }, NOW);
  assert.deepEqual(v.building, { step: "Building the 3D place", percent: 25 });
  assert.equal(cardView({ name: "A", active_job: { status: "queued", stage: null, progress: null } }, NOW).building?.step, "Waiting to start");
  assert.equal(cardView({ name: "A", photo_count: 0 }, NOW).photoLine, "No photos yet");
});

test("relativeTime and formatElapsed", () => {
  assert.equal(relativeTime("2026-10-13T11:59:50Z", NOW), "Just now");
  assert.equal(relativeTime("2026-10-13T09:00:00Z", NOW), "3 h ago");
  assert.equal(relativeTime("2026-10-12T09:00:00Z", NOW), "Yesterday");
  assert.equal(relativeTime("2026-10-10T09:00:00Z", NOW), "3 days ago");
  assert.equal(relativeTime("2026-10-13T11:46:00", NOW), "14 min ago", "a zone-less timestamp is UTC");
  assert.equal(relativeTime("2026-10-13T17:16:00+05:30", NOW), "14 min ago");
  assert.equal(relativeTime(null, NOW), null);
  assert.equal(relativeTime("not a date", NOW), null);
  assert.equal(formatElapsed(42_000), "42 s");
  assert.equal(formatElapsed(125_000), "2 min 05 s");
});

test("fileProblems are plain words", () => {
  const ok = { name: "a.jpg", kind: "photo", duplicate_of: null, decodable: true, too_small: false, relatively_soft: false, looks_like_photo: true, has_gps: false };
  assert.deepEqual(fileProblems(ok), []);
  assert.deepEqual(fileProblems({ ...ok, relatively_soft: true }), ["Looks blurry compared with the rest"]);
  assert.deepEqual(fileProblems({ ...ok, duplicate_of: "b.jpg" }), ["Same photo as b.jpg"]);
  assert.deepEqual(fileProblems({ ...ok, too_small: true }), ["Too small to use"]);
  assert.deepEqual(fileProblems({ ...ok, looks_like_photo: false }), ["Looks like a drawing or screenshot, not a photo"]);
  assert.deepEqual(fileProblems({ ...ok, kind: "other", decodable: null }), ["Can't be opened as a photo"]);
});

test("verdictMessage covers every API verdict and never returns empty", () => {
  for (const v of ["good", "has_problems", "add_angles", "no_usable_photos", "something_new"]) {
    assert.ok(verdictMessage(v).length > 10, v);
  }
});

test("suggestName uses a shared folder name, else the date", () => {
  assert.equal(suggestName([{ name: "a.jpg", webkitRelativePath: "Library/a.jpg" }, { name: "b.jpg", webkitRelativePath: "Library/b.jpg" }], new Date(2026, 9, 13, 8)), "Library");
  assert.equal(suggestName([{ name: "a.jpg" }], new Date(2026, 9, 13, 8)), "Place 2026-10-13");
  assert.equal(suggestName([{ name: "a", webkitRelativePath: "X/a" }, { name: "b", webkitRelativePath: "Y/b" }], new Date(2026, 9, 13, 8)), "Place 2026-10-13");
});

test("entityLabel is plain and never shows a raw code", () => {
  assert.equal(entityLabel("wall"), "Wall");
  assert.equal(entityLabel("STOREY"), "Floor level");
  assert.equal(entityLabel("natural_hazard"), "Natural hazard");
  assert.equal(entityLabel(null), "Unidentified part");
  assert.equal(entityLabel(""), "Unidentified part");
});

test("statusChip follows the priority order and only says good when nothing else applies", () => {
  const base = { building: false, placed: 10, total: 10, scaleState: "metric", weakCount: 0 };
  assert.deepEqual(statusChip(base), { kind: "ok", text: "Looks good" });
  assert.equal(statusChip({ ...base, building: true, placed: 3 }).kind, "building");
  assert.deepEqual(statusChip({ ...base, placed: 7 }), { kind: "unplaced", text: "7 of 10 photos placed" });
  assert.equal(statusChip({ ...base, placed: 7, scaleState: "relative", weakCount: 4 }).kind, "unplaced");
  assert.equal(statusChip({ ...base, scaleState: "relative", weakCount: 4 }).kind, "size");
  assert.deepEqual(statusChip({ ...base, weakCount: 1 }), { kind: "weak", text: "1 weak part" });
  assert.equal(statusChip({ ...base, weakCount: 3 }).text, "3 weak parts");
  assert.equal(statusChip({ ...base, total: 0, placed: 0 }).kind, "ok", "no photos recorded is not photos missing");
});

test("traceMessage never claims more than the engine recorded", () => {
  assert.match(traceMessage("observation"), /Seen in these photos/);
  assert.match(traceMessage("session"), /did not record which ones/);
  assert.match(traceMessage("none"), /generated/);
  assert.match(traceMessage(null), /generated/);
});
