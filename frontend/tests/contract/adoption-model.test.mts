/**
 * Unit & contract tests for World Adoption view model (Task U.7 / Decision 027).
 *
 * Verifies:
 * 1. Alignment tier metadata and confidence formatting.
 * 2. 5-state change category breakdown (matched, new, updated, uncertain, conflicting).
 * 3. Rule 18 conflict default resolution: keep manual user edit.
 * 4. Total change calculation and change set integrity.
 *
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ALIGNMENT_TIER_META,
  CHANGE_CATEGORY_META,
  formatConfidencePercent,
  defaultConflictResolutions,
  getTotalChangesCount,
} from "../../src/lib/adoptions/adoption-model.ts";
import type { AdoptionChangeSet } from "../../src/lib/api/adoptions.ts";

test("alignment tiers provide clear user feedback and WCAG compliant colors", () => {
  assert.equal(ALIGNMENT_TIER_META.HIGH.label, "High Confidence");
  assert.equal(ALIGNMENT_TIER_META.MODERATE.label, "Moderate Confidence");
  assert.equal(ALIGNMENT_TIER_META.LOW.label, "Low Confidence");

  assert.ok(ALIGNMENT_TIER_META.HIGH.fg.includes("success"));
  assert.ok(ALIGNMENT_TIER_META.MODERATE.fg.includes("warning"));
  assert.ok(ALIGNMENT_TIER_META.LOW.fg.includes("error"));
});

test("formatConfidencePercent formats ratios into whole percentages", () => {
  assert.equal(formatConfidencePercent(0.924), "92%");
  assert.equal(formatConfidencePercent(0.85), "85%");
  assert.equal(formatConfidencePercent(0.599), "60%");
  assert.equal(formatConfidencePercent(1.0), "100%");
});

test("change category metadata describes all 5 states defined in Decision 027", () => {
  const keys = Object.keys(CHANGE_CATEGORY_META);
  assert.deepEqual(keys.sort(), ["conflicting", "matched", "new", "uncertain", "updated"]);

  assert.equal(CHANGE_CATEGORY_META.new.label, "New");
  assert.equal(CHANGE_CATEGORY_META.matched.label, "Matched");
  assert.equal(CHANGE_CATEGORY_META.updated.label, "Updated");
  assert.equal(CHANGE_CATEGORY_META.uncertain.label, "Uncertain");
  assert.equal(CHANGE_CATEGORY_META.conflicting.label, "Conflicting");
});

test("Rule 18: default conflict resolutions strictly preserve manual edits", () => {
  const changeSet: AdoptionChangeSet = {
    matched: ["wall_m1", "floor_f1"],
    new: ["room_kitchen_02", "wall_k01"],
    updated: ["wall_u1"],
    uncertain: ["chair_unc_1"],
    conflicting: ["wall_manual_edit_01", "door_custom_02"],
  };

  const resolutions = defaultConflictResolutions(changeSet);
  assert.deepEqual(resolutions, {
    wall_manual_edit_01: "keep_edit",
    door_custom_02: "keep_edit",
  });
});

test("getTotalChangesCount accurately tallies entities across all 5 states", () => {
  const changeSet: AdoptionChangeSet = {
    matched: ["w1", "w2"],
    new: ["r1", "w3"],
    updated: ["w4"],
    uncertain: ["f1"],
    conflicting: ["w5"],
  };

  assert.equal(getTotalChangesCount(changeSet), 7);

  const emptySet: AdoptionChangeSet = {
    matched: [],
    new: [],
    updated: [],
    uncertain: [],
    conflicting: [],
  };
  assert.equal(getTotalChangesCount(emptySet), 0);
});
