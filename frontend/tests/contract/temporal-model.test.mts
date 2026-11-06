/**
 * Contract tests for Multi-Epoch Temporal Client Model (Epic E07-1).
 * Verifies change categorization, change counters, semantic color tokens, and filtering.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  getTemporalChangeColor,
  countChangesByKind,
  filterTemporalChanges,
  type TemporalEntityChangeItem,
} from "../../src/lib/studio/temporal-model.ts";

const sampleChanges: TemporalEntityChangeItem[] = [
  { entity_id: "e-1", name: "Dining Table", change_kind: "moved", displacement_m: 1.25, confidence: 0.95 },
  { entity_id: "e-2", name: "Armchair", change_kind: "added", confidence: 0.92 },
  { entity_id: "e-3", name: "Old Desk", change_kind: "removed", confidence: 0.98 },
  { entity_id: "e-4", name: "Partition Wall", change_kind: "renovated", details: "Wall shortened by 1m", confidence: 0.96 },
  { entity_id: "e-5", name: "Concrete Column", change_kind: "damaged", details: "Surface crack", confidence: 0.90 },
  { entity_id: "e-6", name: "Exterior Window", change_kind: "unchanged", confidence: 0.99 },
];

test("getTemporalChangeColor provides distinct tokens and hex values for each kind", () => {
  const addedCol = getTemporalChangeColor("added");
  assert.equal(addedCol.hex, 0x10b981);
  assert.ok(addedCol.border.includes("success"));

  const removedCol = getTemporalChangeColor("removed");
  assert.equal(removedCol.hex, 0xf43f5e);
  assert.ok(removedCol.border.includes("error"));

  const movedCol = getTemporalChangeColor("moved");
  assert.equal(movedCol.hex, 0xf59e0b);
  assert.ok(movedCol.border.includes("warning"));

  const renovatedCol = getTemporalChangeColor("renovated");
  assert.equal(renovatedCol.hex, 0x0ea5e9);
  assert.ok(renovatedCol.border.includes("info"));

  const damagedCol = getTemporalChangeColor("damaged");
  assert.equal(damagedCol.hex, 0xa855f7);
  assert.ok(damagedCol.border.includes("purple"));
  // each kind still has its own colour token (no two kinds collapsed onto one semantic colour)
  const borders = ["added", "removed", "moved", "renovated", "damaged"].map((k) => getTemporalChangeColor(k as never).border);
  assert.equal(new Set(borders).size, borders.length);
});

test("countChangesByKind aggregates counts across all categories", () => {
  const counts = countChangesByKind(sampleChanges);
  assert.equal(counts.added, 1);
  assert.equal(counts.removed, 1);
  assert.equal(counts.moved, 1);
  assert.equal(counts.renovated, 1);
  assert.equal(counts.damaged, 1);
  assert.equal(counts.unchanged, 1);
});

test("filterTemporalChanges filters by kind and search query", () => {
  const movedOnly = filterTemporalChanges(sampleChanges, "moved");
  assert.equal(movedOnly.length, 1);
  assert.equal(movedOnly[0].entity_id, "e-1");

  const searchTable = filterTemporalChanges(sampleChanges, "all", "table");
  assert.equal(searchTable.length, 1);
  assert.equal(searchTable[0].name, "Dining Table");

  const searchDesk = filterTemporalChanges(sampleChanges, "removed", "desk");
  assert.equal(searchDesk.length, 1);
  assert.equal(searchDesk[0].entity_id, "e-3");
});
