/**
 * Contract tests for Automated Issue & Clash Detection and BCF Model.
 * Verifies clash types, severities, semantic color tokens, and filtering.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  getClashSeverityColor,
  getClashTypeColor,
  countClashesBySeverity,
  filterClashes,
  type ClashItem,
  type ClashReportSummary,
} from "../../src/lib/studio/clash-bcf-model.ts";

const sampleClashes: ClashItem[] = [
  {
    id: "clash-1",
    clash_type: "hard",
    severity: "critical",
    status: "active",
    entity_a_id: "col-101",
    entity_b_id: "duct-201",
    entity_a_name: "Concrete Column C1",
    entity_b_name: "HVAC Main Supply Duct",
    trade_a: "structure",
    trade_b: "mep",
    penetration_depth_m: 0.12,
    clearance_distance_m: 0.0,
    intersection_volume_m3: 0.024,
    clash_point: [2.0, 1.5, 3.0],
    bounds_min: [1.8, 1.3, 2.8],
    bounds_max: [2.2, 1.7, 3.2],
    title: "Hard Clash: Concrete Column C1 & HVAC Duct",
    description: "HVAC duct penetrates structural column without designated sleeve opening.",
    created_at_iso: "2026-11-09T00:00:00Z",
  },
  {
    id: "clash-2",
    clash_type: "clearance",
    severity: "major",
    status: "active",
    entity_a_id: "wall-05",
    entity_b_id: "panel-12",
    entity_a_name: "Partition Wall W5",
    entity_b_name: "Sub-Distribution Board",
    trade_a: "structure",
    trade_b: "mep",
    penetration_depth_m: 0.0,
    clearance_distance_m: 0.025,
    intersection_volume_m3: 0.0,
    clash_point: [5.0, 1.2, 0.5],
    bounds_min: [4.9, 1.1, 0.4],
    bounds_max: [5.1, 1.3, 0.6],
    title: "Clearance Violation: Wall W5 & Panel",
    description: "Required clearance is 0.05m; observed distance is 0.025m.",
    created_at_iso: "2026-11-09T00:00:00Z",
  },
  {
    id: "clash-3",
    clash_type: "duplicate",
    severity: "minor",
    status: "review",
    entity_a_id: "beam-01",
    entity_b_id: "beam-01-dup",
    entity_a_name: "Perimeter Beam B1",
    entity_b_name: "Perimeter Beam B1 Copy",
    trade_a: "structure",
    trade_b: "structure",
    penetration_depth_m: 0.35,
    clearance_distance_m: 0.0,
    intersection_volume_m3: 0.28,
    clash_point: [0.0, 3.5, 2.0],
    bounds_min: [-0.2, 3.3, 1.8],
    bounds_max: [0.2, 3.7, 2.2],
    title: "Duplicate Element: Beam B1",
    description: "Identical bounding volume modeled twice (>95% overlap).",
    created_at_iso: "2026-11-09T00:00:00Z",
  },
];

test("getClashSeverityColor returns distinct semantic tokens and hex codes", () => {
  const crit = getClashSeverityColor("critical");
  assert.equal(crit.hex, 0xf43f5e);
  assert.ok(crit.border.includes("error"));

  const maj = getClashSeverityColor("major");
  assert.equal(maj.hex, 0xf59e0b);
  assert.ok(maj.border.includes("warning"));

  const min = getClashSeverityColor("minor");
  assert.equal(min.hex, 0x0ea5e9);
  assert.ok(min.border.includes("info"));

  const info = getClashSeverityColor("info");
  assert.equal(info.hex, 0x71717a);
});

test("getClashTypeColor returns appropriate badge tokens", () => {
  const hardCol = getClashTypeColor("hard");
  assert.ok(hardCol.border.includes("error"));

  const clearCol = getClashTypeColor("clearance");
  assert.ok(clearCol.border.includes("warning"));

  const dupCol = getClashTypeColor("duplicate");
  assert.ok(dupCol.border.includes("purple"));
});

test("countClashesBySeverity aggregates counts across severities", () => {
  const counts = countClashesBySeverity(sampleClashes);
  assert.equal(counts.critical, 1);
  assert.equal(counts.major, 1);
  assert.equal(counts.minor, 1);
  assert.equal(counts.info, 0);
});

test("filterClashes filters by severity, clash type, and text search", () => {
  const critOnly = filterClashes(sampleClashes, "critical");
  assert.equal(critOnly.length, 1);
  assert.equal(critOnly[0].id, "clash-1");

  const clearOnly = filterClashes(sampleClashes, "all", "clearance");
  assert.equal(clearOnly.length, 1);
  assert.equal(clearOnly[0].id, "clash-2");

  const searchDuct = filterClashes(sampleClashes, "all", "all", "duct");
  assert.equal(searchDuct.length, 1);
  assert.equal(searchDuct[0].entity_b_name, "HVAC Main Supply Duct");

  const searchBeam = filterClashes(sampleClashes, "all", "all", "beam");
  assert.equal(searchBeam.length, 1);
  assert.equal(searchBeam[0].id, "clash-3");
});

test("ClashReportSummary supports trade breakdown and camera viewpoints", () => {
  const report: ClashReportSummary = {
    world_id: "world-123",
    timestamp_iso: "2026-11-09T00:00:00Z",
    total_clashes: 3,
    hard_clashes: 1,
    clearance_clashes: 1,
    duplicate_clashes: 1,
    by_severity: { critical: 1, major: 1, minor: 1, info: 0 },
    by_trade_pair: { "structure-mep": 2, "structure-structure": 1 },
    items: sampleClashes,
  };

  assert.equal(report.total_clashes, 3);
  assert.equal(report.by_trade_pair["structure-mep"], 2);
  assert.equal(report.items[0].clash_point[0], 2.0);
});
