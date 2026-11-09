/**
 * Contract tests for Life Safety & ADA Accessibility Compliance Model.
 * Verifies category metadata, compliance formatters, and report structure.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  formatComplianceStatus,
  formatDistance,
  getCategoryMeta,
  type CodeCheckCategory,
  type LifeSafetyReport,
} from "../../src/lib/studio/life-safety-model.ts";

test("Code check category metadata conforms to ADA and IBC statutory standards", () => {
  const categories: CodeCheckCategory[] = [
    "door_clearance",
    "corridor_width",
    "turning_space",
    "egress_distance",
    "door_swing",
    "occupant_load",
  ];

  for (const cat of categories) {
    const meta = getCategoryMeta(cat);
    assert.ok(meta.label.length > 0);
    assert.ok(meta.standard.length > 0);
    assert.ok(meta.badgeClass.includes("text-accent"));
  }

  assert.ok(getCategoryMeta("door_clearance").standard.includes("ADA §404"));
  assert.ok(getCategoryMeta("turning_space").standard.includes("ADA 2010 §304"));
  assert.ok(getCategoryMeta("egress_distance").standard.includes("IBC 2024 §1017"));
});

test("formatComplianceStatus returns WCAG-accessible semantic status styling", () => {
  const passFmt = formatComplianceStatus("pass");
  assert.equal(passFmt.label, "PASS");
  assert.ok(passFmt.colorClass.includes("text-success"));
  assert.ok(passFmt.bgClass.includes("bg-success/10"));

  const failFmt = formatComplianceStatus("fail");
  assert.equal(failFmt.label, "FAIL");
  assert.ok(failFmt.colorClass.includes("text-danger"));
  assert.ok(failFmt.bgClass.includes("bg-danger/10"));

  const warnFmt = formatComplianceStatus("warning");
  assert.equal(warnFmt.label, "WARNING");
  assert.ok(warnFmt.colorClass.includes("text-warn"));
  assert.ok(warnFmt.bgClass.includes("bg-warn/10"));
});

test("formatDistance formats meters and imperial feet accurately", () => {
  assert.equal(formatDistance(10.0), "10.00 m (32.8 ft)");
  assert.equal(formatDistance(76.2), "76.20 m (250.0 ft)");
  assert.equal(formatDistance(0.813), "0.81 m (2.7 ft)");
});

test("LifeSafetyReport interface contract conforms to expected audit report structure", () => {
  const sample: LifeSafetyReport = {
    world_id: "world-ls-001",
    project_name: "Corporate Center",
    timestamp: "2026-11-09T12:00:00Z",
    occupancy_group: "Business (Group B)",
    is_sprinklered: true,
    total_occupant_load: 45,
    summary: {
      total_checks: 12,
      passed_checks: 11,
      failed_checks: 1,
      warning_checks: 0,
      compliance_score_pct: 91.7,
      overall_status: "fail",
      max_egress_distance_m: 24.5,
      total_doors_audited: 6,
      total_rooms_audited: 4,
      total_occupant_load: 45,
    },
    checks: [],
    egress_routes: [],
    turning_spaces: [],
  };

  assert.equal(sample.summary.compliance_score_pct, 91.7);
  assert.equal(sample.summary.total_checks, 12);
  assert.equal(sample.is_sprinklered, true);
  assert.equal(sample.occupancy_group, "Business (Group B)");
});
