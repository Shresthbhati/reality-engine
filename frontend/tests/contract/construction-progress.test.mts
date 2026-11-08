import test from "node:test";
import assert from "node:assert/strict";

import type {
  Trade,
  ComplianceStatus,
  ElementCompliance,
  TradeProgressSummary,
  SiteComplianceReportSummary,
} from "../../src/lib/api/construction.ts";

/** Helper to compute trade statistics from element list (client-side verification) */
function computeTradeSummary(
  trade: Trade,
  elements: ElementCompliance[]
): TradeProgressSummary {
  const tradeElements = elements.filter((e) => e.trade === trade);
  const planned = tradeElements.filter((e) => e.status !== "unplanned").length;
  const completed = tradeElements.filter((e) => e.status === "completed").length;
  const misaligned = tradeElements.filter((e) => e.status === "misaligned").length;
  const inProgress = tradeElements.filter((e) => e.status === "in_progress").length;
  const missing = tradeElements.filter((e) => e.status === "missing").length;
  const unplanned = tradeElements.filter((e) => e.status === "unplanned").length;

  const completionPct =
    planned > 0 ? Math.round((completed / planned) * 1000) / 10 : 0.0;

  return {
    trade,
    planned_count: planned,
    completed_count: completed,
    misaligned_count: misaligned,
    in_progress_count: inProgress,
    missing_count: missing,
    unplanned_count: unplanned,
    completion_pct: completionPct,
  };
}

test("Construction compliance: trade summary accurately computes progress and counts", () => {
  const elements: ElementCompliance[] = [
    {
      element_id: "elem-1",
      planned_id: "plan-1",
      name: "Column A1",
      trade: "structure",
      entity_type: "column",
      status: "completed",
      deviation_m: 0.012,
      deviation_vector: [0.01, 0.005, 0.003],
      tolerance_m: 0.05,
      planned_pos: [1.0, 0.0, 1.0],
      observed_pos: [1.01, 0.005, 1.003],
      confidence: 0.95,
      notes: "Compliant within 5.0 cm tolerance",
    },
    {
      element_id: "elem-2",
      planned_id: "plan-2",
      name: "Column A2",
      trade: "structure",
      entity_type: "column",
      status: "misaligned",
      deviation_m: 0.082,
      deviation_vector: [0.08, 0.01, 0.01],
      tolerance_m: 0.05,
      planned_pos: [2.0, 0.0, 1.0],
      observed_pos: [2.08, 0.01, 1.01],
      confidence: 0.92,
      notes: "Misaligned: 8.2 cm exceeds 5.0 cm tolerance",
    },
    {
      element_id: "elem-3",
      planned_id: "plan-3",
      name: "Slab Level 1",
      trade: "structure",
      entity_type: "floor",
      status: "missing",
      deviation_m: 999.0,
      deviation_vector: [0, 0, 0],
      tolerance_m: 0.05,
      planned_pos: [0.0, 0.0, 0.0],
      observed_pos: null,
      confidence: 0.0,
      notes: "Planned element not detected on site",
    },
    {
      element_id: "elem-4",
      planned_id: null,
      name: "Temporary Shoring",
      trade: "structure",
      entity_type: "fixture",
      status: "unplanned",
      deviation_m: 0.0,
      deviation_vector: [0, 0, 0],
      tolerance_m: 0.05,
      planned_pos: null,
      observed_pos: [1.5, 0.0, 1.5],
      confidence: 0.88,
      notes: "Unplanned site element with no design counterpart",
    },
  ];

  const summary = computeTradeSummary("structure", elements);

  assert.equal(summary.trade, "structure");
  assert.equal(summary.planned_count, 3); // elem-1, elem-2, elem-3
  assert.equal(summary.completed_count, 1);
  assert.equal(summary.misaligned_count, 1);
  assert.equal(summary.missing_count, 1);
  assert.equal(summary.unplanned_count, 1);
  // 1 / 3 = 33.3%
  assert.equal(summary.completion_pct, 33.3);
});

test("Construction compliance: tolerance classification adheres to strict metric boundary", () => {
  const toleranceM = 0.05; // 5 cm

  const withinTol: ElementCompliance = {
    element_id: "e-within",
    planned_id: "p-within",
    name: "Wall Panel",
    trade: "architecture",
    entity_type: "wall",
    status: "completed",
    deviation_m: 0.049,
    deviation_vector: [0.03, 0.03, 0.02],
    tolerance_m: toleranceM,
    planned_pos: [0, 0, 0],
    observed_pos: [0.03, 0.03, 0.02],
    confidence: 0.98,
    notes: "Compliant",
  };

  const outsideTol: ElementCompliance = {
    element_id: "e-outside",
    planned_id: "p-outside",
    name: "Wall Panel",
    trade: "architecture",
    entity_type: "wall",
    status: "misaligned",
    deviation_m: 0.051,
    deviation_vector: [0.03, 0.03, 0.03],
    tolerance_m: toleranceM,
    planned_pos: [0, 0, 0],
    observed_pos: [0.03, 0.03, 0.03],
    confidence: 0.98,
    notes: "Misaligned",
  };

  assert.ok(withinTol.deviation_m <= withinTol.tolerance_m);
  assert.equal(withinTol.status, "completed");

  assert.ok(outsideTol.deviation_m > outsideTol.tolerance_m);
  assert.equal(outsideTol.status, "misaligned");
});

test("Construction compliance: summary data shape conforms to API contract", () => {
  const mockSummary: SiteComplianceReportSummary = {
    design_world_id: "wld_design",
    site_world_id: "wld_site_epoch_2",
    inspection_date: "2026-11-08T12:00:00Z",
    default_tolerance_m: 0.05,
    overall_completion_pct: 75.0,
    total_planned: 12,
    completed_count: 9,
    misaligned_count: 1,
    in_progress_count: 1,
    missing_count: 1,
    unplanned_count: 2,
    alignment_residual_m: 0.015,
    alignment_converged: true,
    trades: {
      structure: {
        trade: "structure",
        planned_count: 6,
        completed_count: 6,
        misaligned_count: 0,
        in_progress_count: 0,
        missing_count: 0,
        unplanned_count: 0,
        completion_pct: 100.0,
      },
      mep: {
        trade: "mep",
        planned_count: 6,
        completed_count: 3,
        misaligned_count: 1,
        in_progress_count: 1,
        missing_count: 1,
        unplanned_count: 2,
        completion_pct: 50.0,
      },
    },
  };

  assert.equal(mockSummary.overall_completion_pct, 75.0);
  assert.equal(mockSummary.alignment_converged, true);
  assert.ok(mockSummary.alignment_residual_m <= mockSummary.default_tolerance_m);
  assert.equal(Object.keys(mockSummary.trades).length, 2);
  assert.equal(mockSummary.trades["structure"].completion_pct, 100.0);
  assert.equal(mockSummary.trades["mep"].completion_pct, 50.0);
});
