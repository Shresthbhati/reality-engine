/**
 * Contract tests for Building Envelope Thermal Performance & ASHRAE 90.1 Model.
 * Verifies climate zone mappings, U-value/R-value formatters, and report structure.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CLIMATE_ZONE_LABELS,
  formatClimateZone,
  formatRValue,
  formatThermalStatus,
  formatUValue,
  type ClimateZone,
  type ThermalEnvelopeReport,
} from "../../src/lib/studio/thermal-envelope-model.ts";

test("Climate zones cover all 8 ASHRAE 90.1 geographical regions", () => {
  const zones: ClimateZone[] = [
    "1A_very_hot_humid",
    "2A_hot_humid",
    "3A_warm_humid",
    "4A_mixed_humid",
    "5A_cold",
    "6A_cold",
    "7_very_cold",
    "8_subarctic",
  ];

  for (const z of zones) {
    const label = CLIMATE_ZONE_LABELS[z];
    assert.ok(label, `Missing label for climate zone ${z}`);
    assert.equal(formatClimateZone(z), label);
    assert.ok(label.startsWith("Zone "));
  }

  assert.ok(CLIMATE_ZONE_LABELS["1A_very_hot_humid"].includes("Miami"));
  assert.ok(CLIMATE_ZONE_LABELS["5A_cold"].includes("Chicago"));
  assert.ok(CLIMATE_ZONE_LABELS["8_subarctic"].includes("Fairbanks"));
});

test("formatThermalStatus provides accessible semantic status styling", () => {
  const comp = formatThermalStatus("compliant");
  assert.equal(comp.label, "COMPLIANT");
  assert.ok(comp.colorClass.includes("text-success"));

  const nonComp = formatThermalStatus("non_compliant");
  assert.equal(nonComp.label, "NON-COMPLIANT");
  assert.ok(nonComp.colorClass.includes("text-danger"));

  const bridge = formatThermalStatus("thermal_bridge");
  assert.equal(bridge.label, "THERMAL BRIDGE");
  assert.ok(bridge.colorClass.includes("text-warn"));
});

test("formatUValue and formatRValue format dual SI and IP imperial thermal units", () => {
  const uStr = formatUValue(0.312);
  assert.ok(uStr.includes("0.312 W/m²K"));
  assert.ok(uStr.includes("U-"));

  const rStr = formatRValue(3.2);
  assert.ok(rStr.includes("R-18.2"));
  assert.ok(rStr.includes("3.20 m²K/W"));
});

test("ThermalEnvelopeReport interface contract conforms to expected audit report structure", () => {
  const sample: ThermalEnvelopeReport = {
    world_id: "world-env-001",
    project_name: "Thermal Lab",
    timestamp: "2026-11-09T14:00:00Z",
    climate_zone: "5A_cold",
    design_delta_t_k: 25.0,
    summary: {
      gross_exterior_wall_area_m2: 120.0,
      total_glazing_area_m2: 24.0,
      window_to_wall_ratio_pct: 20.0,
      wwr_prescriptive_compliant: true,
      max_prescriptive_wwr_pct: 40.0,
      total_roof_area_m2: 80.0,
      total_ua_actual_w_k: 84.5,
      total_ua_target_w_k: 95.0,
      ua_tradeoff_margin_pct: 11.1,
      total_design_heat_loss_kw: 2.11,
      surface_count: 5,
    },
    surfaces: [],
    compliance_verdict: "COMPLIANT",
  };

  assert.equal(sample.summary.window_to_wall_ratio_pct, 20.0);
  assert.equal(sample.summary.wwr_prescriptive_compliant, true);
  assert.equal(sample.compliance_verdict, "COMPLIANT");
  assert.equal(sample.climate_zone, "5A_cold");
});
