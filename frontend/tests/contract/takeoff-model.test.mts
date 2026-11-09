/**
 * Contract tests for Architectural Bill of Quantities (BoQ) & Material Takeoff Model.
 * Verifies CSI MasterFormat 2020 division mappings, financial formatters, and data shapes.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CSI_DIVISION_LABELS,
  formatCurrency,
  formatCsiDivision,
  formatQuantity,
  type CsiDivision,
  type TakeoffReport,
} from "../../src/lib/studio/takeoff-model.ts";

test("CSI MasterFormat 2020 divisions provide complete architectural coverage", () => {
  const divisions: CsiDivision[] = [
    "03_concrete",
    "04_masonry",
    "06_wood_plastics",
    "07_thermal_moisture",
    "08_openings",
    "09_finishes",
    "22_plumbing",
    "26_electrical",
  ];

  for (const div of divisions) {
    const label = CSI_DIVISION_LABELS[div];
    assert.ok(label, `Missing label for division ${div}`);
    assert.equal(formatCsiDivision(div), label);
    assert.ok(label.startsWith("Division "));
  }

  assert.ok(CSI_DIVISION_LABELS["03_concrete"].includes("Concrete"));
  assert.ok(CSI_DIVISION_LABELS["06_wood_plastics"].includes("Wood"));
  assert.ok(CSI_DIVISION_LABELS["08_openings"].includes("Openings"));
  assert.ok(CSI_DIVISION_LABELS["09_finishes"].includes("Finishes"));
  assert.ok(CSI_DIVISION_LABELS["22_plumbing"].includes("Plumbing"));
  assert.ok(CSI_DIVISION_LABELS["26_electrical"].includes("Electrical"));
});

test("formatCurrency formats monetary amounts with correct precision and symbols", () => {
  assert.equal(formatCurrency(1250.5), "$1,250.50");
  assert.equal(formatCurrency(0), "$0.00");
  assert.equal(formatCurrency(45892.75, "USD"), "$45,892.75");
  assert.equal(formatCurrency(1000, "EUR"), "EUR 1,000.00");
});

test("formatQuantity combines formatted numeric values with unit strings", () => {
  assert.equal(formatQuantity(145.2, "m²"), "145.20 m²");
  assert.equal(formatQuantity(32.0, "m"), "32.00 m");
  assert.equal(formatQuantity(12, "ea"), "12.00 ea");
});

test("TakeoffReport interface contract conforms to expected structure", () => {
  const sampleReport: TakeoffReport = {
    world_id: "world-101",
    project_name: "Executive Suite",
    timestamp: "2026-11-09T10:00:00Z",
    currency: "USD",
    summary: {
      total_floor_area_m2: 120.0,
      total_net_wall_area_m2: 280.5,
      total_openings_count: 8,
      total_fixtures_count: 14,
      grand_total_material: 15400.0,
      grand_total_labor: 18200.0,
      grand_total_cost: 33600.0,
      cost_per_m2: 280.0,
    },
    divisions: [
      {
        division: "09_finishes",
        division_name: "09 - Finishes",
        total_material_cost: 8000.0,
        total_labor_cost: 9500.0,
        total_cost: 17500.0,
        item_count: 6,
        percentage_of_total: 52.1,
      },
    ],
    rooms: [
      {
        room_id: "room-1",
        room_name: "Office",
        floor_area_m2: 30.0,
        perimeter_m: 22.0,
        ceiling_height_m: 2.8,
        wall_gross_area_m2: 61.6,
        wall_opening_area_m2: 5.4,
        wall_net_area_m2: 56.2,
        ceiling_area_m2: 30.0,
        baseboard_length_m: 20.2,
        doors_count: 2,
        windows_count: 1,
        fixtures_count: 4,
        floor_finish: "hardwood",
        wall_finish: "paint",
        ceiling_finish: "acoustic_tile",
        total_cost: 4800.0,
        items: [],
      },
    ],
    items_count: 6,
  };

  assert.equal(sampleReport.summary.grand_total_cost, 33600.0);
  assert.equal(sampleReport.summary.cost_per_m2, 280.0);
  assert.equal(sampleReport.divisions[0].percentage_of_total, 52.1);
  assert.equal(sampleReport.rooms[0].wall_net_area_m2, 56.2);
});
