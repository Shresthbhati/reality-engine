/**
 * Contract tests for MEP & Conduit Routing Topology Model.
 * Verifies system type color tokens, length formatting, cross-section sizing, and topology structures.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  formatMepLength,
  formatMepSize,
  getMepSystemHex,
  type MepFacilityTopology,
  type MepSegment,
} from "../../src/lib/studio/mep-topology-model.ts";

test("getMepSystemHex returns distinct hex colors for all trade systems", () => {
  const elec = getMepSystemHex("electrical");
  const plumb = getMepSystemHex("plumbing");
  const hvac = getMepSystemHex("hvac");
  const fire = getMepSystemHex("fire_protection");
  const data = getMepSystemHex("data_comm");

  assert.equal(elec, "#f59e0b");
  assert.equal(plumb, "#06b6d4");
  assert.equal(hvac, "#10b981");
  assert.equal(fire, "#f43f5e");
  assert.equal(data, "#8b5cf6");
  assert.notEqual(elec, plumb);
  assert.notEqual(elec, hvac);
});

test("formatMepLength formats metric distance accurately", () => {
  assert.equal(formatMepLength(0.0), "0.00 m");
  assert.equal(formatMepLength(4.256), "4.26 m");
  assert.equal(formatMepLength(1250.0), "1.25 km");
});

test("formatMepSize formats circular and rectangular cross-sections correctly", () => {
  const circularSeg: MepSegment = {
    segment_id: "seg-1",
    name: "Conduit 1",
    system_type: "electrical",
    element_type: "conduit",
    start_point: [0, 0, 0],
    end_point: [1, 0, 0],
    waypoints: [[0, 0, 0], [1, 0, 0]],
    cross_section: "circular",
    diameter_mm: 25.0,
    width_mm: 0,
    height_mm: 0,
    length_m: 1.0,
    ports: [],
    bounds: { min: [-0.01, -0.01, -0.01], max: [1.01, 0.01, 0.01] },
    provenance: "observed",
  };
  assert.equal(formatMepSize(circularSeg), "Ø 25 mm");

  const rectSeg: MepSegment = {
    segment_id: "seg-2",
    name: "Duct 1",
    system_type: "hvac",
    element_type: "duct",
    start_point: [0, 0, 0],
    end_point: [2, 0, 0],
    waypoints: [[0, 0, 0], [2, 0, 0]],
    cross_section: "rectangular",
    diameter_mm: 0,
    width_mm: 300.0,
    height_mm: 200.0,
    length_m: 2.0,
    ports: [],
    bounds: { min: [-0.1, -0.1, -0.1], max: [2.1, 0.1, 0.1] },
    provenance: "observed",
  };
  assert.equal(formatMepSize(rectSeg), "300×200 mm");
});

test("MepFacilityTopology parses full structure and checks compliance", () => {
  const sampleTopo: MepFacilityTopology = {
    world_id: "wld-test",
    total_conduit_length_m: 15.4,
    total_pipe_length_m: 8.2,
    total_duct_length_m: 12.0,
    summary: {
      system_count: 3,
      total_elements: 14,
      total_run_length_m: 35.6,
      clearance_violations_count: 0,
      code_compliant: true,
    },
    clearance_violations: [],
    systems: [
      {
        system_id: "sys-elec",
        name: "Power System",
        system_type: "electrical",
        segment_count: 3,
        fitting_count: 2,
        terminal_count: 4,
        total_length_m: 15.4,
        total_volume_m3: 0.0075,
        is_connected_dag: true,
        segments: [],
        fittings: [],
        terminals: [],
      },
    ],
  };

  assert.equal(sampleTopo.world_id, "wld-test");
  assert.equal(sampleTopo.summary.code_compliant, true);
  assert.equal(sampleTopo.systems.length, 1);
  assert.equal(sampleTopo.systems[0].system_type, "electrical");
});
