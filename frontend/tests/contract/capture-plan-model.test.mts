/**
 * Contract tests for Capture Plan Client Model (Epic E08-1).
 * Verifies Next-Best-View planning, information gain sorting, compass orientation, and tilt angles.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  formatCompassHeading,
  formatPitchAngle,
  calculateCoverageImprovementPct,
  filterAndSortViewpoints,
  describeCameraPose,
  type RecommendedPose,
} from "../../src/lib/studio/capture-plan-model.ts";

test("Compass heading formats correct 8-point compass orientations", () => {
  assert.equal(formatCompassHeading(0), "N");
  assert.equal(formatCompassHeading(45), "NE");
  assert.equal(formatCompassHeading(90), "E");
  assert.equal(formatCompassHeading(135), "SE");
  assert.equal(formatCompassHeading(180), "S");
  assert.equal(formatCompassHeading(225), "SW");
  assert.equal(formatCompassHeading(270), "W");
  assert.equal(formatCompassHeading(315), "NW");
  assert.equal(formatCompassHeading(360), "N");

  // Handles negative yaw angles seamlessly
  assert.equal(formatCompassHeading(-90), "W");
  assert.equal(formatCompassHeading(-180), "S");
});

test("Pitch angle formats natural-language instructions", () => {
  assert.equal(formatPitchAngle(0), "Level (horizontal)");
  assert.equal(formatPitchAngle(1.5), "Level (horizontal)");
  assert.equal(formatPitchAngle(15.2), "Tilt up 15°");
  assert.equal(formatPitchAngle(-20.4), "Tilt down 20°");
});

test("Coverage improvement percentage clamps correctly between 0 and 100", () => {
  assert.equal(calculateCoverageImprovementPct(0.80, 0.478), 47.8);
  assert.equal(calculateCoverageImprovementPct(0.0, 0.50), 0);
  assert.equal(calculateCoverageImprovementPct(0.60, 1.2), 100);
});

test("Filter and sort viewpoints orders strictly by descending information gain", () => {
  const poses: RecommendedPose[] = [
    {
      viewpoint_id: "v-low",
      position: [1.0, 2.0, 1.6],
      look_direction: [0, 1, 0],
      yaw_deg: 0,
      pitch_deg: 0,
      fov_deg: 65,
      information_gain: 0.15,
      priority_rank: 3,
      instruction: "Capture west corner",
    },
    {
      viewpoint_id: "v-high",
      position: [3.0, 4.0, 1.6],
      look_direction: [1, 0, 0],
      yaw_deg: 90,
      pitch_deg: 10,
      fov_deg: 65,
      information_gain: 0.85,
      priority_rank: 1,
      instruction: "Capture partition wall gap",
    },
    {
      viewpoint_id: "v-mid",
      position: [2.0, 2.5, 1.6],
      look_direction: [-1, 0, 0],
      yaw_deg: 270,
      pitch_deg: -5,
      fov_deg: 65,
      information_gain: 0.52,
      priority_rank: 2,
      instruction: "Capture window alcove",
    },
  ];

  const filtered = filterAndSortViewpoints(poses, 0.20, 2);
  assert.equal(filtered.length, 2);
  assert.equal(filtered[0].viewpoint_id, "v-high");
  assert.equal(filtered[1].viewpoint_id, "v-mid");

  const desc = describeCameraPose(filtered[0]);
  assert.ok(desc.includes("Position (3.0m, 4.0m, 1.6m)"));
  assert.ok(desc.includes("Facing E"));
  assert.ok(desc.includes("Tilt up 10°"));
});
