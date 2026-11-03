import test from "node:test";
import assert from "node:assert/strict";
import {
  formatExpectedGainPct,
  filterRequestsByStatus,
  sortRequestsByGain,
  describeSuggestedPose,
  groupRequestsByRegionType,
} from "../../src/lib/studio/capture-requests-model.ts";
import type { CaptureRequestItem } from "../../src/lib/api/capture-requests.ts";

const SAMPLE_REQUESTS: CaptureRequestItem[] = [
  {
    id: "cr-01",
    world_id: "wld-1",
    expected_gain: 0.25,
    instruction: "Capture south wall from 2.5m away",
    status: "open",
    target_region: { region_type: "wall" },
    suggested_poses: [
      { distance_m: 2.5, pitch_deg: 0, yaw_deg: 180 },
    ],
  },
  {
    id: "cr-02",
    world_id: "wld-1",
    expected_gain: 0.10,
    instruction: "Photograph unobserved floor under table",
    status: "satisfied",
    target_region: { region_type: "floor" },
    suggested_poses: [
      { distance_m: 1.2, pitch_deg: -30, yaw_deg: 90 },
    ],
  },
  {
    id: "cr-03",
    world_id: "wld-1",
    expected_gain: 0.40,
    instruction: "Capture occluded bookcase interior",
    status: "open",
    target_region: { region_type: "furniture" },
    suggested_poses: [
      { description: "Stand 1.5m opposite bookcase" },
    ],
  },
];

test("formatExpectedGainPct formats clamped integer percentages", () => {
  assert.equal(formatExpectedGainPct(0.254), "+25% coverage");
  assert.equal(formatExpectedGainPct(0.0), "+0% coverage");
  assert.equal(formatExpectedGainPct(1.5), "+100% coverage");
  assert.equal(formatExpectedGainPct(-0.5), "+0% coverage");
});

test("filterRequestsByStatus isolates status subsets strictly", () => {
  const openReqs = filterRequestsByStatus(SAMPLE_REQUESTS, "open");
  assert.equal(openReqs.length, 2);
  assert.ok(openReqs.every((r) => r.status === "open"));

  const satisfiedReqs = filterRequestsByStatus(SAMPLE_REQUESTS, "satisfied");
  assert.equal(satisfiedReqs.length, 1);
  assert.equal(satisfiedReqs[0].id, "cr-02");

  const dismissedReqs = filterRequestsByStatus(SAMPLE_REQUESTS, "dismissed");
  assert.equal(dismissedReqs.length, 0);
});

test("sortRequestsByGain orders strictly descending by information gain", () => {
  const sorted = sortRequestsByGain(SAMPLE_REQUESTS);
  assert.equal(sorted[0].id, "cr-03"); // 0.40
  assert.equal(sorted[1].id, "cr-01"); // 0.25
  assert.equal(sorted[2].id, "cr-02"); // 0.10
});

test("describeSuggestedPose produces clean human guidance", () => {
  // Uses custom description when present
  const desc1 = describeSuggestedPose({ description: "Step back 2 meters" });
  assert.equal(desc1, "Step back 2 meters");

  // Formats distance, level pitch, and heading
  const desc2 = describeSuggestedPose({ distance_m: 2.5, pitch_deg: 0, yaw_deg: 180 });
  assert.equal(desc2, "Suggested viewpoint: at 2.5m distance, level, heading 180°");

  // Formats pitch down
  const desc3 = describeSuggestedPose({ distance_m: 1.0, pitch_deg: -25 });
  assert.equal(desc3, "Suggested viewpoint: at 1.0m distance, 25° pitch down");

  // Fallback
  assert.equal(describeSuggestedPose({}), "Point camera towards unobserved region");
});

test("groupRequestsByRegionType organizes requests into region buckets", () => {
  const groups = groupRequestsByRegionType(SAMPLE_REQUESTS);
  assert.equal(groups["wall"].length, 1);
  assert.equal(groups["floor"].length, 1);
  assert.equal(groups["furniture"].length, 1);
});
