import { test } from "node:test";
import assert from "node:assert/strict";
import {
  interpolateTourPose,
  detectActiveRoom,
  formatTourTime,
  shortestAngleDiff,
  interpolateAngle,
  findClosestBookmark,
} from "../../src/lib/viewport/tour-model.ts";
import type { ArchitecturalTour, RoomWaypoint, SpatialBookmark } from "../../src/lib/api/navigation.ts";

test("formatTourTime formats seconds into M:SS standard time code", () => {
  assert.equal(formatTourTime(0), "0:00");
  assert.equal(formatTourTime(5), "0:05");
  assert.equal(formatTourTime(59), "0:59");
  assert.equal(formatTourTime(60), "1:00");
  assert.equal(formatTourTime(65), "1:05");
  assert.equal(formatTourTime(135), "2:15");
  assert.equal(formatTourTime(-10), "0:00");
});

test("shortestAngleDiff calculates minimal arc difference across +/- PI boundary", () => {
  // Direct difference
  assert.ok(Math.abs(shortestAngleDiff(0.5, 0.2) - 0.3) < 1e-6);

  // Across +/- PI boundary: 3.10 rad and -3.10 rad are only 0.083 rad apart
  const piDiff = shortestAngleDiff(-3.10, 3.10);
  assert.ok(Math.abs(piDiff - (2 * Math.PI - 6.20)) < 1e-4);

  // Reversing direction
  const piDiffRev = shortestAngleDiff(3.10, -3.10);
  assert.ok(Math.abs(piDiffRev - -(2 * Math.PI - 6.20)) < 1e-4);
});

test("interpolateAngle smoothly wraps around +/- PI boundary", () => {
  const fromAngle = 3.10;
  const toAngle = -3.10;
  const mid = interpolateAngle(fromAngle, toAngle, 0.5);

  // Midpoint across +/- PI boundary should be ~ +/- PI
  assert.ok(Math.abs(Math.abs(mid) - Math.PI) < 0.05);
});

test("detectActiveRoom accurately matches 3D position to RoomWaypoint AABB with margin", () => {
  const rooms: RoomWaypoint[] = [
    {
      room_id: "room-living",
      name: "Living Room",
      room_type: "living",
      center: [2, 1.65, 3],
      bounds_min: [0, 0, 1],
      bounds_max: [4, 3, 5],
      floor_elevation_m: 0,
      ceiling_height_m: 3,
      area_sqm: 16,
      openings: [],
    },
    {
      room_id: "room-kitchen",
      name: "Kitchen",
      room_type: "kitchen",
      center: [6, 1.65, 3],
      bounds_min: [4, 0, 1],
      bounds_max: [8, 3, 5],
      floor_elevation_m: 0,
      ceiling_height_m: 3,
      area_sqm: 16,
      openings: [],
    },
  ];

  // Inside Living Room
  const livingMatch = detectActiveRoom([2, 1.65, 3], rooms);
  assert.equal(livingMatch?.room_id, "room-living");

  // Inside Kitchen
  const kitchenMatch = detectActiveRoom([6.5, 1.65, 2.5], rooms);
  assert.equal(kitchenMatch?.room_id, "room-kitchen");

  // In boundary margin (x = -0.1 is within 0.2m margin of Living Room)
  const marginMatch = detectActiveRoom([-0.1, 1.65, 3], rooms);
  assert.equal(marginMatch?.room_id, "room-living");

  // Far away outside
  const farMatch = detectActiveRoom([50, 1.65, 50], rooms);
  assert.equal(farMatch, null);

  // Empty rooms list fallback
  assert.equal(detectActiveRoom([0, 1.65, 0], []), null);
});

test("findClosestBookmark detects nearest spatial bookmark within radius", () => {
  const bookmarks: SpatialBookmark[] = [
    {
      bookmark_id: "bmk-1",
      name: "Balcony View",
      room_id: "room-balcony",
      room_name: "Balcony",
      position: [0, 1.65, 10],
      yaw_rad: 0,
      pitch_rad: 0,
      fov_deg: 60,
      tags: [],
      created_at: "2026-11-08T12:00:00Z",
    },
    {
      bookmark_id: "bmk-2",
      name: "Kitchen Island",
      room_id: "room-kitchen",
      room_name: "Kitchen",
      position: [5, 1.65, 2],
      yaw_rad: 1.57,
      pitch_rad: 0,
      fov_deg: 60,
      tags: [],
      created_at: "2026-11-08T12:00:00Z",
    },
  ];

  // Close to Kitchen Island
  const closest = findClosestBookmark([4.8, 1.65, 2.1], bookmarks, 1.0);
  assert.equal(closest?.bookmark_id, "bmk-2");

  // Too far from any bookmark (threshold 0.1m)
  const none = findClosestBookmark([4.8, 1.65, 2.1], bookmarks, 0.1);
  assert.equal(none, null);
});

test("interpolateTourPose evaluates architectural tour timeline", () => {
  const sampleTour: ArchitecturalTour = {
    tour_id: "tour-sample",
    title: "Apartment Tour",
    description: "Sample walkthrough",
    total_duration_seconds: 20,
    rooms_visited: ["Entrance Hall", "Living Room", "Master Bedroom"],
    waypoint_path: [[0, 1.65, 0], [4, 1.65, 6], [8, 1.65, 12]],
    keyframes: [
      {
        keyframe_id: "kf-1",
        room_id: "room-entry",
        room_name: "Entrance Hall",
        position: [0, 1.65, 0],
        yaw_rad: 0,
        pitch_rad: 0,
        dwell_seconds: 3,
        transition_seconds: 0,
        narration: "Welcome to the entrance foyer.",
        kind: "entry",
      },
      {
        keyframe_id: "kf-2",
        room_id: "room-living",
        room_name: "Living Room",
        position: [4, 1.65, 6],
        yaw_rad: 1.2,
        pitch_rad: -0.1,
        dwell_seconds: 4,
        transition_seconds: 5,
        narration: "Here is the spacious living area.",
        kind: "panorama",
      },
      {
        keyframe_id: "kf-3",
        room_id: "room-bed",
        room_name: "Master Bedroom",
        position: [8, 1.65, 12],
        yaw_rad: 2.0,
        pitch_rad: 0,
        dwell_seconds: 2,
        transition_seconds: 6,
        narration: "Final stop: Master bedroom.",
        kind: "panorama",
      },
    ],
  };

  // Test at t=0: exactly first keyframe
  const pose0 = interpolateTourPose(sampleTour, 0);
  assert.equal(pose0.currentRoom, "Entrance Hall");
  assert.equal(pose0.position[0], 0);
  assert.equal(pose0.position[1], 1.65);
  assert.equal(pose0.position[2], 0);
  assert.equal(pose0.progressPct, 0);

  // Test during dwell period (t=2s, dwell is 3s): stationary at Entrance Hall
  const poseDwell = interpolateTourPose(sampleTour, 2);
  assert.equal(poseDwell.currentRoom, "Entrance Hall");
  assert.equal(poseDwell.position[0], 0);
  assert.equal(poseDwell.position[2], 0);
  assert.equal(poseDwell.narration, "Welcome to the entrance foyer.");

  // Test mid-transition between Entrance and Living (t=5.5s: 3s dwell + 2.5s into 5s transit)
  const poseMidTransit = interpolateTourPose(sampleTour, 5.5);
  // Hermite curve at u=0.5 is exactly 0.5 for position
  assert.ok(Math.abs(poseMidTransit.position[0] - 2.0) < 0.1);
  assert.ok(Math.abs(poseMidTransit.position[2] - 3.0) < 0.1);
  assert.ok(poseMidTransit.yawRad > 0 && poseMidTransit.yawRad < 1.2);

  // Test at second keyframe dwell (t=10s: 3s + 5s transit + 2s into 4s dwell): stationary at Living Room
  const poseLivingDwell = interpolateTourPose(sampleTour, 10);
  assert.equal(poseLivingDwell.currentRoom, "Living Room");
  assert.equal(poseLivingDwell.position[0], 4);
  assert.equal(poseLivingDwell.position[2], 6);
  assert.equal(poseLivingDwell.narration, "Here is the spacious living area.");

  // Test past end of tour (t=25s): clamped to final keyframe
  const poseEnd = interpolateTourPose(sampleTour, 25);
  assert.equal(poseEnd.currentRoom, "Master Bedroom");
  assert.equal(poseEnd.position[0], 8);
  assert.equal(poseEnd.position[2], 12);
  assert.equal(poseEnd.progressPct, 100);

  // Test empty tour fallback
  const emptyTour: ArchitecturalTour = {
    tour_id: "empty",
    title: "",
    description: "",
    total_duration_seconds: 0,
    keyframes: [],
    waypoint_path: [],
    rooms_visited: [],
  };
  const fallbackPose = interpolateTourPose(emptyTour, 5);
  assert.equal(fallbackPose.position[1], 1.65);
  assert.equal(fallbackPose.progressPct, 0);
});
