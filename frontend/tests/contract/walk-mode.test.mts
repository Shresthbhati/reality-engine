import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_EYE_HEIGHT,
  DEFAULT_BODY_RADIUS,
  DEFAULT_WALK_SPEED,
  MIN_PITCH_DEG,
  MAX_PITCH_DEG,
  placeEyeAtHeight,
  clampPitch,
  computeWalkDelta,
  distancePointToSegment,
  distancePointToBox,
  resolveWallCollision,
  classifyTouchRegion,
  computeTouchJoystick,
  type WallSegment,
  type WallBox,
} from "../../src/lib/viewport/walk-model.ts";

test("placeEyeAtHeight sets Y to floor + 1.65m and preserves XZ", () => {
  const p1 = placeEyeAtHeight(0.0);
  assert.equal(p1[0], 0.0);
  assert.equal(p1[1], DEFAULT_EYE_HEIGHT);
  assert.equal(p1[2], 0.0);

  const p2 = placeEyeAtHeight(2.5, 1.65, [3.2, 0, -4.5]);
  assert.equal(p2[0], 3.2);
  assert.equal(p2[1], 4.15);
  assert.equal(p2[2], -4.5);
});

test("clampPitch prevents camera inversion at +/- 75 degrees", () => {
  const maxRad = (MAX_PITCH_DEG * Math.PI) / 180;
  const minRad = (MIN_PITCH_DEG * Math.PI) / 180;

  assert.equal(clampPitch(0), 0);
  assert.equal(clampPitch(1.0), 1.0); // ~57 degrees, within bounds
  assert.equal(clampPitch(2.5), maxRad); // exceeds 75 deg, clamped
  assert.equal(clampPitch(-2.5), minRad); // exceeds -75 deg, clamped
});

test("computeWalkDelta produces metric horizontal translation respecting yaw", () => {
  const dt = 0.02; // 20ms frame
  const expectedStep = DEFAULT_WALK_SPEED * dt;

  // Yaw = 0: forward is -Z
  const [dxFwd, dzFwd] = computeWalkDelta(1, 0, 0, DEFAULT_WALK_SPEED, dt);
  assert.ok(Math.abs(dxFwd) < 1e-6);
  assert.ok(Math.abs(dzFwd - (-expectedStep)) < 1e-6);

  // Yaw = 0: strafe right is +X
  const [dxRight, dzRight] = computeWalkDelta(0, 1, 0, DEFAULT_WALK_SPEED, dt);
  assert.ok(Math.abs(dxRight - expectedStep) < 1e-6);
  assert.ok(Math.abs(dzRight) < 1e-6);

  // Yaw = 90 deg (PI/2): forward is -X
  const [dx90, dz90] = computeWalkDelta(1, 0, Math.PI / 2, DEFAULT_WALK_SPEED, dt);
  assert.ok(Math.abs(dx90 - (-expectedStep)) < 1e-5);
  assert.ok(Math.abs(dz90) < 1e-5);

  // Diagonal normalization: magnitude should not exceed expectedStep
  const [dxDiag, dzDiag] = computeWalkDelta(1, 1, 0, DEFAULT_WALK_SPEED, dt);
  const diagMag = Math.hypot(dxDiag, dzDiag);
  assert.ok(Math.abs(diagMag - expectedStep) < 1e-5);
});

test("distancePointToSegment calculates orthogonal and endpoint distances", () => {
  const p1: [number, number] = [0, 0];
  const p2: [number, number] = [10, 0]; // horizontal segment along X

  // Point on segment
  assert.equal(distancePointToSegment(5, 0, p1, p2), 0);

  // Point orthogonal to segment midpoint
  assert.equal(distancePointToSegment(5, 3, p1, p2), 3);

  // Point past p1 endpoint
  assert.equal(distancePointToSegment(-4, 0, p1, p2), 4);

  // Point past p2 endpoint
  assert.equal(distancePointToSegment(13, 4, p1, p2), 5); // 3-4-5 triangle
});

test("distancePointToBox computes correct distance to AABB obstacle", () => {
  const box: WallBox = { minX: 1, maxX: 3, minZ: 1, maxZ: 3 };

  // Point inside
  assert.equal(distancePointToBox(2, 2, box), 0);

  // Point outside along axis
  assert.equal(distancePointToBox(0, 2, box), 1);
  assert.equal(distancePointToBox(5, 2, box), 2);

  // Point outside corner
  assert.equal(distancePointToBox(0, 0, box), Math.hypot(1, 1));
});

test("resolveWallCollision stops at wall and slides smoothly along unobstructed axis", () => {
  // A vertical wall along Z axis at X = 2 (from Z=-5 to Z=5)
  const wall: WallSegment = { p1: [2, -5], p2: [2, 5] };
  const obstacles = [wall];

  // 1. Unobstructed movement in open space
  const unobs = resolveWallCollision(0, 0, 0.5, 0.5, obstacles, DEFAULT_BODY_RADIUS);
  assert.equal(unobs.collided, false);
  assert.equal(unobs.x, 0.5);
  assert.equal(unobs.z, 0.5);

  // 2. Direct forward collision into wall (trying to move from 1.5 to 2.2 along X, clearance radius 0.35)
  const blocked = resolveWallCollision(1.5, 0, 2.0, 0, obstacles, DEFAULT_BODY_RADIUS);
  assert.equal(blocked.collided, true);
  // Direct collision with no Z delta cannot slide along Z (desiredZ == currentZ), stays at current position
  assert.equal(blocked.x, 1.5);
  assert.equal(blocked.z, 0);

  // 3. Diagonal movement hitting wall slides along unobstructed Z axis
  const sliding = resolveWallCollision(1.5, 0, 1.9, 0.8, obstacles, DEFAULT_BODY_RADIUS);
  assert.equal(sliding.collided, true);
  // X is blocked because 1.9 is within radius 0.35 of wall X=2 (dist = 0.1 < 0.35)
  // But Z movement from 0 to 0.8 is open, so it slides along Z:
  assert.equal(sliding.x, 1.5);
  assert.equal(sliding.z, 0.8);
});

test("resolveWallCollision handles tight corner where both axes are blocked", () => {
  // Corner at X=2, Z=2
  const obstacles: WallSegment[] = [
    { p1: [2, 0], p2: [2, 5] }, // X = 2 wall
    { p1: [0, 2], p2: [5, 2] }, // Z = 2 wall
  ];

  // Current pos at (1.5, 1.5). Desired move into corner (1.9, 1.9)
  const res = resolveWallCollision(1.5, 1.5, 1.9, 1.9, obstacles, DEFAULT_BODY_RADIUS);
  assert.equal(res.collided, true);
  // Both sliding attempts (1.9, 1.5) and (1.5, 1.9) violate wall clearance of 0.35
  assert.equal(res.x, 1.5);
  assert.equal(res.z, 1.5);
});

test("classifyTouchRegion and computeTouchJoystick translate touch input", () => {
  assert.equal(classifyTouchRegion(100, 400), "move");
  assert.equal(classifyTouchRegion(300, 400), "look");

  // Deadzone check
  const deadzone = computeTouchJoystick(2, 2);
  assert.equal(deadzone.forward, 0);
  assert.equal(deadzone.strafe, 0);

  // Drag up (negative deltaY) moves forward
  const up = computeTouchJoystick(0, -30, 60);
  assert.ok(up.forward > 0.45);
  assert.equal(up.strafe, 0);

  // Drag right (positive deltaX) strafes right
  const right = computeTouchJoystick(45, 0, 60);
  assert.ok(right.strafe > 0.7);
  assert.equal(right.forward, 0);
});
