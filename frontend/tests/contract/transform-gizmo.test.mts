import test from "node:test";
import assert from "node:assert/strict";
import {
  distanceToSegment2D,
  pointInPolygon2D,
  polygonCentroid2D,
  snapAngle,
  applyWallSnappingAndCollision,
  extractRoomBoundariesFromWorld,
} from "../../src/lib/studio/room-boundary-snap.ts";
import type {
  RoomBoundary,
  EntityBounds,
  SnapConfig,
  Point3D,
} from "../../src/lib/studio/transform-types.ts";
import { DEFAULT_SNAP_CONFIG } from "../../src/lib/studio/transform-types.ts";
import { HistoryStack } from "../../src/lib/viewport/history-stack.ts";
import type { WorldIR } from "../../src/types/worldir.ts";

test("distanceToSegment2D computes accurate distance, projection, and normal", () => {
  // Horizontal segment from (0, 0) to (10, 0)
  const a: [number, number] = [0, 0];
  const b: [number, number] = [10, 0];

  // Point above midpoint
  const p1: [number, number] = [5, 2];
  const res1 = distanceToSegment2D(p1, a, b);
  assert.equal(Math.abs(res1.distance - 2) < 1e-5, true);
  assert.deepEqual(res1.projection, [5, 0]);
  assert.equal(res1.t, 0.5);

  // Point beyond endpoint B
  const p2: [number, number] = [12, 0];
  const res2 = distanceToSegment2D(p2, a, b);
  assert.equal(Math.abs(res2.distance - 2) < 1e-5, true);
  assert.deepEqual(res2.projection, [10, 0]);
  assert.equal(res2.t, 1.0);
});

test("pointInPolygon2D accurately detects interior and exterior points for non-Manhattan room", () => {
  // Irregular pentagonal room
  const poly: [number, number][] = [
    [0, 0],
    [5, 0],
    [7, 3], // Angled non-Manhattan wall
    [4, 6],
    [0, 6],
  ];

  assert.equal(pointInPolygon2D([2, 2], poly), true, "Center should be inside");
  assert.equal(pointInPolygon2D([5.5, 2], poly), true, "Near angled wall should be inside");
  assert.equal(pointInPolygon2D([8, 3], poly), false, "Outside angled wall should be outside");
  assert.equal(pointInPolygon2D([-1, 2], poly), false, "Outside left wall should be outside");
  assert.equal(pointInPolygon2D([2, 7], poly), false, "Outside top wall should be outside");
});

test("applyWallSnappingAndCollision snaps entity flush to wall within threshold", () => {
  const room: RoomBoundary = {
    roomId: "room-1",
    name: "Living Room",
    polygon3D: [
      [0, 0, 0],
      [6, 0, 0],
      [6, 0, 5],
      [0, 0, 5],
    ],
    polygon2D: [
      [0, 0],
      [6, 0],
      [6, 5],
      [0, 5],
    ],
    floorHeightM: 0,
    ceilingHeightM: 2.8,
    upAxis: "y",
  };

  const bounds: EntityBounds = { width: 1.0, height: 0.8, depth: 0.6 }; // halfDepth = 0.3
  const config: SnapConfig = {
    ...DEFAULT_SNAP_CONFIG,
    wallSnapThresholdM: 0.20,
    gridSnapIncrementM: 0,
  };

  // 1. Candidate position 0.38m from wall at z=0 (depth/2 = 0.3, distance = 0.08m <= 0.20m)
  const candidatePos: Point3D = [3.0, 0.4, 0.38];
  const candidateRot: Point3D = [0, 0, 0];

  const result = applyWallSnappingAndCollision(candidatePos, candidateRot, bounds, room, config);
  assert.equal(result.snapped, true, "Should snap to wall");
  assert.equal(result.wallSegmentIndex, 0, "Should identify segment 0 (wall along z=0)");
  // In Y-up: snapped position Z should be flush with wall face + halfDepth = 0 + 0.3 = 0.3
  assert.equal(Math.abs(result.snappedPosition[2] - 0.3) < 1e-4, true);
  assert.equal(Math.abs(result.snappedPosition[0] - 3.0) < 1e-4, true);

  // 2. Candidate position 1.5m away from wall: should NOT snap
  const farPos: Point3D = [3.0, 0.4, 1.5];
  const farResult = applyWallSnappingAndCollision(farPos, candidateRot, bounds, room, config);
  assert.equal(farResult.snapped, false, "Should not snap when outside threshold");
  assert.deepEqual(farResult.snappedPosition, farPos);
});

test("applyWallSnappingAndCollision aligns yaw rotation to snapped wall", () => {
  // Room with 45-degree angled wall from (0,0) to (4,4)
  const room: RoomBoundary = {
    roomId: "room-angled",
    polygon3D: [
      [0, 0, 0],
      [4, 0, 4],
      [4, 0, 6],
      [0, 0, 6],
    ],
    polygon2D: [
      [0, 0],
      [4, 4],
      [4, 6],
      [0, 6],
    ],
    floorHeightM: 0,
    ceilingHeightM: 3.0,
    upAxis: "y",
  };

  const bounds: EntityBounds = { width: 0.8, height: 0.8, depth: 0.4 };
  const config: SnapConfig = {
    ...DEFAULT_SNAP_CONFIG,
    wallSnapThresholdM: 0.30,
    alignRotationToWall: true,
    angleSnapIncrementDeg: 15,
  };

  // Position near angled segment (1.0, 1.1)
  const pos: Point3D = [1.0, 0.4, 1.1];
  const rot: Point3D = [0, 0, 0];

  const res = applyWallSnappingAndCollision(pos, rot, bounds, room, config);
  assert.equal(res.snapped, true);
  assert.ok(res.snappedRotation, "Snapped rotation should exist");
  // 45 degrees = PI / 4 = ~0.785398 rad
  const expectedYaw = Math.PI / 4;
  assert.equal(Math.abs(res.snappedRotation[1] - expectedYaw) < 0.05, true, "Yaw should align to wall angle 45 deg");
});

test("applyWallSnappingAndCollision prevents collision and penetration through walls", () => {
  const room: RoomBoundary = {
    roomId: "room-box",
    polygon3D: [
      [0, 0, 0],
      [5, 0, 0],
      [5, 0, 5],
      [0, 0, 5],
    ],
    polygon2D: [
      [0, 0],
      [5, 0],
      [5, 5],
      [0, 5],
    ],
    floorHeightM: 0,
    ceilingHeightM: 2.5,
    upAxis: "y",
  };

  const bounds: EntityBounds = { width: 1.0, height: 1.0, depth: 1.0 }; // half = 0.5
  const config: SnapConfig = {
    ...DEFAULT_SNAP_CONFIG,
    enabled: true,
    preventWallPenetration: true,
    constrainToVerticalBounds: true,
  };

  // 1. Entity position outside wall (x = -2.0, far outside room)
  const outsidePos: Point3D = [-2.0, 1.0, 2.5];
  const res = applyWallSnappingAndCollision(outsidePos, [0, 0, 0], bounds, room, config);
  assert.equal(res.clampedDueToCollision, true, "Should trigger collision clamp");
  // Clamped X must be inside room (> 0)
  assert.equal(res.snappedPosition[0] >= 0.5, true, "Entity must be clamped inside room interior");

  // 2. Vertical penetration below floor (y = -1.0)
  const belowFloorPos: Point3D = [2.5, -1.0, 2.5];
  const resY = applyWallSnappingAndCollision(belowFloorPos, [0, 0, 0], bounds, room, config);
  assert.equal(resY.clampedDueToCollision, true);
  // halfHeight = 0.5, floor = 0 => clamped elevation must be >= 0.5
  assert.equal(resY.snappedPosition[1] >= 0.5, true);

  // 3. Vertical penetration above ceiling (y = 5.0)
  const aboveCeilPos: Point3D = [2.5, 5.0, 2.5];
  const resCeil = applyWallSnappingAndCollision(aboveCeilPos, [0, 0, 0], bounds, room, config);
  assert.equal(resCeil.clampedDueToCollision, true);
  // ceiling = 2.5, halfHeight = 0.5 => clamped elevation <= 2.0
  assert.equal(resCeil.snappedPosition[1] <= 2.0, true);
});

test("snapAngle rounds angles to discrete angular increments", () => {
  // 15 deg increment = ~0.2618 rad
  const deg17Rad = (17 * Math.PI) / 180;
  const snapped15 = snapAngle(deg17Rad, 15);
  const expected15Rad = (15 * Math.PI) / 180;
  assert.equal(Math.abs(snapped15 - expected15Rad) < 1e-4, true);

  // 45 deg increment = ~0.7854 rad
  const deg88Rad = (88 * Math.PI) / 180;
  const snapped90 = snapAngle(deg88Rad, 45);
  const expected90Rad = (90 * Math.PI) / 180;
  assert.equal(Math.abs(snapped90 - expected90Rad) < 1e-4, true);
});

test("extractRoomBoundariesFromWorld extracts boundaries from WorldIR", () => {
  const world: WorldIR = {
    id: "world-1",
    coordinate_system: "Y-up",
    entities: {
      "room-living": {
        id: "room-living",
        type: "room",
        name: "Main Salon",
        custom_properties: {
          boundary_polygon_2d: [
            [0, 0],
            [6, 0],
            [6, 4],
            [0, 4],
          ],
          floor_height_m: 0.0,
          ceiling_height_min_m: 2.8,
        },
      },
      "table-1": {
        id: "table-1",
        type: "object",
        name: "Dining Table",
      },
    },
  };

  const boundaries = extractRoomBoundariesFromWorld(world);
  assert.equal(boundaries.length, 1);
  assert.equal(boundaries[0].roomId, "room-living");
  assert.equal(boundaries[0].polygon2D.length, 4);
  assert.equal(boundaries[0].floorHeightM, 0.0);
  assert.equal(boundaries[0].ceilingHeightM, 2.8);
  assert.equal(boundaries[0].upAxis, "y");
});

test("HistoryStack provides 100% reversible undo/redo for sequential entity transforms", () => {
  const history = new HistoryStack(50);
  let entityPos: Point3D = [0, 0, 0];
  let entityRot: Point3D = [0, 0, 0];

  // Action 1: Translate entity to [1, 0, 2]
  history.push({
    type: "transform",
    description: "Translate table to (1, 0, 2)",
    entityId: "table-1",
    before: { position: [0, 0, 0], rotation: [0, 0, 0] },
    after: { position: [1, 0, 2], rotation: [0, 0, 0] },
    apply: (state: { position: Point3D; rotation: Point3D }) => {
      entityPos = [...state.position] as Point3D;
      entityRot = [...state.rotation] as Point3D;
    },
  });
  entityPos = [1, 0, 2];

  // Action 2: Rotate entity by 45 degrees
  history.push({
    type: "transform",
    description: "Rotate table 45 deg",
    entityId: "table-1",
    before: { position: [1, 0, 2], rotation: [0, 0, 0] },
    after: { position: [1, 0, 2], rotation: [0, Math.PI / 4, 0] },
    apply: (state: { position: Point3D; rotation: Point3D }) => {
      entityPos = [...state.position] as Point3D;
      entityRot = [...state.rotation] as Point3D;
    },
  });
  entityRot = [0, Math.PI / 4, 0];

  assert.equal(history.canUndo(), true);
  assert.equal(history.getUndoCount(), 2);

  // Undo Action 2 -> restores rotation to 0
  history.undo();
  assert.deepEqual(entityRot, [0, 0, 0]);
  assert.deepEqual(entityPos, [1, 0, 2]);

  // Undo Action 1 -> restores position to [0, 0, 0]
  history.undo();
  assert.deepEqual(entityPos, [0, 0, 0]);
  assert.deepEqual(entityRot, [0, 0, 0]);

  // Redo Action 1 -> reapplies position to [1, 0, 2]
  history.redo();
  assert.deepEqual(entityPos, [1, 0, 2]);

  // Redo Action 2 -> reapplies rotation to [0, PI/4, 0]
  history.redo();
  assert.deepEqual(entityRot, [0, Math.PI / 4, 0]);
});
