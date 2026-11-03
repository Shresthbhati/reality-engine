/**
 * Room Boundary Snapping & Collision Prevention Engine.
 * Pure geometric algorithms in metric space providing sub-millimeter precision
 * for wall alignment, collision clamping, and rotation snapping without WebGL/DOM dependencies.
 */

import type { WorldIR, Entity } from "@/types/worldir";
import type {
  Point2D,
  Point3D,
  RoomBoundary,
  SnapConfig,
  WallSnapResult,
  EntityBounds,
} from "./transform-types.ts";
import { DEFAULT_SNAP_CONFIG } from "./transform-types.ts";

/**
 * Computes shortest distance from 2D point P to line segment AB.
 */
export function distanceToSegment2D(
  p: Point2D,
  a: Point2D,
  b: Point2D
): {
  distance: number;
  projection: Point2D;
  normal: Point2D;
  t: number;
  segmentVector: Point2D;
  segmentLength: number;
} {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lenSq = dx * dx + dy * dy;
  const len = Math.sqrt(lenSq);

  if (len < 1e-6) {
    const dist = Math.hypot(p[0] - a[0], p[1] - a[1]);
    return {
      distance: dist,
      projection: [a[0], a[1]],
      normal: [0, 1],
      t: 0,
      segmentVector: [0, 0],
      segmentLength: 0,
    };
  }

  // Parameter t of projection onto line AB
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lenSq));
  const projX = a[0] + t * dx;
  const projY = a[1] + t * dy;
  const dist = Math.hypot(p[0] - projX, p[1] - projY);

  // Perpendicular normal (rotated 90 deg counter-clockwise)
  const nx = -dy / len;
  const ny = dx / len;

  return {
    distance: dist,
    projection: [projX, projY],
    normal: [nx, ny],
    t,
    segmentVector: [dx, dy],
    segmentLength: len,
  };
}

/**
 * Standard ray-casting point-in-polygon test.
 */
export function pointInPolygon2D(point: Point2D, polygon: Point2D[]): boolean {
  if (polygon.length < 3) return false;
  let inside = false;
  const [x, y] = point;
  const n = polygon.length;

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];

    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }

  return inside;
}

/**
 * Computes arithmetic centroid of 2D polygon.
 */
export function polygonCentroid2D(polygon: Point2D[]): Point2D {
  if (!polygon.length) return [0, 0];
  let sumX = 0;
  let sumY = 0;
  for (const pt of polygon) {
    sumX += pt[0];
    sumY += pt[1];
  }
  return [sumX / polygon.length, sumY / polygon.length];
}

/**
 * Snaps angle in radians to nearest multiple of incrementDeg.
 */
export function snapAngle(angleRad: number, incrementDeg = 15): number {
  if (incrementDeg <= 0) return angleRad;
  const incrementRad = (incrementDeg * Math.PI) / 180;
  return Math.round(angleRad / incrementRad) * incrementRad;
}

/**
 * Applies room boundary snapping and collision prevention to a candidate position and rotation.
 */
export function applyWallSnappingAndCollision(
  candidatePos: Point3D,
  candidateRot: Point3D,
  bounds: EntityBounds,
  room: RoomBoundary,
  config: SnapConfig = DEFAULT_SNAP_CONFIG
): WallSnapResult {
  const isYup = room.upAxis === "y";
  // In Y-up: plan coordinates are X and Z, elevation is Y.
  // In Z-up: plan coordinates are X and Y, elevation is Z.
  const planPos: Point2D = isYup ? [candidatePos[0], candidatePos[2]] : [candidatePos[0], candidatePos[1]];
  let elev = isYup ? candidatePos[1] : candidatePos[2];

  const poly = room.polygon2D;
  if (!poly || poly.length < 3) {
    return {
      snapped: false,
      originalPosition: [...candidatePos] as Point3D,
      snappedPosition: [...candidatePos] as Point3D,
      originalRotation: [...candidateRot] as Point3D,
      snappedRotation: [...candidateRot] as Point3D,
      wallSegmentIndex: null,
      wallNormal: null,
      distanceToWall: Infinity,
      clampedDueToCollision: false,
    };
  }

  const centroid = polygonCentroid2D(poly);
  let bestDist = Infinity;
  let bestSegIdx = -1;
  let bestProj: Point2D | null = null;
  let bestInwardNormal: Point2D | null = null;
  let bestSegVec: Point2D | null = null;

  // Half dimensions of the entity bounding box
  const halfDepth = Math.max(0.05, bounds.depth / 2);
  const halfHeight = Math.max(0.05, bounds.height / 2);

  // Check all segments of the room boundary polygon
  const numVerts = poly.length;
  for (let i = 0; i < numVerts; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % numVerts];

    const { distance, projection, normal, segmentVector } = distanceToSegment2D(planPos, a, b);

    // Orient normal towards room centroid (inward pointing)
    const midX = (a[0] + b[0]) / 2;
    const midY = (a[1] + b[1]) / 2;
    const toCentroidX = centroid[0] - midX;
    const toCentroidY = centroid[1] - midY;
    const dotNormal = normal[0] * toCentroidX + normal[1] * toCentroidY;
    const inwardNormal: Point2D = dotNormal >= 0 ? normal : [-normal[0], -normal[1]];

    if (distance < bestDist) {
      bestDist = distance;
      bestSegIdx = i;
      bestProj = projection;
      bestInwardNormal = inwardNormal;
      bestSegVec = segmentVector;
    }
  }

  let snapped = false;
  let clamped = false;
  let outPlan: Point2D = [planPos[0], planPos[1]];
  const outRot: Point3D = [...candidateRot] as Point3D;

  // 1. Wall Snapping
  if (
    config.enabled &&
    bestDist <= config.wallSnapThresholdM + halfDepth &&
    bestProj &&
    bestInwardNormal
  ) {
    // Snap entity so its rear edge is flush against the wall interior face
    outPlan = [
      bestProj[0] + bestInwardNormal[0] * halfDepth,
      bestProj[1] + bestInwardNormal[1] * halfDepth,
    ];
    snapped = true;

    // Align rotation to wall
    if (config.alignRotationToWall && bestSegVec) {
      // Wall segment angle
      const wallAngle = Math.atan2(bestSegVec[1], bestSegVec[0]);
      if (isYup) {
        // Yaw is around Y axis: align rotation
        outRot[1] = snapAngle(wallAngle, config.angleSnapIncrementDeg);
      } else {
        // Yaw is around Z axis
        outRot[2] = snapAngle(wallAngle, config.angleSnapIncrementDeg);
      }
    }
  } else if (config.enabled && config.gridSnapIncrementM > 0) {
    // Grid snapping when not snapped to a wall
    const inc = config.gridSnapIncrementM;
    outPlan[0] = Math.round(outPlan[0] / inc) * inc;
    outPlan[1] = Math.round(outPlan[1] / inc) * inc;
  }

  // 2. Collision Prevention (Keep strictly within room polygon interior)
  if (config.preventWallPenetration) {
    const isInside = pointInPolygon2D(outPlan, poly);
    if (!isInside && bestProj && bestInwardNormal) {
      // Clamped back inside the room along the wall normal
      const clearance = 0.02; // 2cm safety buffer from wall
      outPlan = [
        bestProj[0] + bestInwardNormal[0] * (halfDepth + clearance),
        bestProj[1] + bestInwardNormal[1] * (halfDepth + clearance),
      ];
      clamped = true;
    }
  }

  // 3. Vertical slab constraint (Floor and Ceiling)
  if (config.constrainToVerticalBounds) {
    const minElev = room.floorHeightM + halfHeight;
    const maxElev = room.ceilingHeightM - halfHeight;
    if (maxElev > minElev) {
      if (elev < minElev) {
        elev = minElev;
        clamped = true;
      } else if (elev > maxElev) {
        elev = maxElev;
        clamped = true;
      }
    }
  }

  // Assemble 3D snapped position
  const snappedPos: Point3D = isYup ? [outPlan[0], elev, outPlan[1]] : [outPlan[0], outPlan[1], elev];

  const wallNormal3D: Point3D | null = bestInwardNormal
    ? isYup
      ? [bestInwardNormal[0], 0, bestInwardNormal[1]]
      : [bestInwardNormal[0], bestInwardNormal[1], 0]
    : null;

  return {
    snapped,
    originalPosition: [...candidatePos] as Point3D,
    snappedPosition: snappedPos,
    originalRotation: [...candidateRot] as Point3D,
    snappedRotation: outRot,
    wallSegmentIndex: bestSegIdx >= 0 ? bestSegIdx : null,
    wallNormal: wallNormal3D,
    distanceToWall: bestDist,
    clampedDueToCollision: clamped,
  };
}

/**
 * Extracts RoomBoundary descriptors from WorldIR.
 */
export function extractRoomBoundariesFromWorld(world: WorldIR): RoomBoundary[] {
  if (!world || !world.entities) return [];

  const boundaries: RoomBoundary[] = [];
  const upAxis: "y" | "z" = (world.coordinate_system?.toLowerCase().includes("z-up") || (world.metadata as Record<string, unknown>)?.up_axis === "z") ? "z" : "y";

  for (const [id, entity] of Object.entries(world.entities)) {
    if (entity.type !== "room" && entity.type !== "space") continue;

    const cp = (entity.custom_properties || {}) as Record<string, unknown>;
    const name = entity.name || id;

    // Check boundary_polygon_2d, footprint_polygon_xyz_m, or boundary_polygon_3d
    let poly2D: Point2D[] = [];
    const poly3D: Point3D[] = [];

    if (Array.isArray(cp.boundary_polygon_2d) && cp.boundary_polygon_2d.length >= 3) {
      poly2D = (cp.boundary_polygon_2d as unknown[][]).map(
        (pt) => [Number(pt[0]) || 0, Number(pt[1]) || 0] as Point2D
      );
      for (const p of poly2D) {
        poly3D.push(upAxis === "y" ? [p[0], Number(cp.floor_height_m) || 0, p[1]] : [p[0], p[1], Number(cp.floor_height_m) || 0]);
      }
    } else if (Array.isArray(cp.footprint_polygon_xyz_m) && cp.footprint_polygon_xyz_m.length >= 3) {
      for (const pt of cp.footprint_polygon_xyz_m as unknown[][]) {
        const x = Number(pt[0]) || 0;
        const y = Number(pt[1]) || 0;
        const z = Number(pt[2]) || 0;
        poly3D.push([x, y, z]);
        poly2D.push(upAxis === "y" ? [x, z] : [x, y]);
      }
    } else if (Array.isArray(cp.boundary_polygon_3d) && cp.boundary_polygon_3d.length >= 3) {
      for (const pt of cp.boundary_polygon_3d as unknown[][]) {
        const x = Number(pt[0]) || 0;
        const y = Number(pt[1]) || 0;
        const z = Number(pt[2]) || 0;
        poly3D.push([x, y, z]);
        poly2D.push(upAxis === "y" ? [x, z] : [x, y]);
      }
    }

    const floorHeightM = typeof cp.floor_height_m === "number" ? cp.floor_height_m : 0;
    const ceilingHeightM = typeof cp.ceiling_height_min_m === "number"
      ? cp.ceiling_height_min_m
      : typeof cp.ceiling_height_m === "number"
      ? cp.ceiling_height_m
      : floorHeightM + 2.6;

    if (poly2D.length >= 3) {
      boundaries.push({
        roomId: id,
        name,
        polygon3D: poly3D,
        polygon2D: poly2D,
        floorHeightM,
        ceilingHeightM,
        upAxis,
      });
    }
  }

  return boundaries;
}
