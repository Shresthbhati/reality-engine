/**
 * Realistic window and portal visibility (N-31).
 *
 * Openings are spatial connections, not just transparent faces. A window or
 * door should show the correct exterior or adjacent space from the viewer's
 * viewpoint, with correct alignment and occlusion.
 *
 * This module computes:
 *   1. Which portals (windows/doors/openings) are visible from a camera.
 *   2. The view-through geometry: what is visible through each portal.
 *   3. Occlusion: whether another wall/building blocks the view through.
 *
 * Honesty rules:
 *   * A portal is only shown if there is actual geometry on the other side.
 *   * If no exterior geometry is known, the portal shows "unknown" (not a
 *     fake backdrop).
 *   * Occlusion is computed from real geometry; portals are not shown through
 *     solid walls.
 */

import type { Entity, Geometry } from "@/types/worldir";

/** One portal (window/door/opening) with its view-through info. */
export interface Portal {
  id: string;
  entityId: string;
  kind: "door" | "window" | "opening";
  /** Portal centre in world space. */
  centre: [number, number, number];
  /** Portal facing direction (outward normal). */
  facing: [number, number, number];
  /** Portal bounds in world space [minX, minY, minZ, maxX, maxY, maxZ]. */
  portalBounds: [number, number, number, number, number, number];
  /** Whether this portal is visible from the camera. */
  visible: boolean;
  /** What is visible through this portal. */
  viewThrough: PortalView | null;
}

/** What is visible through a portal. */
export interface PortalView {
  /** The entity on the other side (the exterior/adjacent space). */
  targetEntityId: string | null;
  /** Target entity type. */
  targetEntityType: Entity["type"] | null;
  /** Visible fraction of the portal (0..1) after occlusion. */
  visibleFraction: number;
  /** Distance to the portal in metres. */
  distance: number;
  /** Reason for the visibility result. */
  reason: string;
}

/** Input context for portal visibility computation. */
export interface PortalVisibilityContext {
  /** All entities in the world. */
  entities: Entity[];
  /** Entity geometries by id. */
  geometries: Map<string, Geometry>;
  /** Camera position in world space. */
  cameraPosition: [number, number, number];
  /** Camera look direction (normalized). */
  cameraDirection: [number, number, number];
  /** Camera field of view (horizontal, degrees). */
  fovDeg: number;
  /** Near/far clipping planes. */
  near: number;
  far: number;
}

/**
 * Compute portal visibility for all portals in the world from a camera.
 */
export function computePortalVisibility(ctx: PortalVisibilityContext): Portal[] {
  const portals: Portal[] = [];

  for (const ent of ctx.entities) {
    if (!isPortal(ent)) continue;

    const portal = evaluatePortalVisibility(ent, ctx.geometries, ctx);
    if (portal) portals.push(portal);
  }

  return portals;
}

/**
 * Evaluate portal visibility for a single entity.
 */
function evaluatePortalVisibility(
  ent: Entity,
  geometries: Map<string, Geometry>,
  ctx: PortalVisibilityContext,
): Portal | null {
  const cp = ent.custom_properties;

  // Get portal bounds from custom properties or geometry.
  const portalBounds = cp?.["portal_bounds"] as [number, number, number, number, number, number] | undefined;
  const centre = cp?.["centre"] as [number, number, number] | undefined;
  const facing = cp?.["facing"] as [number, number, number] | undefined;

  if (!centre || !facing) return null;

  // Compute portal world bounds from entity geometry if portal bounds not stored.
  let pbounds = portalBounds;
  if (!pbounds) {
    const geomBounds = entityGeometryBounds(ent, ctx.geometries);
    if (!geomBounds) return null;
    pbounds = geomBounds;
  }

  // Compute distance from camera to portal centre.
  const distance = distanceToPoint(ctx.cameraPosition, centre);

  // Distance culling.
  if (distance < ctx.near || distance > ctx.far) return null;

  // View frustum culling (simplified: dot product test).
  const toPortal: [number, number, number] = [
    centre[0] - ctx.cameraPosition[0],
    centre[1] - ctx.cameraPosition[1],
    centre[2] - ctx.cameraPosition[2],
  ];
  const dist = Math.sqrt(toPortal[0] ** 2 + toPortal[1] ** 2 + toPortal[2] ** 2);
  if (dist < 1e-6) return null;
  const toPortalNorm: [number, number, number] = [
    toPortal[0] / dist,
    toPortal[1] / dist,
    toPortal[2] / dist,
  ];
  const facingDot = dot(toPortalNorm, facing);
  // Portal faces away from camera.
  if (facingDot > 0) return null;

  // FOV culling.
  const fovCos = Math.cos((ctx.fovDeg * Math.PI / 180) / 2);
  if (facingDot < fovCos) return null;

  // Occlusion test: is there geometry between camera and portal?
  const occluded = isOccluded(ctx.cameraPosition, centre, ctx.entities, ctx.geometries, ent.id);
  const visible = !occluded;

  // Find what is visible through the portal.
  const viewThrough = findViewThrough(centre, facing, ctx);

  return {
    id: ent.id,
    entityId: ent.id,
    kind: (cp?.["portal_kind"] as "door" | "window" | "opening") ?? "opening",
    centre,
    facing,
    portalBounds: pbounds,
    visible,
    viewThrough,
  };
}

/**
 * Find what is visible through a portal: cast a ray through the portal centre
 * and find the first entity hit on the other side.
 */
function findViewThrough(
  portalCentre: [number, number, number],
  portalFacing: [number, number, number],
  ctx: PortalVisibilityContext,
): PortalView | null {
  // Cast a ray from the portal centre along the facing direction.
  const rayDir: [number, number, number] = [
    portalFacing[0],
    portalFacing[1],
    portalFacing[2],
  ];
  const rayLen = Math.sqrt(rayDir[0] ** 2 + rayDir[1] ** 2 + rayDir[2] ** 2);
  if (rayLen < 1e-6) return null;
  const rayDirNorm: [number, number, number] = [
    rayDir[0] / rayLen,
    rayDir[1] / rayLen,
    rayDir[2] / rayLen,
  ];

  // Step along the ray and test intersection with entities.
  const step = 0.25; // metres
  const maxDist = ctx.far;
  let dist = 0;

  while (dist < maxDist) {
    const testPoint: [number, number, number] = [
      portalCentre[0] + rayDirNorm[0] * dist,
      portalCentre[1] + rayDirNorm[1] * dist,
      portalCentre[2] + rayDirNorm[2] * dist,
    ];

    const hit = findEntityAtPoint(testPoint, ctx.entities, ctx.geometries);
    if (hit) {
      const [hitEnt, hitDist] = hit;
      return {
        targetEntityId: hitEnt.id,
        targetEntityType: hitEnt.type,
        visibleFraction: 1.0,
        distance: hitDist,
        reason: `Views ${hitEnt.type} "${hitEnt.name ?? hitEnt.id}" through portal.`,
      };
    }

    dist += step;
  }

  // Nothing found within range.
  return {
    targetEntityId: null,
    targetEntityType: null,
    visibleFraction: 0,
    distance: maxDist,
    reason: "No geometry visible through portal (empty or out of range).",
  };
}

/**
 * Find the first entity whose bounds contain the given point.
 */
function findEntityAtPoint(
  point: [number, number, number],
  entities: Entity[],
  geometries: Map<string, Geometry>,
): [Entity, number] | null {
  let closest: [Entity, number] | null = null;
  let closestDist = Infinity;

  for (const ent of entities) {
    const bounds = entityGeometryBounds(ent, geometries);
    if (!bounds) continue;

    const [minX, minY, minZ, maxX, maxY, maxZ] = bounds;
    if (
      point[0] >= minX && point[0] <= maxX &&
      point[1] >= minY && point[1] <= maxY &&
      point[2] >= minZ && point[2] <= maxZ
    ) {
      const centre: [number, number, number] = [
        (minX + maxX) / 2,
        (minY + maxY) / 2,
        (minZ + maxZ) / 2,
      ];
      const dist = distanceToPoint(point, centre);
      if (dist < closestDist) {
        closestDist = dist;
        closest = [ent, dist];
      }
    }
  }

  return closest;
}

/**
 * Check if the segment from camera to portal centre is occluded by any
 * entity other than the portal's host wall.
 */
function isOccluded(
  cameraPos: [number, number, number],
  portalCentre: [number, number, number],
  entities: Entity[],
  geometries: Map<string, Geometry>,
  portalEntityId: string,
): boolean {
  const dir: [number, number, number] = [
    portalCentre[0] - cameraPos[0],
    portalCentre[1] - cameraPos[1],
    portalCentre[2] - cameraPos[2],
  ];
  const dist = Math.sqrt(dir[0] ** 2 + dir[1] ** 2 + dir[2] ** 2);
  if (dist < 1e-6) return false;
  const dirNorm: [number, number, number] = [
    dir[0] / dist,
    dir[1] / dist,
    dir[2] / dist,
  ];

  // Step along the ray and test intersection.
  const step = 0.25;
  let d = 0;
  while (d < dist - 0.1) {
    const testPoint: [number, number, number] = [
      cameraPos[0] + dirNorm[0] * d,
      cameraPos[1] + dirNorm[1] * d,
      cameraPos[2] + dirNorm[2] * d,
    ];

    for (const ent of entities) {
      if (ent.id === portalEntityId) continue;
      const bounds = entityGeometryBounds(ent, geometries);
      if (!bounds) continue;
      if (pointInBox(testPoint, bounds)) return true;
    }

    d += step;
  }

  return false;
}

/** Check if a point is inside an axis-aligned bounding box. */
function pointInBox(point: [number, number, number], bounds: [number, number, number, number, number, number]): boolean {
  const [minX, minY, minZ, maxX, maxY, maxZ] = bounds;
  return (
    point[0] >= minX && point[0] <= maxX &&
    point[1] >= minY && point[1] <= maxY &&
    point[2] >= minZ && point[2] <= maxZ
  );
}

/** Check if an entity is a portal (door/window/opening). */
function isPortal(ent: Entity): boolean {
  const cp = ent.custom_properties;
  const kind = cp?.["portal_kind"] as string | undefined;
  if (kind) return true;
  // Also check entity type.
  return ent.type === "door" || ent.type === "window" || ent.type === "opening";
}

/** Get entity geometry bounds. */
function entityGeometryBounds(
  ent: Entity,
  geometries: Map<string, Geometry>,
): [number, number, number, number, number, number] | null {
  const gids = ent.geometry_ids;
  if (!gids) return null;
  for (const gid of gids) {
    const geom = geometries.get(gid);
    if (geom && geom.bounds_min && geom.bounds_max) {
      return [geom.bounds_min.x, geom.bounds_min.y, geom.bounds_min.z, geom.bounds_max.x, geom.bounds_max.y, geom.bounds_max.z];
    }
  }
  return null;
}

/** Dot product of two 3D vectors. */
function dot(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Distance between two 3D points. */
function distanceToPoint(a: [number, number, number], b: [number, number, number]): number {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
}

