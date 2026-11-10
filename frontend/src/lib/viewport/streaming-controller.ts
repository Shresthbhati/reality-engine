/**
 * Streaming + memory-budget controller (N-29).
 *
 * Loads and unloads geometry, textures, splats and other assets according to
 * view distance, importance and available hardware memory. Exposes a pure,
 * testable model: given a set of assets with bounds and importance, plus the
 * current camera position and memory budget, it returns which assets should be
 * resident and which should be evicted.
 *
 * Honesty rules:
 *   * An asset that exceeds the budget is never silently loaded; it is
 *     reported as over-budget with its size so the UI can show a placeholder.
 *   * Eviction is always LRU-by-distance (furthest first), never random.
 *   * Every eviction / load carries a reason so the UI can explain it.
 */

import type { Entity } from "@/types/worldir";

/** One loadable asset with bounds and importance. */
export interface LoadableAsset {
  id: string;
  /** World-space bounding box [minX, minY, minZ, maxX, maxY, maxZ]. */
  bounds: [number, number, number, number, number, number];
  /** Importance 0..1 (1 = always resident, 0 = first to evict). */
  importance: number;
  /** Asset size in KB (decoded, on GPU). */
  sizeKb: number;
  /** Asset kind for logging / UI. */
  kind: "mesh" | "splat" | "texture" | "pointcloud" | "architectural";
  /** Source entity ids that back this asset. */
  sourceEntityIds: string[];
}

/** One load/evict decision with reason. */
export interface AssetDecision {
  id: string;
  kind: "load" | "unload" | "keep" | "over_budget";
  /** Human-readable reason for the decision. */
  reason: string;
  /** Size in KB involved. */
  sizeKb: number;
  /** New resident set size in KB after applying this decision. */
  newResidentKb: number;
}

/** Streaming controller state. */
export interface StreamingControllerState {
  /** Currently resident asset ids. */
  resident: Set<string>;
  /** Total resident size in KB. */
  residentKb: number;
  /** Decisions to apply this frame. */
  decisions: AssetDecision[];
  /** Assets that exceed the budget entirely. */
  overBudget: AssetDecision[];
}

/** Camera state for distance-based eviction. */
export interface CameraState {
  position: [number, number, number];
  /** Far plane in metres. */
  far: number;
}

/**
 * Compute streaming decisions for a frame.
 *
 * Algorithm:
 *   1. Compute per-asset distance from camera to bounds centre.
 *   2. Sort assets by (importance desc, distance asc): important + close = keep.
 *   3. Greedily load until memory budget is exhausted.
 *   4. Evict furthest low-importance assets first if over budget.
 *   5. Anything that still doesn't fit is over_budget.
 *
 * Pure function: same inputs -> same outputs.
 */
export function streamFrame(
  assets: LoadableAsset[],
  camera: CameraState,
  budgetKb: number,
  previousResident: Set<string>,
): StreamingControllerState {
  const decisions: AssetDecision[] = [];
  const newResident = new Set<string>();
  let residentKb = 0;

  // Compute distance for each asset.
  const withDistance = assets.map((a) => ({
    ...a,
    distance: distanceToBounds(camera.position, a.bounds),
  }));

  // Sort: important first, then closer first.
  withDistance.sort((a, b) => {
    if (b.importance !== a.importance) return b.importance - a.importance;
    return a.distance - b.distance;
  });

  // Greedy load.
  for (const a of withDistance) {
    if (residentKb + a.sizeKb <= budgetKb) {
      newResident.add(a.id);
      residentKb += a.sizeKb;
      decisions.push({
        id: a.id,
        kind: previousResident.has(a.id) ? "keep" : "load",
        reason: previousResident.has(a.id)
          ? "Already resident; keeping in memory."
          : `Loaded within budget (${a.sizeKb.toFixed(0)} KB; ${Math.round(((residentKb / budgetKb) * 100))}% of budget used).`,
        sizeKb: a.sizeKb,
        newResidentKb: residentKb,
      });
    } else {
      // Doesn't fit.
      decisions.push({
        id: a.id,
        kind: "over_budget",
        reason: `Exceeds remaining budget (needs ${a.sizeKb.toFixed(0)} KB; ${Math.round(
          ((budgetKb - residentKb) / budgetKb) * 100,
        )}% of budget remains).`,
        sizeKb: a.sizeKb,
        newResidentKb: residentKb,
      });
    }
  }

  // Evict previously resident assets that didn't make the cut (furthest first).
  const previouslyResident = [...previousResident].filter((id) => !newResident.has(id));
  previouslyResident.sort((aId, bId) => {
    const a = assets.find((x) => x.id === aId);
    const b = assets.find((x) => x.id === bId);
    const ad = a ? distanceToBounds(camera.position, a.bounds) : Infinity;
    const bd = b ? distanceToBounds(camera.position, b.bounds) : Infinity;
    return bd - ad; // furthest first
  });

  for (const id of previouslyResident) {
    const a = assets.find((x) => x.id === id);
    decisions.push({
      id,
      kind: "unload",
      reason: a
        ? `Evicted to free memory (${a.sizeKb.toFixed(0)} KB; furthest / lowest importance).`
        : "Evicted (asset no longer known).",
        sizeKb: a?.sizeKb ?? 0,
        newResidentKb: residentKb,
      });
  }

  // Over-budget list (sorted by size desc so the UI can show the biggest first).
  const overBudget = decisions
    .filter((d) => d.kind === "over_budget")
    .sort((a, b) => b.sizeKb - a.sizeKb);

  return {
    resident: newResident,
    residentKb,
    decisions,
    overBudget,
  };
}

/** Distance from a point to an axis-aligned bounding box. */
function distanceToBounds(point: [number, number, number], bounds: [number, number, number, number, number, number]): number {
  const [minX, minY, minZ, maxX, maxY, maxZ] = bounds;
  const [px, py, pz] = point;

  // Clamp point to box.
  const cx = Math.min(Math.max(px, minX), maxX);
  const cy = Math.min(Math.max(py, minY), maxY);
  const cz = Math.min(Math.max(pz, minZ), maxZ);

  const dx = px - cx;
  const dy = py - cy;
  const dz = pz - cz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Recommend a memory budget based on device class and available RAM.
 * Returns KB.
 */
export function recommendBudgetKb(deviceClass: "mobile" | "desktop" | "workstation", availableRamMb: number): number {
  // Conservative: use at most 25% of available RAM for GPU assets.
  const baseKb = (availableRamMb * 1024) / 4;
  switch (deviceClass) {
    case "mobile":
      return Math.min(baseKb, 512 * 1024); // Cap at 512 MB on mobile.
    case "desktop":
      return Math.min(baseKb, 2 * 1024 * 1024); // Cap at 2 GB on desktop.
    case "workstation":
      return Math.min(baseKb, 4 * 1024 * 1024); // Cap at 4 GB on workstation.
    default:
      return Math.min(baseKb, 1 * 1024 * 1024);
  }
}

/** Build asset list from WorldIR entities + geometries. */
export function buildAssetsFromWorld(
  entities: Entity[],
  geometries: Map<string, { type: string; vertex_count?: number; triangle_count?: number; data_uri?: string }>,
  defaultSizeKb = 256,
): LoadableAsset[] {
  const assets: LoadableAsset[] = [];

  for (const ent of entities) {
    const gids = ent.geometry_ids;
    if (!gids) continue;
    for (const gid of gids) {
      const geom = geometries.get(gid);
      if (!geom) continue;

      const kind = geometryKind(geom.type);
      const vc = geom.vertex_count ?? 0;
      const tc = geom.triangle_count ?? 0;
      const sizeKb = (vc * 12 + tc * 12) / 1024;

      assets.push({
        id: gid,
        bounds: geometryBounds(geom),
        importance: entityImportance(ent),
        sizeKb,
        kind,
        sourceEntityIds: [ent.id],
      });
    }
  }

  return assets;
}

function geometryKind(type: string): LoadableAsset["kind"] {
  switch (type) {
    case "splat":
      return "splat";
    case "pointcloud":
      return "pointcloud";
    case "mesh":
      return "mesh";
    case "box":
    case "cad":
    case "parametric":
    case "brep":
      return "architectural";
    default:
      return "mesh";
  }
}

function geometryBounds(geom: any): [number, number, number, number, number, number] {
  if (geom.bounds_min && geom.bounds_max) {
    return [geom.bounds_min[0], geom.bounds_min[1], geom.bounds_min[2], geom.bounds_max[0], geom.bounds_max[1], geom.bounds_max[2]];
  }
  // Default: unit box at origin.
  return [0, 0, 0, 1, 1, 1];
}

function entityImportance(ent: Entity): number {
  // Architectural entities (walls, floors, rooms) are more important than
  // furniture; structural > decorative.
  const structural = new Set(["wall", "floor", "ceiling", "room", "door", "window"]);
  const furniture = new Set(["table", "chair", "cabinet", "sofa", "bed", "shelf", "fixture"]);
  if (structural.has(ent.type as any)) return 0.9;
  if (furniture.has(ent.type as any)) return 0.6;
  return 0.3;
}

