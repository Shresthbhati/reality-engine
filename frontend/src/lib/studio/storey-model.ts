/**
 * Studio Multi-Storey & Vertical Topology Client Model (Epic E04-3).
 *
 * Provides models and helper algorithms for multi-level building navigation:
 * - Storey level definition (elevations, heights, rooms, entities)
 * - Staircase connectors (treads, risers, landings, connectivity)
 * - Inter-level clipping and isolation planes
 * - Multi-floor vertical path planning
 */

export interface StoreyLevelItem {
  storey_id: string;
  name: string;
  level_index: number;
  floor_elevation_m: number;
  ceiling_elevation_m: number;
  height_m: number;
  room_ids: string[];
  entity_ids: string[];
}

export interface StaircaseConnectorItem {
  stair_id: string;
  name: string;
  from_storey_id: string;
  to_storey_id: string;
  num_risers: number;
  total_rise_m: number;
  total_run_m: number;
  flight_width_m: number;
}

export interface StoreySelectionState {
  activeStoreyId: string | "all";
  isolateActiveStorey: boolean;
  highlightStaircases: boolean;
}

/**
 * Sorts storey levels in ascending vertical elevation order.
 */
export function sortStoreysByElevation(storeys: StoreyLevelItem[]): StoreyLevelItem[] {
  return [...storeys].sort((a, b) => a.floor_elevation_m - b.floor_elevation_m);
}

/**
 * Determines whether a given 3D point lies within the vertical bounds of a storey.
 */
export function isPointInStorey(
  point: [number, number, number],
  storey: StoreyLevelItem,
  tolerance_m: number = 0.05,
): boolean {
  const z = point[2];
  return z >= storey.floor_elevation_m - tolerance_m && z <= storey.ceiling_elevation_m + tolerance_m;
}

/**
 * Resolves the corresponding storey for a given elevation in meters.
 */
export function getStoreyForElevation(
  elevation_m: number,
  storeys: StoreyLevelItem[],
): StoreyLevelItem | null {
  const sorted = sortStoreysByElevation(storeys);
  for (const s of sorted) {
    if (elevation_m >= s.floor_elevation_m && elevation_m <= s.ceiling_elevation_m) {
      return s;
    }
  }
  // Fallback to closest storey
  if (sorted.length === 0) return null;
  let closest = sorted[0];
  let minDiff = Math.abs(elevation_m - closest.floor_elevation_m);
  for (let i = 1; i < sorted.length; i++) {
    const diff = Math.abs(elevation_m - sorted[i].floor_elevation_m);
    if (diff < minDiff) {
      minDiff = diff;
      closest = sorted[i];
    }
  }
  return closest;
}

/**
 * Computes near and far Z-clip planes to isolate a selected storey in Three.js viewport.
 */
export function calculateStoreyClipBounds(
  storey: StoreyLevelItem,
  padding_m: number = 0.20,
): { minZ: number; maxZ: number } {
  return {
    minZ: storey.floor_elevation_m - padding_m,
    maxZ: storey.ceiling_elevation_m + padding_m,
  };
}

/**
 * Summarizes building storeys (e.g., "Ground Floor (0.0m), Level 1 (+3.0m)").
 */
export function formatStoreyBadge(storey: StoreyLevelItem): string {
  const sign = storey.floor_elevation_m > 0 ? "+" : "";
  return `${storey.name} (${sign}${storey.floor_elevation_m.toFixed(1)}m)`;
}
