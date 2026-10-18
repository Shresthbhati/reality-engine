/**
 * Pure model behind the map / plot view (B6). No React, no DOM: everything the
 * panel prints is computed here from real WorldIR geometry so it can be
 * contract-tested.
 *
 * Frame: WorldIR is Y-up, so the plan (top-down) plane is x/z.
 */
import type { CameraPose, WorldIR } from "@/types/worldir";

export interface Footprint {
  id: string;
  name: string;
  type: string;
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  /** Axis-aligned box footprint, in model units squared. */
  area: number;
}

export interface PlanExtent {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  width: number;
  depth: number;
}

export type LengthUnit = "m" | "model units";

export interface UnitContext {
  unit: LengthUnit;
  /** Multiply model-unit lengths by this to get the displayed unit. */
  factor: number;
  /** Why this unit was chosen; shown verbatim as a tooltip. */
  reason: string;
}

const finite = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);

/** Footprints from the first geometry of each entity (same box the 3D viewport draws). */
export function entityFootprints(world: WorldIR | null | undefined): Footprint[] {
  if (!world?.entities || !world.geometries) return [];
  const out: Footprint[] = [];
  for (const e of Object.values(world.entities)) {
    const g = world.geometries[(e.geometry_ids ?? [])[0] ?? ""];
    if (!g?.bounds_min || !g?.bounds_max) continue;
    const { x: ax, z: az } = g.bounds_min;
    const { x: bx, z: bz } = g.bounds_max;
    if (![ax, az, bx, bz].every(finite)) continue;
    const minX = Math.min(ax, bx), maxX = Math.max(ax, bx);
    const minZ = Math.min(az, bz), maxZ = Math.max(az, bz);
    out.push({
      id: e.id,
      name: e.name || e.id,
      type: e.type,
      minX, maxX, minZ, maxZ,
      area: (maxX - minX) * (maxZ - minZ),
    });
  }
  return out;
}

/** Bounding extent of footprints (and camera positions, if given). Null when nothing to bound. */
export function planExtent(fps: Footprint[], cameras: CameraPose[] = []): PlanExtent | null {
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const f of fps) {
    minX = Math.min(minX, f.minX); maxX = Math.max(maxX, f.maxX);
    minZ = Math.min(minZ, f.minZ); maxZ = Math.max(maxZ, f.maxZ);
  }
  for (const c of cameras) {
    const [x, , z] = c.position_m;
    if (!finite(x) || !finite(z)) continue;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
  }
  if (!finite(minX) || !finite(minZ)) return null;
  return { minX, maxX, minZ, maxZ, width: maxX - minX, depth: maxZ - minZ };
}

/**
 * Area covered by the union of the footprint boxes (overlaps counted once).
 * Slab sweep over compressed x coordinates: O(n^2 log n), fine for scene sizes.
 */
export function unionArea(fps: Footprint[]): number {
  if (fps.length === 0) return 0;
  const xs = [...new Set(fps.flatMap((f) => [f.minX, f.maxX]))].sort((a, b) => a - b);
  let total = 0;
  for (let i = 0; i + 1 < xs.length; i++) {
    const x0 = xs[i], x1 = xs[i + 1];
    const spans = fps
      .filter((f) => f.minX <= x0 && f.maxX >= x1)
      .map((f) => [f.minZ, f.maxZ] as const)
      .sort((a, b) => a[0] - b[0]);
    let covered = 0, curStart = 0, curEnd = 0, open = false;
    for (const [s, e] of spans) {
      if (!open) { curStart = s; curEnd = e; open = true; }
      else if (s <= curEnd) curEnd = Math.max(curEnd, e);
      else { covered += curEnd - curStart; curStart = s; curEnd = e; }
    }
    if (open) covered += curEnd - curStart;
    total += covered * (x1 - x0);
  }
  return total;
}

/**
 * Unit rule (matches the 3D viewport's `isUnscaled` check): lengths are metres
 * only when the world records a positive meters_per_unit AND its scale state is
 * not "unscaled"/"nominal". Otherwise they stay in model units - never relabelled.
 */
export function resolveUnitContext(world: WorldIR | null | undefined): UnitContext {
  const scale = world?.metadata?.scale;
  const state = scale?.state;
  if (state === "unscaled" || state === "nominal") {
    return { unit: "model units", factor: 1, reason: `Scale state is "${state}": absolute size is not established.` };
  }
  if (finite(scale?.meters_per_unit) && scale.meters_per_unit > 0) {
    return { unit: "m", factor: scale.meters_per_unit, reason: `Scaled by ${scale.meters_per_unit} m per model unit${scale.method ? ` (${scale.method})` : ""}.` };
  }
  return { unit: "model units", factor: 1, reason: "No metric scale is recorded for this world." };
}

/** A world is georeferenced only if it has a real, in-range origin. */
export function isGeoreferenced(lat: number | null | undefined, lng: number | null | undefined): boolean {
  return finite(lat) && finite(lng) && Math.abs(lat) <= 90 && Math.abs(lng) <= 180 && !(lat === 0 && lng === 0);
}

export function formatLength(modelUnits: number, ctx: UnitContext): string {
  const v = modelUnits * ctx.factor;
  return `${v >= 100 ? v.toFixed(0) : v.toFixed(2)} ${ctx.unit}`;
}

export function formatAreaValue(modelUnitsSq: number, ctx: UnitContext): string {
  const v = modelUnitsSq * ctx.factor * ctx.factor;
  const u = ctx.unit === "m" ? "m²" : "model units²";
  return `${v >= 100 ? v.toFixed(0) : v.toFixed(2)} ${u}`;
}

/** Status-bar text for a cursor over the geographic map. */
export function formatLatLon(lat: number, lng: number): string {
  return `${Math.abs(lat).toFixed(5)}°${lat >= 0 ? "N" : "S"}  ${Math.abs(lng).toFixed(5)}°${lng >= 0 ? "E" : "W"}`;
}
