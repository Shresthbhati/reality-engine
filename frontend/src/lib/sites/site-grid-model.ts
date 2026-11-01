/**
 * Site composer view-model: pure, no React and no `@/` imports (contract tests
 * load it directly).
 *
 * Coordinates: metres on a top-down plane. +x to the right, +z toward the
 * viewer (front row = larger z), so "back" is drawn at the top of the screen.
 * Placement is USER-DIRECTED - nothing here geo-registers worlds.
 *
 * center_m = ((cell_x+0.5)*grid_cell_m + offset_x_m, (cell_z+0.5)*grid_cell_m + offset_z_m)
 * is the contract the API also computes; `centerM` mirrors it for live drags and
 * the server value stays the source of truth after every save.
 */

export interface Footprint {
  width_m: number | null;
  depth_m: number | null;
  source: "worldir" | "unknown";
}

export interface Placement {
  id: string;
  site_id: string;
  world_id: string;
  world_name: string;
  label: string;
  cell_x: number;
  cell_z: number;
  offset_x_m: number;
  offset_z_m: number;
  rotation_deg: number;
  scale: number;
  center_m: { x: number; z: number };
  footprint: Footprint;
}

export type PlacementPosition = Pick<Placement, "cell_x" | "cell_z" | "offset_x_m" | "offset_z_m">;

export const SCALE_MIN = 0.1;
export const SCALE_MAX = 10;
export const DEFAULT_GRID_CELL_M = 50;

export interface Vec2 {
  x: number;
  z: number;
}

/** Centre of a placement in metres, from its cell + offset (matches the API). */
export function centerM(p: PlacementPosition, gridCellM: number): Vec2 {
  return {
    x: (p.cell_x + 0.5) * gridCellM + p.offset_x_m,
    z: (p.cell_z + 0.5) * gridCellM + p.offset_z_m,
  };
}

/** Inverse of centerM: any metre point -> {cell, offset from that cell's centre}. */
export function positionFromPoint(pt: Vec2, gridCellM: number): PlacementPosition {
  const cell_x = Math.floor(pt.x / gridCellM);
  const cell_z = Math.floor(pt.z / gridCellM);
  return {
    cell_x,
    cell_z,
    offset_x_m: round(pt.x - (cell_x + 0.5) * gridCellM),
    offset_z_m: round(pt.z - (cell_z + 0.5) * gridCellM),
  };
}

/** Placing "in a cell" puts the footprint at that cell's centre. */
export function positionForCell(cellX: number, cellZ: number): PlacementPosition {
  return { cell_x: Math.trunc(cellX), cell_z: Math.trunc(cellZ), offset_x_m: 0, offset_z_m: 0 };
}

export function snapToStep(v: number, step: number): number {
  return step > 0 ? Math.round(v / step) * step : v;
}

export function snapPoint(pt: Vec2, step: number): Vec2 {
  return { x: snapToStep(pt.x, step), z: snapToStep(pt.z, step) };
}

/** Arrow-key nudge: move by (dx, dz) metres, renormalising into cell + offset. */
export function nudge(p: PlacementPosition, dxM: number, dzM: number, gridCellM: number): PlacementPosition {
  const c = centerM(p, gridCellM);
  return positionFromPoint({ x: c.x + dxM, z: c.z + dzM }, gridCellM);
}

/** Keyboard step: a quarter cell, or a twentieth with Shift (fine). */
export function nudgeStepM(gridCellM: number, fine: boolean): number {
  return fine ? round(gridCellM / 20) : round(gridCellM / 4);
}

export function normalizeDeg(deg: number): number {
  const d = deg % 360;
  return round(d < 0 ? d + 360 : d);
}

export function rotateBy(current: number, deltaDeg: number): number {
  return normalizeDeg(current + deltaDeg);
}

export function clampScale(s: number): number {
  if (!Number.isFinite(s)) return 1;
  return Math.min(SCALE_MAX, Math.max(SCALE_MIN, round(s, 3)));
}

export interface FootprintRect {
  /** Width/depth in metres AFTER the placement's scale. */
  width_m: number;
  depth_m: number;
  /** True when the world's size is unknown and a stand-in box is drawn (dashed). */
  isFallback: boolean;
}

/** Unknown footprints are drawn as a clearly-marked box, 60% of a grid cell. */
export function fallbackBoxM(gridCellM: number): number {
  return round(gridCellM * 0.6);
}

export function footprintRect(p: Pick<Placement, "footprint" | "scale">, gridCellM: number): FootprintRect {
  const { width_m, depth_m } = p.footprint;
  const known = p.footprint.source === "worldir" && width_m != null && depth_m != null && width_m > 0 && depth_m > 0;
  const base = fallbackBoxM(gridCellM);
  return {
    width_m: (known ? (width_m as number) : base) * p.scale,
    depth_m: (known ? (depth_m as number) : base) * p.scale,
    isFallback: !known,
  };
}

/** Axis-aligned bounds of a rotated rectangle centred at c. */
export function rotatedExtent(c: Vec2, w: number, d: number, rotationDeg: number): Bounds {
  const r = (rotationDeg * Math.PI) / 180;
  const cos = Math.abs(Math.cos(r));
  const sin = Math.abs(Math.sin(r));
  const hx = (w * cos + d * sin) / 2;
  const hz = (w * sin + d * cos) / 2;
  return { minX: c.x - hx, maxX: c.x + hx, minZ: c.z - hz, maxZ: c.z + hz };
}

export interface ViewTransform {
  /** pixels per metre */
  pxPerM: number;
  /** metre coordinate at the top-left of the viewport */
  originX: number;
  originZ: number;
}

export interface Bounds {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

/** Bounds of all placements, snapped outward to whole cells; spans at least `minCells` per axis so an empty/tiny site stays readable. */
export function contentBounds(placements: Placement[], gridCellM: number, minCells = 3): Bounds {
  let b: Bounds | null = null;
  for (const p of placements) {
    const r = footprintRect(p, gridCellM);
    const e = rotatedExtent(p.center_m, r.width_m, r.depth_m, p.rotation_deg);
    b = b
      ? {
          minX: Math.min(b.minX, e.minX),
          maxX: Math.max(b.maxX, e.maxX),
          minZ: Math.min(b.minZ, e.minZ),
          maxZ: Math.max(b.maxZ, e.maxZ),
        }
      : e;
  }
  const minSpan = minCells * gridCellM;
  if (!b) return { minX: 0, maxX: minSpan, minZ: 0, maxZ: minSpan };
  const minX = Math.floor(b.minX / gridCellM) * gridCellM;
  const minZ = Math.floor(b.minZ / gridCellM) * gridCellM;
  const maxX = Math.max(Math.ceil(b.maxX / gridCellM) * gridCellM, minX + minSpan);
  const maxZ = Math.max(Math.ceil(b.maxZ / gridCellM) * gridCellM, minZ + minSpan);
  return { minX, maxX, minZ, maxZ };
}

/** Fit `bounds` (+ padding in px) inside a viewW x viewH viewport, centred. */
export function fitView(bounds: Bounds, viewW: number, viewH: number, paddingPx = 24): ViewTransform {
  const w = Math.max(bounds.maxX - bounds.minX, 1e-6);
  const h = Math.max(bounds.maxZ - bounds.minZ, 1e-6);
  const availW = Math.max(viewW - 2 * paddingPx, 1);
  const availH = Math.max(viewH - 2 * paddingPx, 1);
  const pxPerM = Math.min(availW / w, availH / h);
  const originX = bounds.minX - (viewW / pxPerM - w) / 2;
  const originZ = bounds.minZ - (viewH / pxPerM - h) / 2;
  return { pxPerM, originX, originZ };
}

export function toScreen(pt: Vec2, v: ViewTransform): { x: number; y: number } {
  return { x: (pt.x - v.originX) * v.pxPerM, y: (pt.z - v.originZ) * v.pxPerM };
}

export function toMetres(px: { x: number; y: number }, v: ViewTransform): Vec2 {
  return { x: px.x / v.pxPerM + v.originX, z: px.y / v.pxPerM + v.originZ };
}

/** Grid line coordinates (metres) covering the visible viewport. */
export function gridLines(v: ViewTransform, viewW: number, viewH: number, gridCellM: number) {
  const maxX = v.originX + viewW / v.pxPerM;
  const maxZ = v.originZ + viewH / v.pxPerM;
  const xs: number[] = [];
  const zs: number[] = [];
  for (let x = Math.floor(v.originX / gridCellM) * gridCellM; x <= maxX; x += gridCellM) xs.push(x);
  for (let z = Math.floor(v.originZ / gridCellM) * gridCellM; z <= maxZ; z += gridCellM) zs.push(z);
  return { xs, zs };
}

/** Which placement ids are in any server-reported overlap pair. */
export function overlappingIds(overlaps: [string, string][]): Set<string> {
  const s = new Set<string>();
  for (const [a, b] of overlaps) {
    s.add(a);
    s.add(b);
  }
  return s;
}

/** Ids that overlap `id` according to the server (never computed client-side). */
export function overlapPartners(id: string, overlaps: [string, string][]): string[] {
  const out: string[] = [];
  for (const [a, b] of overlaps) {
    if (a === id) out.push(b);
    else if (b === id) out.push(a);
  }
  return out;
}

export function cellLabel(cellX: number, cellZ: number): string {
  return `X ${cellX}, Z ${cellZ}`;
}

/** Default label when the user does not type one. */
export function defaultLabel(worldName: string): string {
  return worldName.trim() || "Untitled world";
}

/** Parse a user-typed integer cell index; null if not a finite integer. */
export function parseCell(raw: string): number | null {
  const t = raw.trim();
  if (!/^-?\d+$/.test(t)) return null;
  const n = Number(t);
  return Number.isSafeInteger(n) ? n : null;
}

/** Parse a positive metre value (grid size); null if invalid. */
export function parsePositiveM(raw: string): number | null {
  const n = Number(raw.trim());
  return raw.trim() !== "" && Number.isFinite(n) && n > 0 ? n : null;
}

/** Which grid cell a metre point lands in. */
export function cellAtPoint(pt: Vec2, gridCellM: number): { cellX: number; cellZ: number } {
  return { cellX: Math.floor(pt.x / gridCellM), cellZ: Math.floor(pt.z / gridCellM) };
}

function round(n: number, dp = 4): number {
  const f = 10 ** dp;
  return Math.round(n * f) / f;
}
