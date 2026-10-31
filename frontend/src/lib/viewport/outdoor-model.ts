/**
 * Outdoor model: turns map-imported / generated entities (buildings, roads, rail, bridges, street furniture, trees)
 * into plain massing parts and triangle arrays. No three.js, so it is unit-testable under Node; three-scene.ts only
 * wraps the arrays. Everything drawn comes from a WorldIR entity that carries geometry (footprint polygon, polyline,
 * position or bounds). Nothing is invented: an entity without geometry is listed in `diagnostics.skipped` instead.
 *
 * These are honest massing proxies, not photorealism: sizes that were defaulted (not tagged) are flagged `estimated`.
 * Frame: the world's own frame. Y-up worlds (OSM) keep ground = (x, z), height = y; Z-up worlds (composed) keep
 * ground = (x, y), height = z. Geometry is emitted in the data frame so the viewer's existing root rotation applies.
 */

import type { Entity, WorldIR } from "@/types/worldir";
import { triangulate2D } from "./room-shell-model.ts";
import { upAxisOf } from "./view-model.ts";

type RGB = [number, number, number];
type P2 = [number, number];
type Rec = Record<string, unknown>;

export type OutdoorShape = "extrusion" | "strip" | "pole" | "box" | "tree" | "stairs";
export type OutdoorGroup = "buildings" | "roads" | "rail" | "bridges" | "furniture" | "trees" | "other";

export const OUTDOOR_GROUP_ORDER: OutdoorGroup[] = ["buildings", "roads", "rail", "bridges", "furniture", "trees", "other"];
export const OUTDOOR_GROUP_LABEL: Record<OutdoorGroup, string> = {
  buildings: "Buildings",
  roads: "Roads & paths",
  rail: "Rail & metro",
  bridges: "Bridges",
  furniture: "Street furniture",
  trees: "Trees",
  other: "Other map features",
};

export interface OutdoorPart {
  entityId: string;
  /** normalised kind, e.g. "building", "road", "lamp_post", "jersey_barrier" */
  kind: string;
  group: OutdoorGroup;
  shape: OutdoorShape;
  name: string;
  /** closed ground polygon (extrusion, and flat polygon strips such as a generated driveway) */
  polygon: P2[] | null;
  /** ground polyline with `width` (strips, stairs) */
  polyline: P2[] | null;
  width: number;
  /** ground point (pole, box, tree) */
  position: P2 | null;
  /** box footprint (ground axes) */
  size: P2;
  /** height of the base above the ground plane */
  baseY: number;
  /** vertical extent above baseY (extrusion / box / pole / canopy top / strip thickness) */
  height: number;
  color: RGB;
  /** entity confidence 0-1 straight from the WorldIR; null = not stated (never invented) */
  confidence: number | null;
  /** some dimension was a default, or the shape is a fallback proxy */
  estimated: boolean;
  imported: boolean;
  generated: boolean;
  /** bridges: draw two thin piers under the deck ends */
  piers: boolean;
  /** why the shape is a proxy (footprint missing, unknown kind, underground ...) */
  note?: string;
}

export interface OutdoorDiagnostics {
  drawn: number;
  skipped: Array<{ id: string; reason: string }>;
  byGroup: Record<OutdoorGroup, number>;
  byKind: Record<string, number>;
}

export interface OutdoorResult {
  up: "y" | "z";
  parts: OutdoorPart[];
  diagnostics: OutdoorDiagnostics;
}

// ---- palette (derived from the scene's CSS tokens) ---------------------------------------------------------------

export interface OutdoorPalette {
  building: RGB; road: RGB; sidewalk: RGB; rail: RGB; bridge: RGB; furniture: RGB; tree: RGB; neutral: RGB;
}

export const DEFAULT_OUTDOOR_PALETTE: OutdoorPalette = {
  building: [168, 158, 142], road: [66, 70, 66], sidewalk: [120, 124, 116], rail: [197, 138, 96],
  bridge: [132, 136, 130], furniture: [222, 182, 120], tree: [82, 132, 88], neutral: [140, 142, 136],
};

const mix = (a: RGB, b: RGB, t: number): RGB => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)) as RGB;

/** Token-derived palette: surfaces from --bg/--border, buildings warmed toward --accent, rail/furniture in the accent. */
export function derivePalette(t: { bg: RGB; gridMajor: RGB; text: RGB; textSecondary: RGB; accent: RGB; high: RGB }): OutdoorPalette {
  return {
    building: mix(mix(t.textSecondary, t.text, 0.15), t.accent, 0.12),
    road: mix(t.bg, t.textSecondary, 0.34),
    sidewalk: mix(t.bg, t.textSecondary, 0.62),
    rail: t.accent,
    bridge: mix(t.bg, t.text, 0.5),
    furniture: mix(t.accent, t.text, 0.35),
    tree: mix(t.high, t.bg, 0.4),
    neutral: mix(t.bg, t.textSecondary, 0.6),
  };
}

// ---- kind table ------------------------------------------------------------------------------------------------

const BARRIER_KINDS = new Set([
  "barrier", "gate", "jersey_barrier", "fence", "wall", "bollard", "guard_rail", "hedge", "retaining_wall", "kerb",
  "chain", "cable_barrier", "city_wall", "handrail", "lift_gate", "block", "yes", "barricade",
]);
const ROAD_KINDS = new Set(["road", "street", "driveway", "sidewalk", "footway", "path", "cycleway", "pedestrian", "corridor"]);
const RAIL_KINDS = new Set(["metro_line", "rail", "subway", "tram", "light_rail"]);
const POLE_KINDS = new Set(["lamp_post", "traffic_sign"]);
const SMALL_BOX_KINDS = new Set(["bin", "post_box", "bench", "fire_hydrant"]);

/** Fallback sizes (side m, height m) for point features that carry no dimensions; always flagged estimated. */
const POINT_FALLBACK: Record<string, [number, number]> = {
  lamp_post: [0.3, 5], traffic_sign: [0.3, 2.5], bin: [0.5, 0.9], post_box: [0.5, 1.2], bench: [1.5, 0.8],
  fire_hydrant: [0.3, 0.8], station: [20, 6], station_entrance: [3, 3], tree: [3, 6],
};

const lower = (v: unknown): string => (typeof v === "string" ? v.toLowerCase() : "");
const cp = (e: Entity): Rec => (e.custom_properties ?? {}) as Rec;
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Map-imported or generated (connector) entities: the only ones the outdoor layer ever draws. */
export function isOutdoorEntity(e: Entity): boolean {
  const p = cp(e);
  if (p.organizational === true) return false; // compose containers (plot / building shells) own no geometry
  return p.imported === true || p.source === "OpenStreetMap" || !!p.generated_by;
}

function kindOf(e: Entity): string {
  const p = cp(e);
  const k = lower(p.kind) || lower(p.road_kind) || lower(p.connector_kind);
  if (k) return k;
  const t = lower(e.type);
  if (t === "road" || t === "street" || t === "sidewalk" || t === "corridor") return t;
  return t === "building" ? "building" : "unknown";
}

interface KindSpec { group: OutdoorGroup; label: string; plural: string }
const SPEC: Record<string, KindSpec> = {
  building: { group: "buildings", label: "Building", plural: "buildings" },
  road: { group: "roads", label: "Road", plural: "roads" },
  street: { group: "roads", label: "Street", plural: "streets" },
  driveway: { group: "roads", label: "Driveway", plural: "driveways" },
  sidewalk: { group: "roads", label: "Sidewalk", plural: "sidewalks" },
  corridor: { group: "roads", label: "Corridor", plural: "corridors" },
  steps: { group: "roads", label: "Steps", plural: "steps" },
  stairs: { group: "roads", label: "Steps", plural: "steps" },
  metro_line: { group: "rail", label: "Metro line", plural: "metro lines" },
  rail: { group: "rail", label: "Rail", plural: "rail lines" },
  station: { group: "rail", label: "Station", plural: "stations" },
  station_entrance: { group: "rail", label: "Station entrance", plural: "station entrances" },
  bridge: { group: "bridges", label: "Bridge", plural: "bridges" },
  lamp_post: { group: "furniture", label: "Lamp post", plural: "lamp posts" },
  bin: { group: "furniture", label: "Bin", plural: "bins" },
  post_box: { group: "furniture", label: "Post box", plural: "post boxes" },
  bench: { group: "furniture", label: "Bench", plural: "benches" },
  traffic_sign: { group: "furniture", label: "Traffic sign", plural: "traffic signs" },
  fire_hydrant: { group: "furniture", label: "Fire hydrant", plural: "fire hydrants" },
  gate: { group: "furniture", label: "Gate", plural: "gates" },
  tree: { group: "trees", label: "Tree", plural: "trees" },
};

function specFor(kind: string): KindSpec {
  const s = SPEC[kind];
  if (s) return s;
  const label = kind.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
  if (BARRIER_KINDS.has(kind) || RAIL_KINDS.has(kind) || ROAD_KINDS.has(kind)) {
    const group: OutdoorGroup = RAIL_KINDS.has(kind) ? "rail" : ROAD_KINDS.has(kind) ? "roads" : "furniture";
    return { group, label, plural: `${label.toLowerCase()}s` };
  }
  return { group: "other", label: kind === "unknown" ? "Map feature" : label, plural: kind === "unknown" ? "map features" : `${label.toLowerCase()}s` };
}

export function kindLabel(kind: string): string { return specFor(kind).label; }

/** Scene-tree / HUD group of an entity, or null when it is not an outdoor entity. */
export function outdoorGroupOf(e: Entity): OutdoorGroup | null {
  return isOutdoorEntity(e) ? specFor(kindOf(e)).group : null;
}

export function outdoorKindOf(e: Entity): string | null {
  return isOutdoorEntity(e) ? kindOf(e) : null;
}

/** Display name: the OSM `name` tag when present, else "<Kind> <n>" (n is the caller's per-kind running number). */
export function outdoorDisplayName(e: Entity, n: number): string {
  const tags = (cp(e).osm_tags ?? {}) as Rec;
  const tagged = typeof tags.name === "string" ? tags.name.trim() : "";
  return tagged || `${kindLabel(kindOf(e))} ${String(n).padStart(3, "0")}`;
}

// ---- geometry extraction --------------------------------------------------------------------------------------

type Up = "y" | "z";
const isV3 = (p: unknown): p is [number, number, number] =>
  Array.isArray(p) && p.length === 3 && p.every((n) => typeof n === "number" && Number.isFinite(n));

const ground = (p: [number, number, number], up: Up): P2 => (up === "y" ? [p[0], p[2]] : [p[0], p[1]]);
const heightOf = (p: [number, number, number], up: Up): number => (up === "y" ? p[1] : p[2]);

function ring(value: unknown, up: Up): { pts: P2[]; baseY: number } | null {
  if (!Array.isArray(value) || value.length < 3 || !value.every(isV3)) return null;
  let pts = (value as Array<[number, number, number]>).map((p) => ground(p, up));
  if (pts.length > 3 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]) pts = pts.slice(0, -1);
  if (pts.length < 3) return null;
  const vs = (value as Array<[number, number, number]>).map((p) => heightOf(p, up));
  return { pts, baseY: vs.reduce((s, v) => s + v, 0) / vs.length };
}

function line(value: unknown, up: Up): { pts: P2[]; baseY: number } | null {
  if (!Array.isArray(value) || value.length < 2 || !value.every(isV3)) return null;
  const raw = value as Array<[number, number, number]>;
  const pts: P2[] = [];
  raw.forEach((p) => { const g = ground(p, up); const l = pts[pts.length - 1]; if (!l || l[0] !== g[0] || l[1] !== g[1]) pts.push(g); });
  if (pts.length < 2) return null;
  return { pts, baseY: raw.reduce((s, p) => s + heightOf(p, up), 0) / raw.length };
}

function boundsOf(p: Rec): { min: [number, number, number]; max: [number, number, number] } | null {
  return isV3(p.bounds_min) && isV3(p.bounds_max) ? { min: p.bounds_min, max: p.bounds_max } : null;
}

function dimValue(p: Rec, key: string): { value: number; estimated: boolean } | null {
  const d = (p.dimensions ?? {}) as Rec;
  const dim = d[key] as Rec | undefined;
  const v = num(dim?.value);
  return v !== null && v > 0 ? { value: v, estimated: lower(dim?.state) === "estimated" } : null;
}

function anyEstimatedDim(p: Rec): boolean {
  return Object.values((p.dimensions ?? {}) as Rec).some((d) => lower((d as Rec | undefined)?.state) === "estimated");
}

/** Build every outdoor part. Entities without drawable geometry are listed in diagnostics.skipped, never drawn. */
export function buildOutdoor(world: WorldIR | null | undefined, palette: OutdoorPalette = DEFAULT_OUTDOOR_PALETTE): OutdoorResult {
  const up = upAxisOf(world);
  const parts: OutdoorPart[] = [];
  const skipped: Array<{ id: string; reason: string }> = [];
  const entities = Object.values(world?.entities ?? {}).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  for (const e of entities) {
    if (!isOutdoorEntity(e)) continue;
    const r = partFor(e, up, palette);
    if ("reason" in r) skipped.push({ id: e.id, reason: r.reason });
    else parts.push(r);
  }
  const byGroup = Object.fromEntries(OUTDOOR_GROUP_ORDER.map((g) => [g, 0])) as Record<OutdoorGroup, number>;
  const byKind: Record<string, number> = {};
  for (const p of parts) { byGroup[p.group]++; byKind[p.kind] = (byKind[p.kind] ?? 0) + 1; }
  return { up, parts, diagnostics: { drawn: parts.length, skipped, byGroup, byKind } };
}

export function buildOutdoorParts(world: WorldIR | null | undefined, palette?: OutdoorPalette): OutdoorPart[] {
  return buildOutdoor(world, palette).parts;
}

function partFor(e: Entity, up: Up, pal: OutdoorPalette): OutdoorPart | { reason: string } {
  const p = cp(e);
  const kind = kindOf(e);
  const spec = specFor(kind);
  const bounds = boundsOf(p);
  const base: OutdoorPart = {
    entityId: e.id, kind, group: spec.group, shape: "box" as OutdoorShape, name: e.name || spec.label, polygon: null, polyline: null, width: 0,
    position: null, size: [1, 1] as P2, baseY: 0, height: 0.1, color: pal.neutral,
    confidence: num(e.confidence), estimated: anyEstimatedDim(p), imported: p.imported === true || p.source === "OpenStreetMap",
    generated: !!p.generated_by, piers: false,
  };
  const out = (o: Partial<OutdoorPart>): OutdoorPart => ({ ...base, ...o });
  const boundsHeight = bounds ? Math.max(0, up === "y" ? bounds.max[1] - bounds.min[1] : bounds.max[2] - bounds.min[2]) : 0;
  const boundsCentre = (): P2 | null => bounds ? [(bounds.min[0] + bounds.max[0]) / 2, up === "y" ? (bounds.min[2] + bounds.max[2]) / 2 : (bounds.min[1] + bounds.max[1]) / 2] : null;
  const boundsSize = (): P2 | null => bounds ? [Math.max(0.1, bounds.max[0] - bounds.min[0]), Math.max(0.1, up === "y" ? bounds.max[2] - bounds.min[2] : bounds.max[1] - bounds.min[1])] : null;
  const boundsBase = bounds ? (up === "y" ? bounds.min[1] : bounds.min[2]) : 0;

  const poly = ring(p.footprint_polygon_xyz_m, up);
  const path = line(p.polyline_xyz_m, up);
  const at = isV3(p.position_xyz_m) ? ground(p.position_xyz_m, up) : null;
  const widthDim = dimValue(p, "width") ?? dimValue(p, "width_m");

  // ---- buildings
  if (spec.group === "buildings") {
    const h = dimValue(p, "height");
    if (poly) {
      return out({ shape: "extrusion", polygon: poly.pts, baseY: poly.baseY, height: h?.value ?? (boundsHeight || 6), color: pal.building,
        estimated: base.estimated || !h, note: p.footprint_estimated === true ? "footprint estimated (centre point only)" : undefined });
    }
    const c = boundsCentre(), s = boundsSize();
    if (c && s) {
      return out({ shape: "box", position: c, size: s, baseY: boundsBase, height: h?.value ?? (boundsHeight || 6), color: pal.building,
        estimated: true, note: "footprint missing: bounding box proxy" });
    }
    return { reason: "building has no footprint polygon or bounds" };
  }

  // ---- ways: roads, sidewalks, rail, bridges, steps, barriers drawn along their way
  const isRoad = spec.group === "roads" && kind !== "steps" && kind !== "stairs";
  const isRail = RAIL_KINDS.has(kind);
  if (path || (poly && (isRoad || isRail))) {
    const w = widthDim?.value ?? (kind === "bridge" ? 6 : isRail ? 3 : isRoad ? 5 : BARRIER_KINDS.has(kind) ? 0.3 : 1.5);
    const wEstimated = !widthDim || widthDim.estimated;
    if (kind === "bridge") {
      const deck = dimValue(p, "deck_elevation")?.value ?? path?.baseY ?? 0;
      return out({ shape: "strip", polyline: path?.pts ?? null, polygon: path ? null : poly!.pts, width: w, baseY: deck, height: 0.8,
        color: pal.bridge, estimated: true, piers: deck > 0.5 && !!path, note: "deck elevation and piers estimated" });
    }
    if (kind === "steps" || kind === "stairs") {
      return out({ shape: "stairs", polyline: path?.pts ?? null, width: w, baseY: path?.baseY ?? 0, height: 0.6, color: pal.sidewalk, estimated: true,
        note: "step count and rise estimated" });
    }
    if (isRail) {
      const underground = p.underground === true || p.tunnel === true;
      return out({ shape: "strip", polyline: path?.pts ?? null, polygon: path ? null : poly!.pts, width: w, baseY: underground ? -0.6 : 0, height: 0.12,
        color: pal.rail, estimated: base.estimated || wEstimated || underground, note: underground ? "underground (drawn below ground)" : undefined });
    }
    if (isRoad) {
      const sidewalk = kind !== "road" && kind !== "street" && kind !== "driveway" && kind !== "corridor";
      return out({ shape: "strip", polyline: path?.pts ?? null, polygon: path ? null : poly!.pts, width: w, baseY: path?.baseY ?? poly!.baseY,
        height: sidewalk ? 0.08 : 0.04, color: sidewalk ? pal.sidewalk : pal.road, estimated: base.estimated || wEstimated });
    }
    // barrier-like or unknown way: a low wall along the way
    const h = dimValue(p, "height")?.value ?? (boundsHeight || 1);
    return out({ shape: "strip", polyline: path?.pts ?? null, polygon: path ? null : poly!.pts, width: w, baseY: path?.baseY ?? poly!.baseY, height: h,
      color: BARRIER_KINDS.has(kind) ? pal.furniture : pal.neutral, estimated: base.estimated || wEstimated || !dimValue(p, "height"),
      note: BARRIER_KINDS.has(kind) ? undefined : "unknown kind: neutral proxy" });
  }

  // ---- point features
  const pos = at ?? boundsCentre();
  if (!pos) return { reason: "no footprint, polyline, position or bounds" };
  const fb = POINT_FALLBACK[kind];
  const hDim = dimValue(p, "height");
  const sDim = dimValue(p, "footprint_side");
  const height = hDim?.value ?? (boundsHeight || fb?.[1] || 1);
  const side = sDim?.value ?? fb?.[0] ?? (boundsSize() ? Math.max(...boundsSize()!) : 1);
  const est = base.estimated || !hDim || !sDim;
  const baseY = at ? 0 : boundsBase;
  if (kind === "tree") return out({ shape: "tree", position: pos, size: [side, side], baseY, height, color: pal.tree, estimated: est });
  if (POLE_KINDS.has(kind)) return out({ shape: "pole", position: pos, size: [Math.min(side, 0.3), Math.min(side, 0.3)], baseY, height, color: pal.furniture, estimated: est });
  if (SMALL_BOX_KINDS.has(kind) || BARRIER_KINDS.has(kind)) return out({ shape: "box", position: pos, size: [side, side], baseY, height, color: pal.furniture, estimated: est });
  if (kind === "station" || kind === "station_entrance") return out({ shape: "box", position: pos, size: [side, side], baseY, height, color: pal.rail, estimated: true });
  return out({ shape: "box", position: pos, size: [Math.min(side, 1), Math.min(side, 1)], baseY, height: Math.min(height, 1), color: pal.neutral, estimated: true,
    note: "unknown kind: small neutral proxy" });
}

/** "95 buildings, 75 roads, 42 lamp posts" ordered by count; at most `max` kinds then "+N more kinds". */
export function outdoorSummary(parts: OutdoorPart[], max = 6): string {
  const counts = new Map<string, number>();
  for (const p of parts) counts.set(p.kind, (counts.get(p.kind) ?? 0) + 1);
  const rows = [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  const shown = rows.slice(0, max).map(([k, n]) => `${n} ${n === 1 ? specFor(k).label.toLowerCase() : specFor(k).plural}`);
  if (rows.length > max) shown.push(`+${rows.length - max} more kinds`);
  return shown.join(", ");
}

// ---- triangle geometry (data frame) ----------------------------------------------------------------------------

export interface PartMesh { positions: number[]; indices: number[] }

class Mesh {
  positions: number[] = [];
  indices: number[] = [];
  private up: Up;
  constructor(up: Up) { this.up = up; }
  vert(a: number, h: number, b: number): number {
    this.positions.push(...(this.up === "y" ? [a, h, b] : [a, b, h]));
    return this.positions.length / 3 - 1;
  }
  tri(i: number, j: number, k: number) { this.indices.push(i, j, k); }
  quad(i: number, j: number, k: number, l: number) { this.tri(i, j, k); this.tri(i, k, l); }
  /** vertical wall quad between two ground points */
  wall(p: P2, q: P2, y0: number, y1: number) {
    this.quad(this.vert(p[0], y0, p[1]), this.vert(q[0], y0, q[1]), this.vert(q[0], y1, q[1]), this.vert(p[0], y1, p[1]));
  }
}

function extrude(m: Mesh, poly: P2[], y0: number, y1: number) {
  const tri = triangulate2D(poly);
  const top = poly.map((p) => m.vert(p[0], y1, p[1]));
  for (let i = 0; i < tri.length; i += 3) m.tri(top[tri[i]], top[tri[i + 1]], top[tri[i + 2]]);
  for (let i = 0; i < poly.length; i++) m.wall(poly[i], poly[(i + 1) % poly.length], y0, y1);
}

function box(m: Mesh, c: P2, sx: number, sz: number, y0: number, y1: number, angle = 0) {
  const cos = Math.cos(angle), sin = Math.sin(angle);
  const corners = ([[-1, -1], [1, -1], [1, 1], [-1, 1]] as P2[]).map(([ux, uz]): P2 => {
    const x = (ux * sx) / 2, z = (uz * sz) / 2;
    return [c[0] + x * cos - z * sin, c[1] + x * sin + z * cos];
  });
  extrude(m, corners, y0, y1);
}

/** Offset polyline: left/right edge points with mitred joints (mitre clamped so sharp corners do not spike). */
function ribbonEdges(pts: P2[], width: number): { left: P2[]; right: P2[] } {
  const half = width / 2;
  const nrm = (a: P2, b: P2): P2 => { const dx = b[0] - a[0], dz = b[1] - a[1], l = Math.hypot(dx, dz) || 1; return [-dz / l, dx / l]; };
  const left: P2[] = [], right: P2[] = [];
  pts.forEach((p, i) => {
    const n0 = i > 0 ? nrm(pts[i - 1], p) : null;
    const n1 = i < pts.length - 1 ? nrm(p, pts[i + 1]) : null;
    let n: P2 = (n0 ?? n1)!;
    let scale = 1;
    if (n0 && n1) {
      const sx = n0[0] + n1[0], sz = n0[1] + n1[1], l = Math.hypot(sx, sz);
      if (l > 1e-6) { n = [sx / l, sz / l]; scale = Math.min(2, 1 / Math.max(0.5, n[0] * n0[0] + n[1] * n0[1])); }
    }
    left.push([p[0] + n[0] * half * scale, p[1] + n[1] * half * scale]);
    right.push([p[0] - n[0] * half * scale, p[1] - n[1] * half * scale]);
  });
  return { left, right };
}

function ribbon(m: Mesh, pts: P2[], width: number, y0: number, y1: number) {
  const { left, right } = ribbonEdges(pts, width);
  const lt = left.map((p) => m.vert(p[0], y1, p[1])), rt = right.map((p) => m.vert(p[0], y1, p[1]));
  for (let i = 0; i < pts.length - 1; i++) m.quad(lt[i], lt[i + 1], rt[i + 1], rt[i]);
  for (let i = 0; i < pts.length - 1; i++) {
    m.wall(left[i], left[i + 1], y0, y1);
    m.wall(right[i + 1], right[i], y0, y1);
  }
}

function lowPolySphere(m: Mesh, c: P2, cy: number, r: number, ry: number, seg = 8, rings = 5) {
  const idx: number[][] = [];
  for (let j = 0; j <= rings; j++) {
    const phi = (Math.PI * j) / rings;
    const row: number[] = [];
    for (let i = 0; i < (j === 0 || j === rings ? 1 : seg); i++) {
      const th = (2 * Math.PI * i) / seg;
      row.push(m.vert(c[0] + r * Math.sin(phi) * Math.cos(th), cy + ry * Math.cos(phi), c[1] + r * Math.sin(phi) * Math.sin(th)));
    }
    idx.push(row);
  }
  for (let j = 0; j < rings; j++) {
    const a = idx[j], b = idx[j + 1];
    for (let i = 0; i < seg; i++) {
      const i1 = (i + 1) % seg;
      const a0 = a[a.length === 1 ? 0 : i], a1 = a[a.length === 1 ? 0 : i1], b0 = b[b.length === 1 ? 0 : i], b1 = b[b.length === 1 ? 0 : i1];
      if (a.length === 1) m.tri(a0, b1, b0);
      else if (b.length === 1) m.tri(a0, a1, b0);
      else m.quad(a0, a1, b1, b0);
    }
  }
}

/** Triangles for one part, in the data frame (`up` from OutdoorResult). Pure: same part, same arrays. */
export function partMesh(part: OutdoorPart, up: Up): PartMesh {
  const m = new Mesh(up);
  const y0 = part.baseY, y1 = part.baseY + part.height;
  const at = part.position;
  switch (part.shape) {
    case "extrusion":
      if (part.polygon) extrude(m, part.polygon, y0, y1);
      break;
    case "strip":
      if (part.polyline) ribbon(m, part.polyline, part.width, y0, y1);
      else if (part.polygon) extrude(m, part.polygon, y0, y1);
      if (part.piers && part.polyline) {
        const ends = [part.polyline[0], part.polyline[part.polyline.length - 1]];
        ends.forEach((c) => box(m, c, 0.8, Math.max(0.8, part.width * 0.5), 0, y0));
      }
      break;
    case "pole":
      if (at) {
        const t = Math.max(0.12, part.size[0] / 2);
        box(m, at, t, t, y0, y1);
        box(m, at, 0.5, 0.25, y1 - 0.15, y1 + 0.1); // lamp / sign head
      }
      break;
    case "box":
      if (at) box(m, at, part.size[0], part.size[1], y0, y1);
      break;
    case "tree":
      if (at) {
        const trunkTop = y0 + part.height * 0.4;
        box(m, at, 0.3, 0.3, y0, trunkTop);
        const r = Math.max(0.6, part.size[0] / 2);
        lowPolySphere(m, at, y0 + part.height - Math.min(r, part.height * 0.35), r, Math.min(r, part.height * 0.35));
      }
      break;
    case "stairs":
      if (part.polyline) {
        const steps = 5;
        for (let s = 0; s < part.polyline.length - 1; s++) {
          const a = part.polyline[s], b = part.polyline[s + 1];
          const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
          const len = Math.hypot(b[0] - a[0], b[1] - a[1]) / steps;
          for (let i = 0; i < steps; i++) {
            const t0 = (i + 0.5) / steps;
            box(m, [a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0], len, part.width, y0, y0 + (part.height * (i + 1)) / steps, ang);
          }
        }
      }
      break;
  }
  return { positions: m.positions, indices: m.indices };
}

/** Axis-aligned bounds of a part in the data frame, or null for an empty mesh. */
export function meshBounds(positions: ArrayLike<number>): { min: [number, number, number]; max: [number, number, number] } | null {
  if (positions.length < 3) return null;
  const min: [number, number, number] = [Infinity, Infinity, Infinity], max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], positions[i + k]); max[k] = Math.max(max[k], positions[i + k]); }
  return { min, max };
}
