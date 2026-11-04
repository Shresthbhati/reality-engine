/**
 * Room shell model: turns the measured interior semantics (room footprint polygon, per-edge wall heights, ceiling
 * layers, windows, furniture candidates) into plain triangle arrays. No three.js here, so it is unit-testable under
 * Node; three-scene.ts only wraps the arrays in THREE geometries.
 *
 * Everything drawn comes from measured WorldIR properties written by perception/architecture/room_semantics.py. If a
 * room carries no footprint polygon the model returns no parts and the viewer keeps its legacy bounding boxes.
 */

import type { WorldIR } from "@/types/worldir";

export type ShellKind =
  | "floor"
  | "wall"
  | "ceiling"
  | "false_ceiling"
  | "ceiling_step"
  | "window_glass"
  | "window_frame"
  | "furniture";

export type EvidenceState =
  | "OBSERVED"
  | "PARTIAL"
  | "OCCLUDED"
  | "INFERRED"
  | "PRIOR_GENERATED"
  | "CONFLICTED"
  | "UNKNOWN";

export interface ShellPart {
  id: string;
  kind: ShellKind;
  /** entity this part visualises (for selection / inspection) */
  entityId: string;
  positions: number[];
  indices: number[];
  /** sRGB 0-255 */
  color: [number, number, number];
  opacity: number;
  /** false when `color` is a TYPE_FALLBACK default: nothing was measured, so it must never read as a measured colour */
  colorMeasured: boolean;
  /** entity confidence 0-1 straight from the WorldIR; null = the entity carries none (never invented) */
  confidence: number | null;
  /** entity provenance straight from the WorldIR (OBSERVED / INFERRED / ESTIMATED ...); null = not stated */
  provenance: string | null;
  confidenceState: "scored" | "unknown" | "unobserved";
  /** Blueprint 14 Evidence state classification */
  evidenceState?: EvidenceState;
  /** M2: image-projected atlas for this part's entity (perception/materials/surface_appearance.py). Absent = flat/striped fallback. */
  texture?: ShellTexture;
}

/** Reference to a stored atlas PNG; alpha=0 texels are unobserved (never inpainted) and must stay transparent. */
export interface ShellTexture {
  uri: string;
  hash: string | null;
  /** per-vertex atlas coordinates (2 per vertex, same order as positions), t=0 at the atlas bottom row */
  uvs: number[];
}

/** Fields filled in by one annotate pass after the geometry is built. */
type RawPart = Omit<ShellPart, "confidence" | "provenance" | "confidenceState" | "texture">;

/** The only thing the viewer modes may differ in. Geometry (ids, positions, indices) is identical. */
export type ShellRenderMode = "solid" | "wireframe" | "both" | "confidence";

type V3 = [number, number, number];
type Props = Record<string, unknown>;

const TYPE_FALLBACK: Record<ShellKind, V3> = {
  floor: [150, 125, 95],
  wall: [200, 194, 184],
  ceiling: [235, 233, 228],
  false_ceiling: [245, 243, 238],
  ceiling_step: [220, 215, 205],
  window_glass: [150, 195, 240],
  window_frame: [235, 235, 235],
  furniture: [140, 105, 70],
};

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3): number => Math.hypot(a[0], a[1], a[2]);
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function isV3(p: unknown): p is V3 {
  return Array.isArray(p) && p.length === 3 && p.every((n) => typeof n === "number" && Number.isFinite(n));
}

function polygon3(value: unknown): V3[] | null {
  if (!Array.isArray(value) || value.length < 3) return null;
  const pts = value.filter(isV3) as V3[];
  return pts.length === value.length ? pts : null;
}

function rgbOrNull(value: unknown): V3 | null {
  return Array.isArray(value) && value.length === 3 && value.every((n) => typeof n === "number") ? (value as V3) : null;
}

interface Resolved { color: V3; measured: boolean }

function meanColor(world: WorldIR, ids: string[], fallback: V3): Resolved {
  const cols: V3[] = [];
  for (const id of ids) {
    const c = rgbOrNull((world.entities?.[id]?.custom_properties as Props | undefined)?.mean_color_rgb);
    if (c) cols.push(c);
  }
  if (!cols.length) return { color: fallback, measured: false };
  return { color: [0, 1, 2].map((i) => Math.round(cols.reduce((s, c) => s + c[i], 0) / cols.length)) as V3, measured: true };
}

/** A directly stored colour wins; else the mean of the entities' measured colours; else the flagged type default. */
function resolveColor(direct: unknown, world: WorldIR, ids: string[], fallback: V3): Resolved {
  const d = rgbOrNull(direct);
  return d ? { color: d, measured: true } : meanColor(world, ids, fallback);
}

/** Ear-clipping triangulation of a simple (possibly concave) polygon given in 2D. Returns index triples. */
export function triangulate2D(pts: Array<[number, number]>): number[] {
  const n = pts.length;
  if (n < 3) return [];
  let area = 0;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % n];
    area += x1 * y2 - x2 * y1;
  }
  const idx = Array.from({ length: n }, (_, i) => i);
  if (area < 0) idx.reverse();
  const orient = (o: number, a: number, b: number) =>
    (pts[a][0] - pts[o][0]) * (pts[b][1] - pts[o][1]) - (pts[a][1] - pts[o][1]) * (pts[b][0] - pts[o][0]);
  const inside = (p: number, a: number, b: number, c: number) =>
    orient(a, b, p) >= -1e-12 && orient(b, c, p) >= -1e-12 && orient(c, a, p) >= -1e-12;
  const out: number[] = [];
  let guard = 0;
  while (idx.length > 3 && guard++ < n * n) {
    let clipped = false;
    for (let k = 0; k < idx.length; k++) {
      const a = idx[(k + idx.length - 1) % idx.length];
      const b = idx[k];
      const c = idx[(k + 1) % idx.length];
      if (orient(a, b, c) <= 1e-12) continue; // reflex or degenerate
      if (idx.some((p) => p !== a && p !== b && p !== c && inside(p, a, b, c))) continue;
      out.push(a, b, c);
      idx.splice(k, 1);
      clipped = true;
      break;
    }
    if (!clipped) return []; // not a simple polygon: emit nothing rather than garbage
  }
  if (idx.length === 3) out.push(idx[0], idx[1], idx[2]);
  return out;
}

/** Same basis the backend used (perception/architecture/room_semantics.py::_basis). */
function planBasis(up: V3): { e1: V3; e2: V3 } {
  const u = mul(up, 1 / (len(up) || 1));
  const ref: V3 = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = cross(u, ref);
  const e1n = mul(e1, 1 / len(e1));
  return { e1: e1n, e2: cross(u, e1n) };
}

function polygonPart(
  id: string, kind: ShellKind, entityId: string, poly: V3[], e1: V3, e2: V3, color: Resolved, opacity: number,
): RawPart | null {
  const tri = triangulate2D(poly.map((p) => [dot(p, e1), dot(p, e2)] as [number, number]));
  if (!tri.length) return null;
  return { id, kind, entityId, positions: poly.flat(), indices: tri, color: color.color, colorMeasured: color.measured, opacity };
}

function rectPart(
  id: string, kind: ShellKind, entityId: string, o: V3, du: V3, dv: V3, color: Resolved, opacity: number,
): RawPart {
  const p1 = add(o, du);
  const p2 = add(p1, dv);
  const p3 = add(o, dv);
  return {
    id, kind, entityId, positions: [...o, ...p1, ...p2, ...p3], indices: [0, 1, 2, 0, 2, 3],
    color: color.color, colorMeasured: color.measured, opacity,
  };
}

interface WindowOnEdge { s0: number; s1: number; v0: number; v1: number; entityId: string }

/** Horizontal (plan) distance from a point to the infinite line through `a` along unit `dir`. */
function planOffset(c: V3, a: V3, dir: V3, up: V3): number {
  const d = sub(c, a);
  const dh = sub(d, mul(up, dot(d, up)));
  return len(sub(dh, mul(dir, dot(dh, dir))));
}

function windowsOnEdge(
  windows: Array<{ id: string; custom_properties?: Props }>, a: V3, dir: V3, L: number, up: V3, floorLevel: V3, h: number,
): WindowOnEdge[] {
  const out: WindowOnEdge[] = [];
  for (const w of windows) {
    const wp = w.custom_properties ?? {};
    if (!isV3(wp.bounds_min) || !isV3(wp.bounds_max)) continue;
    const lo = wp.bounds_min;
    const hi = wp.bounds_max;
    const corners: V3[] = [];
    for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) corners.push([x, y, z]);
    const ss = corners.map((c) => dot(sub(c, a), dir));
    const vs = corners.map((c) => dot(sub(c, floorLevel), up));
    const s0 = Math.min(...ss);
    const s1 = Math.max(...ss);
    if (Math.max(...corners.map((c) => planOffset(c, a, dir, up))) > 0.35 || s1 < 0 || s0 > L) continue;
    const v0 = Math.max(0, Math.min(...vs));
    const v1 = Math.min(h, Math.max(...vs));
    if (v1 - v0 > 0.05 && s1 - s0 > 0.05) out.push({ s0: Math.max(0, s0), s1: Math.min(L, s1), v0, v1, entityId: w.id });
  }
  return out.sort((p, q) => p.s0 - q.s0);
}


// ---- solid wall slabs (perception/architecture/wall_slabs.py): walls built from WALL entities, not from the footprint ----

interface SlabOpening { entityId: string; kind: string; offset: number; width: number; sill: number; height: number }

interface Slab {
  entity: { id: string; custom_properties?: Props };
  start: V3; dir: V3; out: V3; up: V3; length: number; height: number; thickness: number;
  openings: SlabOpening[];
}

function numberOr(v: unknown): number | null { return typeof v === "number" && Number.isFinite(v) ? v : null; }

function slabOf(e: { id: string; type?: string; custom_properties?: Props }): Slab | null {
  const ws = e.custom_properties?.wall_slab as Props | undefined;
  if (e.type !== "wall" || !ws || !isV3(ws.start_xyz_m) || !isV3(ws.direction) || !isV3(ws.outward) || !isV3(ws.up)) return null;
  const length = numberOr(ws.length_m), height = numberOr(ws.height_m), thickness = numberOr(ws.thickness_m);
  if (length === null || height === null || thickness === null) return null;
  const openings: SlabOpening[] = [];
  for (const o of Array.isArray(ws.openings) ? (ws.openings as Props[]) : []) {
    const [offset, width, sill, h] = [o.offset_m, o.width_m, o.sill_m, o.height_m].map(numberOr);
    if (offset === null || width === null || sill === null || h === null) continue;
    openings.push({ entityId: String(o.entity_id ?? ""), kind: String(o.kind ?? "opening"), offset, width, sill, height: h });
  }
  return { entity: e, start: ws.start_xyz_m, dir: ws.direction, out: ws.outward, up: ws.up, length, height, thickness, openings };
}

/** Axis-aligned (s0, s1, v0, v1) boxes tiling a wall rectangle minus its rectangular openings (mirror of room_layout.wall_pieces). */
export function wallPieces(length: number, height: number, openings: Array<{ offset: number; width: number; sill: number; height: number }>):
  Array<[number, number, number, number]> {
  const holes = openings.map((o) => [Math.max(0, o.offset), Math.min(length, o.offset + o.width), Math.max(0, o.sill), Math.min(height, o.sill + o.height)])
    .filter((h) => h[1] - h[0] > 1e-4 && h[3] - h[2] > 1e-4);
  const cuts = Array.from(new Set([0, length, ...holes.flatMap((h) => [h[0], h[1]])])).sort((a, b) => a - b);
  const out: Array<[number, number, number, number]> = [];
  for (let i = 0; i + 1 < cuts.length; i++) {
    const [s0, s1] = [cuts[i], cuts[i + 1]];
    if (s1 - s0 < 1e-4) continue;
    const mid = (s0 + s1) / 2;
    const spans = holes.filter((h) => h[0] <= mid && mid <= h[1]).map((h) => [h[2], h[3]]).sort((a, b) => a[0] - b[0]);
    let v = 0;
    for (const [h0, h1] of spans) {
      if (h0 > v + 1e-4) out.push([s0, s1, v, h0]);
      v = Math.max(v, h1);
    }
    if (height > v + 1e-4) out.push([s0, s1, v, height]);
  }
  return out;
}

const BOX_QUADS = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];

/** Convex hexahedron (bottom ring 0-3, top ring 4-7) with outward-wound triangles. */
function boxPart(id: string, kind: ShellKind, entityId: string, c: V3[], color: Resolved, opacity: number): RawPart {
  const centre = mul(c.reduce((a, p) => add(a, p), [0, 0, 0] as V3), 1 / c.length);
  const indices: number[] = [];
  for (const q of BOX_QUADS) {
    for (const t of [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]) {
      const n = cross(sub(c[t[1]], c[t[0]]), sub(c[t[2]], c[t[0]]));
      indices.push(...(dot(n, sub(c[t[0]], centre)) < 0 ? [t[0], t[2], t[1]] : t));
    }
  }
  return { id, kind, entityId, positions: c.flat(), indices, color: color.color, colorMeasured: color.measured, opacity };
}

function slabParts(world: WorldIR, roomId: string, sl: Slab, fallbackColor: Resolved): RawPart[] {
  const parts: RawPart[] = [];
  const id = sl.entity.id;
  const direct = rgbOrNull((sl.entity.custom_properties as Props | undefined)?.mean_color_rgb);
  const color: Resolved = direct ? { color: direct, measured: true } : fallbackColor;
  const at = (s: number, t: number, v: number): V3 => add(add(add(sl.start, mul(sl.dir, s)), mul(sl.out, t)), mul(sl.up, v));
  wallPieces(sl.length, sl.height, sl.openings).forEach(([s0, s1, v0, v1], k) => {
    const T = sl.thickness;
    parts.push(boxPart(`${id}:box${k}`, "wall", id, [at(s0, 0, v0), at(s1, 0, v0), at(s1, T, v0), at(s0, T, v0),
      at(s0, 0, v1), at(s1, 0, v1), at(s1, T, v1), at(s0, T, v1)], color, 1));
  });
  const flagged = (k: ShellKind): Resolved => ({ color: TYPE_FALLBACK[k], measured: false });
  for (const o of sl.openings) {
    const o0 = at(o.offset, sl.thickness / 2, o.sill);
    if (o.kind === "window") {
      parts.push(rectPart(`${id}:glass:${o.entityId}`, "window_glass", o.entityId || id, o0, mul(sl.dir, o.width), mul(sl.up, o.height),
        flagged("window_glass"), 0.35));
    }
    const t = 0.05;
    const bars: Array<[V3, V3, V3]> = [
      [at(o.offset, 0, o.sill + o.height - t), mul(sl.dir, o.width), mul(sl.up, t)],
      [at(o.offset, 0, o.sill), mul(sl.dir, t), mul(sl.up, o.height)],
      [at(o.offset + o.width - t, 0, o.sill), mul(sl.dir, t), mul(sl.up, o.height)],
    ];
    if (o.kind !== "door") bars.push([at(o.offset, 0, o.sill), mul(sl.dir, o.width), mul(sl.up, t)]);
    bars.forEach(([oo, du, dv], k) => parts.push(rectPart(`${id}:frame:${o.entityId}:${k}`, "window_frame", o.entityId || id, oo, du, dv,
      flagged("window_frame"), 1)));
  }
  return parts;
}

/** Build every shell part for every room that carries a measured footprint polygon, or solid wall slabs. */
export function buildRoomShellParts(world: WorldIR | null | undefined): ShellPart[] {
  if (!world?.entities) return [];
  return annotate(world, buildRawParts(world));
}

const UNOBSERVED = new Set(["UNOBSERVED", "MISSING", "UNSEEN"]);

/** Browser URL for an atlas uri. `artifact://<sha256>` is served by the world's artifact route; plain URLs pass through. */
export function atlasUrl(uri: string, worldId: string | undefined): string | null {
  if (/^(https?:|data:image\/|blob:|\/)/.test(uri)) return uri;
  const m = /^artifact:\/\/([0-9a-f]{64})$/.exec(uri);
  return m && worldId ? `/api/worlds/${encodeURIComponent(worldId)}/artifacts/${m[1]}` : null;
}

const TEXTURED_KINDS = new Set<ShellKind>(["wall", "floor", "ceiling", "false_ceiling", "ceiling_step"]);

/**
 * The entity's atlas reference + frame -> a projector from world position to atlas (s,t). Null unless the appearance
 * carries a uri and a complete, non-degenerate frame (so a malformed reference falls back instead of drawing garbage).
 */
export function atlasProjector(appearance: unknown): { uri: string; hash: string | null; project: (p: V3) => [number, number] } | null {
  const tex = (appearance as Props | undefined)?.texture as Props | null | undefined;
  const fr = tex?.frame as Props | undefined;
  if (!tex || typeof tex.uri !== "string" || !tex.uri || !fr) return null;
  const { origin, u_axis: ua, v_axis: va } = fr;
  const ur = fr.u_range_m as unknown, vr = fr.v_range_m as unknown;
  if (!isV3(origin) || !isV3(ua) || !isV3(va)) return null;
  const rng = (r: unknown): [number, number] | null =>
    Array.isArray(r) && r.length === 2 && r.every((n) => typeof n === "number" && Number.isFinite(n)) && r[1] > r[0] ? [r[0], r[1]] : null;
  const u = rng(ur), v = rng(vr);
  if (!u || !v) return null;
  return {
    uri: tex.uri,
    hash: typeof tex.hash === "string" ? tex.hash : null,
    project: (p) => {
      const d = sub(p, origin);
      return [(dot(d, ua) - u[0]) / (u[1] - u[0]), (dot(d, va) - v[0]) / (v[1] - v[0])];
    },
  };
}

function textureFor(world: WorldIR, p: RawPart): ShellTexture | undefined {
  if (!TEXTURED_KINDS.has(p.kind)) return undefined;
  const proj = atlasProjector((world.entities[p.entityId]?.custom_properties as Props | undefined)?.appearance);
  if (!proj) return undefined;
  const uvs: number[] = [];
  for (let i = 0; i + 2 < p.positions.length; i += 3) uvs.push(...proj.project([p.positions[i], p.positions[i + 1], p.positions[i + 2]]));
  return { uri: proj.uri, hash: proj.hash, uvs };
}


export const EVIDENCE_PALETTE: Record<EvidenceState, V3> = {
  OBSERVED: [16, 185, 129],
  PARTIAL: [245, 158, 11],
  OCCLUDED: [139, 92, 246],
  INFERRED: [59, 130, 246],
  PRIOR_GENERATED: [14, 165, 233],
  CONFLICTED: [239, 68, 68],
  UNKNOWN: [107, 114, 128],
};

export function resolveEvidenceState(
  evidenceState?: string | null,
  provenance?: string | null,
  confidence?: number | null,
): EvidenceState {
  const raw = (evidenceState || provenance || '').toUpperCase().trim();
  if (raw === 'OBSERVED' || raw === 'RECONSTRUCTED' || raw === 'MEASURED') {
    if (typeof confidence === 'number' && confidence < 0.5) return 'PARTIAL';
    return 'OBSERVED';
  }
  if (raw === 'PARTIAL' || raw === 'PARTIALLY_OBSERVED') return 'PARTIAL';
  if (raw === 'OCCLUDED' || raw === 'UNOBSERVED') return 'OCCLUDED';
  if (raw === 'INFERRED' || raw === 'ESTIMATED') return 'INFERRED';
  if (raw === 'GENERATED' || raw === 'PRIOR_GENERATED' || raw === 'SYNTHESIZED') return 'PRIOR_GENERATED';
  if (raw === 'CONFLICTED' || raw === 'DISPUTED') return 'CONFLICTED';
  if (typeof confidence === 'number') {
    if (confidence >= 0.75) return 'OBSERVED';
    if (confidence >= 0.45) return 'PARTIAL';
    if (confidence > 0) return 'CONFLICTED';
  }
  return 'UNKNOWN';
}

function annotate(world: WorldIR, raw: RawPart[]): ShellPart[] {
  return raw.map((p) => {
    const e = world.entities[p.entityId];
    const c = typeof e?.confidence === "number" ? e.confidence
      : typeof e?.uncertainty?.confidence === "number" ? e.uncertainty.confidence : null;
    const provenance = typeof e?.provenance === "string" ? e.provenance : null;
    const cpr = (e?.custom_properties ?? {}) as Props;
    const rawEvState = typeof cpr.evidence_state === "string" ? cpr.evidence_state : null;
    const evState = resolveEvidenceState(rawEvState, provenance, c);
    const unobserved = provenance !== null && UNOBSERVED.has(provenance.toUpperCase());
    const texture = textureFor(world, p);
    // Not-captured entities read as such: generated = translucent, map-imported = slightly muted.
    const origin = cpr.generated_by ? 0.45 : cpr.imported === true || cpr.source === "OpenStreetMap" ? 0.7 : 1;
    return {
      ...p,
      opacity: Math.min(p.opacity, origin),
      confidence: c,
      provenance,
      evidenceState: evState,
      confidenceState: unobserved ? "unobserved" : c === null ? "unknown" : "scored",
      ...(texture ? { texture } : {}),
    };
  });
}

function buildRawParts(world: WorldIR): RawPart[] {
  const parts: RawPart[] = [];
  const entities = Object.values(world.entities);
  const windows = entities.filter((e) => e.type === "window");
  const slabWalls = entities.map((e) => slabOf(e as never)).filter((x): x is Slab => x !== null);
  const boundaryIds = (cp: Props) => (Array.isArray(cp.boundary_element_ids) ? (cp.boundary_element_ids as string[]) : []).filter((id) => world.entities[id]?.type === "wall");
  const polygonSlabParts = (world: WorldIR, roomId: string): RawPart[] => {
    const out: RawPart[] = [];
    for (const e of entities) {
      const sp = (e.custom_properties ?? {}) as Props;
      const slab = sp.slab as Props | undefined;
      if (sp.room_id !== roomId || !slab || !isV3(slab.up) || (e.type !== "floor" && e.type !== "ceiling")) continue;
      const pts = polygon3(slab.polygon_xyz_m);
      if (!pts) continue;
      const { e1, e2 } = planBasis(slab.up);
      const kind: ShellKind = e.type === "floor" ? "floor" : sp.ceiling_role === "false_ceiling" ? "false_ceiling" : "ceiling";
      const part = polygonPart(`${e.id}:top`, kind, e.id, pts, e1, e2, resolveColor(sp.mean_color_rgb, world, [], TYPE_FALLBACK[kind]), 1);
      if (part) out.push(part);
    }
    return out;
  };
  for (const room of entities) {
    if (room.type !== "room") continue;
    const cp = (room.custom_properties ?? {}) as Props;
    if (typeof cp.superseded_by === "string") continue; // a fragment of another room, not a room of its own
    const poly0 = polygon3(cp.footprint_polygon_xyz_m);
    const basis = cp.footprint_basis as { up?: unknown } | undefined;
    const slabs = slabWalls.filter((w) => w.entity.custom_properties?.room_id === room.id);
    const hasFootprint = !!poly0 && isV3(basis?.up) && typeof cp.floor_height_m === "number";
    if (!hasFootprint && !slabs.length) continue;
    if (!hasFootprint) {
      // no footprint polygon: the room is still drawn, from its solid wall / floor / ceiling slabs alone
      const wallCol = resolveColor(cp.wall_color_rgb, world, boundaryIds(cp), TYPE_FALLBACK.wall);
      slabs.forEach((sl) => parts.push(...slabParts(world, room.id, sl, wallCol)));
      parts.push(...polygonSlabParts(world, room.id));
      continue;
    }
    const poly = poly0 as V3[];
    const up = (basis as { up: unknown }).up as V3;
    const { e1, e2 } = planBasis(up);
    const floorLevel = mul(up, (cp.floor_height_m as number));
    const ceilingLayers = Array.isArray(cp.ceiling_layers) ? (cp.ceiling_layers as Props[]) : [];
    const edgeHeights = Array.isArray(cp.footprint_edge_heights_m) ? (cp.footprint_edge_heights_m as unknown[]) : [];
    const fallbackH = typeof cp.ceiling_height_min_m === "number" ? cp.ceiling_height_min_m : 2.4;
    const boundary = Array.isArray(cp.boundary_element_ids) ? (cp.boundary_element_ids as string[]) : [];
    const byType = (t: string) => boundary.filter((id) => world.entities[id]?.type === t);

    const floor = polygonPart(`${room.id}:floor`, "floor", byType("floor")[0] ?? room.id, poly, e1, e2,
      resolveColor(cp.floor_color_rgb, world, byType("floor"), TYPE_FALLBACK.floor), 1);
    if (floor) parts.push(floor);

    // walls, split around the windows that sit on them
    const wallColor = resolveColor(cp.wall_color_rgb, world, byType("wall"), TYPE_FALLBACK.wall);
    const flagged = (k: ShellKind): Resolved => ({ color: TYPE_FALLBACK[k], measured: false });
    const wallEntity = byType("wall")[0] ?? room.id;
    if (slabs.length) slabs.forEach((sl) => parts.push(...slabParts(world, room.id, sl, wallColor)));
    for (let i = 0; i < (slabs.length ? 0 : poly.length); i++) {
      const a = poly[i];
      const edge = sub(poly[(i + 1) % poly.length], a);
      const L = len(edge);
      if (L < 1e-6) continue;
      const dir = mul(edge, 1 / L);
      const h = typeof edgeHeights[i] === "number" ? (edgeHeights[i] as number) : fallbackH;
      let k = 0;
      const seg = (s0: number, s1: number, v0: number, v1: number) => {
        if (s1 - s0 < 1e-4 || v1 - v0 < 1e-4) return;
        parts.push(rectPart(`${room.id}:wall${i}:${k++}`, "wall", wallEntity, add(add(a, mul(dir, s0)), mul(up, v0)),
          mul(dir, s1 - s0), mul(up, v1 - v0), wallColor, 1));
      };
      let cursor = 0;
      for (const w of windowsOnEdge(windows, a, dir, L, up, floorLevel, h)) {
        seg(cursor, w.s0, 0, h);
        seg(w.s0, w.s1, 0, w.v0);
        seg(w.s0, w.s1, w.v1, h);
        const o = add(add(a, mul(dir, w.s0)), mul(up, w.v0));
        const W = w.s1 - w.s0;
        const Hh = w.v1 - w.v0;
        parts.push(rectPart(`${room.id}:glass:${w.entityId}`, "window_glass", w.entityId, o, mul(dir, W), mul(up, Hh),
          flagged("window_glass"), 0.35));
        const t = 0.05;
        const frame: Array<[V3, V3, V3]> = [
          [o, mul(dir, W), mul(up, t)],
          [add(o, mul(up, Hh - t)), mul(dir, W), mul(up, t)],
          [o, mul(dir, t), mul(up, Hh)],
          [add(o, mul(dir, W - t)), mul(dir, t), mul(up, Hh)],
        ];
        frame.forEach(([oo, du, dv]) =>
          parts.push(rectPart(`${room.id}:frame:${w.entityId}:${k++}`, "window_frame", w.entityId, oo, du, dv,
            flagged("window_frame"), 1)));
        cursor = w.s1;
      }
      seg(cursor, L, 0, h);
    }

    // ceiling layers, plus the vertical step faces between layers of different height
    const tops = ceilingLayers.map((l) => (typeof l.height_above_floor_m === "number" ? l.height_above_floor_m : 0));
    const topH = tops.length ? Math.max(...tops) : fallbackH;
    const nearBoundary = (x: V3) => poly.some((v, vi) => {
      const e = sub(poly[(vi + 1) % poly.length], v);
      const el = len(e) || 1;
      const t = Math.max(0, Math.min(1, dot(sub(x, v), e) / (el * el)));
      const d = sub(sub(x, v), mul(e, t));
      return len(sub(d, mul(up, dot(d, up)))) < 0.06;
    });
    ceilingLayers.forEach((l, li) => {
      const lp = polygon3(l.polygon_xyz_m);
      if (!lp) return;
      const kind: ShellKind = l.role === "false_ceiling" ? "false_ceiling" : "ceiling";
      const ids = Array.isArray(l.entity_ids) ? (l.entity_ids as string[]) : [];
      const part = polygonPart(`${room.id}:ceil${li}`, kind, ids[0] ?? room.id, lp, e1, e2,
        resolveColor(l.color_rgb, world, ids, TYPE_FALLBACK[kind]), 1);
      if (part) parts.push(part);
      if (tops[li] < topH - 0.05) {
        lp.forEach((p, j) => {
          const q = lp[(j + 1) % lp.length];
          // sample along the edge; only the runs that are NOT on a room wall are a free drop face
          const N = 96;
          const free = Array.from({ length: N + 1 }, (_, t) => !nearBoundary(add(p, mul(sub(q, p), t / N))));
          let start = -1;
          for (let t = 0; t <= N + 1; t++) {
            if (t <= N && free[t]) {
              if (start < 0) start = t;
            } else if (start >= 0) {
              const pa = add(p, mul(sub(q, p), Math.max(0, start - 0.5) / N));
              const pb = add(p, mul(sub(q, p), Math.min(N, t - 0.5) / N));
              if (len(sub(pb, pa)) > 0.05) {
                parts.push(rectPart(`${room.id}:step${li}:${j}:${start}`, "ceiling_step", ids[0] ?? room.id, pa, sub(pb, pa),
                  mul(up, topH - tops[li]), flagged("ceiling_step"), 1));
              }
              start = -1;
            }
          }
        });
      }
    });

    // furniture candidates of this room: oriented plan boxes
    for (const f of entities) {
      const fp = (f.custom_properties ?? {}) as Props;
      if (!(f.semantic_labels ?? []).includes("furniture_candidate") || fp.room_id !== room.id) continue;
      const size = fp.plan_obb_size_m;
      const centre = fp.plan_obb_centre_m;
      const base = fp.base_height_above_floor_m;
      const height = fp.height_m;
      const angle = fp.plan_obb_angle_deg;
      if (!Array.isArray(size) || !Array.isArray(centre) || typeof base !== "number" || typeof height !== "number" ||
          typeof angle !== "number") continue;
      const th = (angle * Math.PI) / 180;
      const hw = (size[0] as number) / 2;
      const hh = (size[1] as number) / 2;
      const corner = (sx: number, sy: number, vv: number): V3 => {
        const a2 = (centre[0] as number) + sx * hw * Math.cos(th) - sy * hh * Math.sin(th);
        const b2 = (centre[1] as number) + sx * hw * Math.sin(th) + sy * hh * Math.cos(th);
        return add(add(mul(e1, a2), mul(e2, b2)), mul(up, (cp.floor_height_m as number) + vv));
      };
      const c = [corner(-1, -1, base), corner(1, -1, base), corner(1, 1, base), corner(-1, 1, base),
        corner(-1, -1, base + height), corner(1, -1, base + height), corner(1, 1, base + height), corner(-1, 1, base + height)];
      parts.push({
        id: `${f.id}:box`, kind: "furniture", entityId: f.id, positions: c.flat(),
        indices: [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7],
        color: rgbOrNull(fp.mean_color_rgb) ?? TYPE_FALLBACK.furniture,
        colorMeasured: rgbOrNull(fp.mean_color_rgb) !== null, opacity: 1,
      });
    }
  }
  return parts;
}

// ---- Material per render mode (the ONLY thing the modes change) -------------------------------------------------

export interface PartMaterial {
  color: V3;
  /** diagonal stripe overlay */
  hatch: boolean;
  opacity: number;
  /** draw the part's atlas (colour white, alpha=0 texels transparent); never in confidence mode */
  textured: boolean;
}

// Defaults mirror the --conf-low / --warning / --conf-high / --conf-unknown tokens in globals.css; the viewer overrides
// them from the live CSS custom properties (see three-scene.ts readPalette).
export interface ConfidencePalette { low: V3; mid: V3; high: V3; unknown: V3 }
export const DEFAULT_CONF_PALETTE: ConfidencePalette = {
  low: [224, 82, 74], mid: [224, 164, 88], high: [76, 175, 98], unknown: [107, 112, 104],
};
const CONF_UNKNOWN: V3 = DEFAULT_CONF_PALETTE.unknown; // neutral grey: the entity states no confidence, so it is never coloured
const CONF_UNOBSERVED: V3 = [96, 52, 128]; // dark violet: explicitly not observed
const ESTIMATED = new Set(["INFERRED", "ESTIMATED", "SYNTHESIZED"]);

export function confidenceRamp(c: number, pal: ConfidencePalette = DEFAULT_CONF_PALETTE): V3 {
  const t = Math.max(0, Math.min(1, c));
  const RAMP: Array<[number, V3]> = [[0, pal.low], [0.5, pal.mid], [1, pal.high]];
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1];
      const [t1, c1] = RAMP[i];
      const k = (t - t0) / (t1 - t0);
      return [0, 1, 2].map((j) => Math.round(c0[j] + (c1[j] - c0[j]) * k)) as V3;
    }
  }
  return RAMP[RAMP.length - 1][1];
}

export function isEstimated(part: ShellPart): boolean {
  return part.provenance !== null && ESTIMATED.has(part.provenance.toUpperCase());
}

export function partMaterial(part: ShellPart, mode: ShellRenderMode, pal: ConfidencePalette = DEFAULT_CONF_PALETTE): PartMaterial {
  if (mode !== "confidence") {
    const textured = !!part.texture;
    // a textured part is not "colour not measured": the flat colour only shows through where the atlas has no texel
    return { color: part.color, hatch: !part.colorMeasured && !textured, opacity: part.opacity, textured };
  }
  if (part.evidenceState && part.evidenceState in EVIDENCE_PALETTE && part.confidenceState !== "scored") {
    const evColor = EVIDENCE_PALETTE[part.evidenceState];
    const isHatched = part.evidenceState === "INFERRED" || part.evidenceState === "PRIOR_GENERATED" || isEstimated(part);
    return { color: evColor, hatch: isHatched, opacity: part.opacity, textured: false };
  }
  const color = part.confidenceState === "unobserved" ? CONF_UNOBSERVED
    : part.confidenceState === "unknown" || part.confidence === null ? pal.unknown
    : confidenceRamp(part.confidence, pal);
  return { color, hatch: isEstimated(part), opacity: part.opacity, textured: false };
}

export interface LegendItem { label: string; color: V3; hatch: boolean }

export function legendFor(mode: ShellRenderMode): LegendItem[] {
  if (mode === "confidence") {
    return [
      { label: "Observed", color: EVIDENCE_PALETTE.OBSERVED, hatch: false },
      { label: "Partial", color: EVIDENCE_PALETTE.PARTIAL, hatch: false },
      { label: "Occluded", color: EVIDENCE_PALETTE.OCCLUDED, hatch: false },
      { label: "Inferred", color: EVIDENCE_PALETTE.INFERRED, hatch: true },
      { label: "Prior / Generated", color: EVIDENCE_PALETTE.PRIOR_GENERATED, hatch: true },
      { label: "Conflicted", color: EVIDENCE_PALETTE.CONFLICTED, hatch: false },
      { label: "Unknown", color: EVIDENCE_PALETTE.UNKNOWN, hatch: false },
    ];
  }
  if (mode === "wireframe") return [];
  return [{ label: "Striped = colour not measured (type default)", color: [200, 194, 184], hatch: true }];
}
