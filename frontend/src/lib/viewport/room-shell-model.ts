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
}

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

function meanColor(world: WorldIR, ids: string[], fallback: V3): V3 {
  const cols: V3[] = [];
  for (const id of ids) {
    const c = rgbOrNull((world.entities?.[id]?.custom_properties as Props | undefined)?.mean_color_rgb);
    if (c) cols.push(c);
  }
  if (!cols.length) return fallback;
  return [0, 1, 2].map((i) => Math.round(cols.reduce((s, c) => s + c[i], 0) / cols.length)) as V3;
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
  id: string, kind: ShellKind, entityId: string, poly: V3[], e1: V3, e2: V3, color: V3, opacity: number,
): ShellPart | null {
  const tri = triangulate2D(poly.map((p) => [dot(p, e1), dot(p, e2)] as [number, number]));
  if (!tri.length) return null;
  return { id, kind, entityId, positions: poly.flat(), indices: tri, color, opacity };
}

function rectPart(
  id: string, kind: ShellKind, entityId: string, o: V3, du: V3, dv: V3, color: V3, opacity: number,
): ShellPart {
  const p1 = add(o, du);
  const p2 = add(p1, dv);
  const p3 = add(o, dv);
  return { id, kind, entityId, positions: [...o, ...p1, ...p2, ...p3], indices: [0, 1, 2, 0, 2, 3], color, opacity };
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

/** Build every shell part for every room that carries a measured footprint polygon. */
export function buildRoomShellParts(world: WorldIR | null | undefined): ShellPart[] {
  if (!world?.entities) return [];
  const parts: ShellPart[] = [];
  const entities = Object.values(world.entities);
  const windows = entities.filter((e) => e.type === "window");
  for (const room of entities) {
    if (room.type !== "room") continue;
    const cp = (room.custom_properties ?? {}) as Props;
    if (typeof cp.superseded_by === "string") continue; // a fragment of another room, not a room of its own
    const poly = polygon3(cp.footprint_polygon_xyz_m);
    const basis = cp.footprint_basis as { up?: unknown } | undefined;
    if (!poly || !isV3(basis?.up) || typeof cp.floor_height_m !== "number") continue;
    const up = basis.up as V3;
    const { e1, e2 } = planBasis(up);
    const floorLevel = mul(up, cp.floor_height_m);
    const ceilingLayers = Array.isArray(cp.ceiling_layers) ? (cp.ceiling_layers as Props[]) : [];
    const edgeHeights = Array.isArray(cp.footprint_edge_heights_m) ? (cp.footprint_edge_heights_m as unknown[]) : [];
    const fallbackH = typeof cp.ceiling_height_min_m === "number" ? cp.ceiling_height_min_m : 2.4;
    const boundary = Array.isArray(cp.boundary_element_ids) ? (cp.boundary_element_ids as string[]) : [];
    const byType = (t: string) => boundary.filter((id) => world.entities[id]?.type === t);

    const floor = polygonPart(`${room.id}:floor`, "floor", byType("floor")[0] ?? room.id, poly, e1, e2,
      rgbOrNull(cp.floor_color_rgb) ?? meanColor(world, byType("floor"), TYPE_FALLBACK.floor), 1);
    if (floor) parts.push(floor);

    // walls, split around the windows that sit on them
    const wallColor = rgbOrNull(cp.wall_color_rgb) ?? meanColor(world, byType("wall"), TYPE_FALLBACK.wall);
    const wallEntity = byType("wall")[0] ?? room.id;
    for (let i = 0; i < poly.length; i++) {
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
          TYPE_FALLBACK.window_glass, 0.35));
        const t = 0.05;
        const frame: Array<[V3, V3, V3]> = [
          [o, mul(dir, W), mul(up, t)],
          [add(o, mul(up, Hh - t)), mul(dir, W), mul(up, t)],
          [o, mul(dir, t), mul(up, Hh)],
          [add(o, mul(dir, W - t)), mul(dir, t), mul(up, Hh)],
        ];
        frame.forEach(([oo, du, dv]) =>
          parts.push(rectPart(`${room.id}:frame:${w.entityId}:${k++}`, "window_frame", w.entityId, oo, du, dv,
            TYPE_FALLBACK.window_frame, 1)));
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
        rgbOrNull(l.color_rgb) ?? meanColor(world, ids, TYPE_FALLBACK[kind]), 1);
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
                  mul(up, topH - tops[li]), TYPE_FALLBACK.ceiling_step, 1));
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
        color: rgbOrNull(fp.mean_color_rgb) ?? TYPE_FALLBACK.furniture, opacity: 1,
      });
    }
  }
  return parts;
}
