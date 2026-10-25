/**
 * Pure logic for the Generated panel and the Build dialog: opening
 * validation, stored-params -> form mapping, honest result text,
 * dependency labels. No React, no fetch (unit-tested by `npm run test:contract`).
 * The backend stays authoritative; this only gives early inline messages.
 */

import { districtSummary } from "./district-model.ts";

export type Wall ="north" | "south" | "east" | "west";
export const WALLS: Wall[] = ["north", "south", "east", "west"];

export interface OpeningDraft {
  wall: Wall;
  kind: "door" | "window";
  lateral_offset: number;
  width: number;
  bottom: number;
  top: number;
}

/** Wall length: north/south run along the room width (x), east/west along its depth (y). */
export function wallLength(wall: Wall, room: { width: number; depth: number }): number {
  return wall === "north" || wall === "south" ? room.width : room.depth;
}

/** Inline messages for one opening ([] = fine). Mirrors the backend rules. */
export function validateOpening(o: OpeningDraft, room: { width: number; depth: number }): string[] {
  const out: string[] = [];
  const nums = [o.lateral_offset, o.width, o.bottom, o.top];
  if (nums.some((n) => !Number.isFinite(n))) return ["Enter numbers for every field."];
  if (o.width <= 0) out.push("Width must be greater than 0.");
  if (o.lateral_offset < 0) out.push("Offset cannot be negative.");
  const len = wallLength(o.wall, room);
  if (o.lateral_offset + o.width > len + 1e-9) {
    out.push(`Offset + width (${round(o.lateral_offset + o.width)} m) exceeds the ${o.wall} wall length (${round(len)} m).`);
  }
  if (o.bottom < 0) out.push("Bottom cannot be below the floor (0).");
  if (o.top <= o.bottom) out.push("Top must be above bottom.");
  return out;
}

export function validateOpenings(list: OpeningDraft[], room: { width: number; depth: number }): string[][] {
  return list.map((o) => validateOpening(o, room));
}

export function hasOpeningErrors(errs: string[][]): boolean {
  return errs.some((e) => e.length > 0);
}

export function newOpening(kind: "door" | "window"): OpeningDraft {
  return kind === "door"
    ? { wall: "south", kind, lateral_offset: 0.5, width: 0.9, bottom: 0, top: 2.1 }
    : { wall: "north", kind, lateral_offset: 0.5, width: 1.2, bottom: 0.9, top: 2.1 };
}

function round(n: number): number {
  return Math.round(n * 1000) / 1000;
}

/** What the Generated list item looks like (GET /api/worlds/{id}/generated). */
export interface GeneratedItem {
  entity_id: string;
  kind: "room" | "corridor" | "city" | "apartment" | "building" | "district";
  name: string;
  params: Record<string, unknown>;
  origin: [number, number, number];
  attach: { to: string; side: string } | null;
  depends_on: string[];
}

function num(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

export interface RoomForm {
  width: number; depth: number; height: number; seed: number;
  openings: OpeningDraft[]; x: number; y: number;
}

/** Prefill a room edit form from its stored params (no invented defaults for present values). */
export function roomFormFrom(item: GeneratedItem): RoomForm {
  const p = item.params;
  const raw = Array.isArray(p.openings) ? (p.openings as Partial<OpeningDraft>[]) : [];
  return {
    width: num(p.width, 0), depth: num(p.depth, 0), height: num(p.height, 0), seed: num(p.seed, 0),
    openings: raw.map((o) => ({
      wall: (WALLS.includes(o.wall as Wall) ? o.wall : "north") as Wall,
      kind: o.kind === "window" ? "window" : "door",
      lateral_offset: num(o.lateral_offset, 0), width: num(o.width, 0),
      bottom: num(o.bottom, 0), top: num(o.top, 0),
    })),
    x: num(item.origin?.[0], 0), y: num(item.origin?.[1], 0),
  };
}

export interface CorridorForm {
  length: number; width: number; height: number; axis: "x" | "y";
  /** Rooms connected explicitly, excluding the attach target (the server re-adds it). */
  connects: string[]; attach: { to: string; side: string } | null; x: number; y: number;
}

export function corridorFormFrom(item: GeneratedItem): CorridorForm {
  const p = item.params;
  const connects = Array.isArray(p.connects) ? (p.connects as string[]) : [];
  return {
    length: num(p.length, 0), width: num(p.width, 0), height: num(p.height, 0),
    axis: p.axis === "y" ? "y" : "x",
    connects: item.attach ? connects.filter((c) => c !== item.attach!.to) : connects,
    attach: item.attach, x: num(item.origin?.[0], 0), y: num(item.origin?.[1], 0),
  };
}

export interface CityForm {
  blocks_x: number; blocks_y: number; lot_size: number; street_width: number;
  min_height: number; max_height: number; seed: number; x: number; y: number;
}

export function cityFormFrom(item: GeneratedItem): CityForm {
  const p = item.params;
  return {
    blocks_x: num(p.blocks_x, 0), blocks_y: num(p.blocks_y, 0), lot_size: num(p.lot_size, 0),
    street_width: num(p.street_width, 0), min_height: num(p.min_height, 0),
    max_height: num(p.max_height, 0), seed: num(p.seed, 0),
    x: num(item.origin?.[0], 0), y: num(item.origin?.[1], 0),
  };
}

/** Honest result line from the PUT response counts. */
export function resultLine(r: { changed: number; unchanged: number; removed: number }): string {
  return `${r.changed} changed - ${r.unchanged} unchanged - ${r.removed} removed`;
}

/** "6 rooms, 10 x 8 m" from stored apartment params (extent = bounding box of the stored rects). */
export function apartmentSummary(params: Record<string, unknown>): string {
  const rooms = Array.isArray(params.rooms) ? params.rooms.length : 0;
  const rects = (params.rects ?? {}) as Record<string, number[]>;
  const vals = Object.values(rects).filter((r) => Array.isArray(r) && r.length === 4);
  const round = (n: number) => Math.round(n * 10) / 10;
  const w = Math.max(0, ...vals.map((r) => num(r[0], 0) + num(r[2], 0)));
  const d = Math.max(0, ...vals.map((r) => num(r[1], 0) + num(r[3], 0)));
  return `${rooms} room${rooms === 1 ? "" : "s"}, ${round(w)} x ${round(d)} m`;
}

export const MAX_UI_STOREYS = 12;
export const H_MIN = 2.4;
export const H_MAX = 6;
export const DEFAULT_STOREY_HEIGHT = 3;

export interface BuildingForm {
  storeys: number;
  /** One height per storey (m); stored null means every storey is the 3 m default. */
  heights: number[];
}

export function buildingForm(item: GeneratedItem): BuildingForm {
  const p = item.params;
  const storeys = num(p.storeys, 1);
  const raw = Array.isArray(p.storey_heights) ? (p.storey_heights as unknown[]) : [];
  return { storeys, heights: Array.from({ length: storeys }, (_, i) => num(raw[i], DEFAULT_STOREY_HEIGHT)) };
}

/** Resize like the server does when only `storeys` changes: pad with the last height / truncate. */
export function resizeHeights(heights: number[], storeys: number): number[] {
  if (!Number.isInteger(storeys) || storeys < 1) return heights;
  const last = heights[heights.length - 1] ?? DEFAULT_STOREY_HEIGHT;
  return Array.from({ length: storeys }, (_, i) => heights[i] ?? last);
}

/** Inline messages for a building edit ([] = fine). */
export function validateBuildingForm(f: BuildingForm): string[] {
  const out: string[] = [];
  if (!Number.isInteger(f.storeys) || f.storeys < 1 || f.storeys > MAX_UI_STOREYS) {
    out.push(`Storeys must be a whole number from 1 to ${MAX_UI_STOREYS}.`);
  }
  if (f.heights.some((h) => !Number.isFinite(h) || h < H_MIN || h > H_MAX)) {
    out.push(`Each storey height must be between ${H_MIN} and ${H_MAX} m.`);
  }
  return out;
}

export function buildingPatch(f: BuildingForm): Record<string, unknown> {
  return { storeys: f.storeys, storey_heights: f.heights };
}

export interface ApartmentEditForm {
  seed: number;
  plan_index: number;
}

export function apartmentEditFrom(item: GeneratedItem): ApartmentEditForm {
  return { seed: num(item.params.seed, 0), plan_index: num(item.params.plan_index, 0) };
}

/** Only changed fields: any structural edit makes the server re-solve the layout. */
export function apartmentPatch(before: ApartmentEditForm, after: ApartmentEditForm): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (after.seed !== before.seed) out.seed = after.seed;
  if (after.plan_index !== before.plan_index) out.plan_index = after.plan_index;
  return out;
}

export function validateApartmentEdit(f: ApartmentEditForm): string[] {
  const out: string[] = [];
  if (!Number.isInteger(f.seed)) out.push("Seed must be a whole number.");
  if (!Number.isInteger(f.plan_index) || f.plan_index < 0 || f.plan_index > 9) {
    out.push("Plan index must be a whole number from 0 to 9.");
  }
  return out;
}

/** One-line key params for the list. */
export function summarize(item: GeneratedItem): string {
  const p = item.params;
  if (item.kind === "building") {
    const fp = Array.isArray(p.footprint) ? (p.footprint as unknown[]) : [];
    const n = num(p.storeys, 1);
    return `${n} storey${n === 1 ? "" : "s"}, ${num(fp[0], 0)} x ${num(fp[1], 0)} m`;
  }
  if (item.kind === "district") return districtSummary(p);
  if (item.kind === "apartment") return apartmentSummary(p);
  if (item.kind === "room") return `${num(p.width, 0)} x ${num(p.depth, 0)} x ${num(p.height, 0)} m`;
  if (item.kind === "corridor") return `${num(p.length, 0)} m long, ${num(p.width, 0)} m wide, along ${p.axis === "y" ? "y" : "x"}`;
  return `${num(p.blocks_x, 0)} x ${num(p.blocks_y, 0)} blocks, lot ${num(p.lot_size, 0)} m`;
}

function nameOf(items: GeneratedItem[], entityId: string): string {
  return items.find((i) => i.entity_id === entityId)?.name ?? entityId;
}

/** "attached east of A", "connects A, B" and the reverse "used by: Hall" lines. */
export function dependencyLines(item: GeneratedItem, items: GeneratedItem[]): string[] {
  const out: string[] = [];
  if (item.attach) out.push(`Attached to ${nameOf(items, item.attach.to)} (${item.attach.side} side)`);
  const others = item.depends_on.filter((d) => d !== item.attach?.to);
  if (others.length) out.push(`Connects ${others.map((d) => nameOf(items, d)).join(", ")}`);
  const users = items.filter((i) => i.entity_id !== item.entity_id && i.depends_on.includes(item.entity_id));
  if (users.length) out.push(`Used by ${users.map((u) => u.name).join(", ")}`);
  return out;
}

/** Message for a duplicate-name 409 on create. */
export function duplicateNameMessage(kind: string, name: string): string {
  return `A ${kind} named ${name} already exists - edit it in Generated`;
}

/** Pull dependent names out of the delete-409 text ("has dependents: a, b; delete..."). */
export function parseDependents(message: string): string[] {
  const m = /has dependents:\s*([^;]+);/.exec(message);
  return m ? m[1].split(",").map((s) => s.trim()).filter(Boolean) : [];
}
