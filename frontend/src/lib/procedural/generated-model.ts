/**
 * Pure logic for the Generated panel and the Build dialog: opening
 * validation, stored-params -> form mapping, honest result text,
 * dependency labels. No React, no fetch (unit-tested by `npm run test:contract`).
 * The backend stays authoritative; this only gives early inline messages.
 */

export type Wall = "north" | "south" | "east" | "west";
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
  kind: "room" | "corridor" | "city";
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

/** One-line key params for the list. */
export function summarize(item: GeneratedItem): string {
  const p = item.params;
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
