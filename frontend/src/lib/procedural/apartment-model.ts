/**
 * Pure logic for the Apartment builder: templates -> rows, early validation,
 * request bodies, plan -> SVG geometry, adjacency summary. No React, no fetch
 * (unit-tested by `npm run test:contract`). The backend stays authoritative;
 * this mirrors procedural/layout_solver.py only to give early inline messages.
 */

export const APARTMENT_KINDS = [
  "living", "kitchen", "bedroom", "bathroom", "hallway", "balcony", "storage", "office",
] as const;
export type ApartmentKind = (typeof APARTMENT_KINDS)[number];
export const MAX_APARTMENT_ROOMS = 20;
export const MAX_SIDE_M = 100;

export type AdjacencyLevel = "required" | "preferred";

export interface RoomRow {
  id: string;
  kind: ApartmentKind;
  min_area: number;
  max_area: number;
}

export interface AdjacencyDraft {
  a: string;
  b: string;
  level: AdjacencyLevel;
}

export interface ApartmentDraft {
  name: string;
  rows: RoomRow[];
  entrance: string;
  /** null = let the solver derive the footprint. */
  footprint: { width: number; depth: number } | null;
  adjacency: AdjacencyDraft[];
  seed: number;
}

export interface ApartmentTemplate {
  key: string;
  label: string;
  rows: RoomRow[];
}

const row = (id: string, kind: ApartmentKind, min_area: number, max_area: number): RoomRow => ({
  id, kind, min_area, max_area,
});

/** Template rows (solver-checked as feasible, with templateAdjacency, on an auto footprint). */
export const APARTMENT_TEMPLATES: ApartmentTemplate[] = [
  {
    key: "studio",
    label: "Studio",
    rows: [row("living", "living", 20, 30), row("kitchen", "kitchen", 6, 10),
      row("bathroom", "bathroom", 4, 6), row("hallway", "hallway", 3, 6)],
  },
  {
    key: "2-bedroom",
    label: "2-bedroom",
    rows: [row("living", "living", 18, 26), row("kitchen", "kitchen", 8, 12),
      row("bedroom-1", "bedroom", 11, 15), row("bedroom-2", "bedroom", 10, 14),
      row("bathroom", "bathroom", 4, 6), row("hallway", "hallway", 4, 8)],
  },
  {
    key: "3-bedroom",
    label: "3-bedroom",
    // Areas are generous because every room must touch the hallway (solver-checked).
    rows: [row("living", "living", 22, 32), row("kitchen", "kitchen", 10, 15),
      row("bedroom-1", "bedroom", 12, 18), row("bedroom-2", "bedroom", 11, 16),
      row("bedroom-3", "bedroom", 10, 14), row("bathroom", "bathroom", 5, 8),
      row("bathroom-2", "bathroom", 4, 6), row("hallway", "hallway", 12, 20)],
  },
];

/** Fresh copy of a template's rows (never the shared objects). */
export function templateRows(key: string): RoomRow[] {
  const t = APARTMENT_TEMPLATES.find((c) => c.key === key);
  return t ? t.rows.map((r) => ({ ...r })) : [];
}

/** Sensible default adjacency: the hallway touches every bedroom, bathroom and living room; living touches the kitchen (all required). */
export function templateAdjacency(rows: RoomRow[]): AdjacencyDraft[] {
  const hall = rows.find((r) => r.kind === "hallway");
  const out: AdjacencyDraft[] = [];
  if (hall) {
    for (const r of rows) {
      if (["bedroom", "bathroom", "living"].includes(r.kind)) out.push({ a: hall.id, b: r.id, level: "required" });
    }
  }
  const living = rows.find((r) => r.kind === "living");
  const kitchen = rows.find((r) => r.kind === "kitchen");
  if (living && kitchen) out.push({ a: living.id, b: kitchen.id, level: "required" });
  return out;
}

/** The entrance to use: the chosen one if it still exists, else the hallway, else the first row. */
export function effectiveEntrance(rows: RoomRow[], chosen: string): string {
  if (rows.some((r) => r.id === chosen)) return chosen;
  return (rows.find((r) => r.kind === "hallway") ?? rows[0])?.id ?? "";
}

/** Next unused id like "bedroom-2" for a new row. */
export function nextRoomId(rows: RoomRow[], kind: ApartmentKind): string {
  const ids = new Set(rows.map((r) => r.id));
  if (!ids.has(kind)) return kind;
  for (let n = 2; ; n++) if (!ids.has(`${kind}-${n}`)) return `${kind}-${n}`;
}

/** Inline messages ([] = fine). */
export function validateApartment(d: ApartmentDraft): string[] {
  const out: string[] = [];
  if (!d.name.trim()) out.push("Enter an apartment name.");
  if (d.rows.length === 0) out.push("Add at least one room.");
  if (d.rows.length > MAX_APARTMENT_ROOMS) out.push(`An apartment is limited to ${MAX_APARTMENT_ROOMS} rooms.`);
  const ids = d.rows.map((r) => r.id.trim());
  if (ids.some((i) => i === "")) out.push("Every room needs an id.");
  const dupes = ids.filter((id, i) => id !== "" && ids.indexOf(id) !== i);
  if (dupes.length) out.push(`Room ids must be unique (repeated: ${[...new Set(dupes)].join(", ")}).`);
  for (const r of d.rows) {
    const label = r.id.trim() || "(unnamed room)";
    if (!Number.isFinite(r.min_area) || !Number.isFinite(r.max_area)) {
      out.push(`${label}: enter numbers for min and max area.`);
    } else if (r.min_area <= 0 || r.max_area <= 0) {
      out.push(`${label}: areas must be greater than 0.`);
    } else if (r.min_area > r.max_area) {
      out.push(`${label}: min area (${r.min_area}) is larger than max area (${r.max_area}).`);
    }
  }
  if (d.rows.length > 0 && !ids.includes(d.entrance)) out.push("Choose an entrance room from the list.");
  if (d.footprint) {
    const { width, depth } = d.footprint;
    if (![width, depth].every((v) => Number.isFinite(v) && v > 0 && v <= MAX_SIDE_M)) {
      out.push(`Footprint sides must be between 0 and ${MAX_SIDE_M} m.`);
    }
  }
  if (!Number.isInteger(d.seed)) out.push("Seed must be a whole number.");
  const seen = new Set<string>();
  for (const a of d.adjacency) {
    if (!ids.includes(a.a) || !ids.includes(a.b)) {
      out.push(`Adjacency ${a.a} - ${a.b} names a room that is not in the list.`);
    } else if (a.a === a.b) {
      out.push(`Adjacency ${a.a} - ${a.b} joins a room to itself.`);
    } else {
      const key = [a.a, a.b].sort().join("\u0000");
      if (seen.has(key)) out.push(`Adjacency between ${a.a} and ${a.b} is listed twice.`);
      seen.add(key);
    }
  }
  return out;
}

/** Body for POST .../apartments/solve and (plus plan_index) .../apartments. */
export interface ApartmentBody {
  name: string;
  rooms: { id: string; kind: string; min_area: number; max_area: number }[];
  entrance: string;
  footprint: [number, number] | null;
  adjacency: [string, string, AdjacencyLevel][];
  seed: number;
}

export function toRequestBody(d: ApartmentDraft): ApartmentBody {
  return {
    name: d.name.trim(),
    rooms: d.rows.map((r) => ({
      id: r.id.trim(), kind: r.kind, min_area: r.min_area, max_area: r.max_area,
    })),
    entrance: d.entrance,
    footprint: d.footprint ? [d.footprint.width, d.footprint.depth] : null,
    adjacency: d.adjacency.map((a) => [a.a, a.b, a.level]),
    seed: d.seed,
  };
}

/** Solver plan as returned by the API. */
export interface ApartmentPlan {
  rects: Record<string, [number, number, number, number]>;
  score: number;
  satisfied: [string, string, string][];
  unsatisfied: [string, string, string][];
}

/** "2 of 3 required adjacencies met" -- totals come from the spec, met from the plan. */
export function adjacencySummary(plan: ApartmentPlan, adjacency: AdjacencyDraft[]): string {
  const required = adjacency.filter((a) => a.level === "required");
  const total = required.length;
  const met = required.filter((a) =>
    plan.satisfied.some((s) => (s[0] === a.a && s[1] === a.b) || (s[0] === a.b && s[1] === a.a)),
  ).length;
  return `${met} of ${total} required adjacencies met`;
}

export interface SvgRoom {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  /** Label centre. */
  cx: number;
  cy: number;
  areaText: string;
  circulation: boolean;
  /** Room is too small for a label inside; show the id only. */
  tiny: boolean;
}

export interface PlanSvg {
  width: number;
  height: number;
  rooms: SvgRoom[];
}

/**
 * Scale a plan to an SVG of `viewWidth` px wide (to scale, uniform factor).
 * The extent is the solver footprint when given, else the rects' bounding box.
 */
export function planToSvg(
  plan: ApartmentPlan,
  kinds: Record<string, string>,
  footprint: [number, number] | null,
  viewWidth = 220,
): PlanSvg {
  const entries = Object.entries(plan.rects);
  const maxX = Math.max(...entries.map(([, r]) => r[0] + r[2]), footprint?.[0] ?? 0, 0.001);
  const maxY = Math.max(...entries.map(([, r]) => r[1] + r[3]), footprint?.[1] ?? 0, 0.001);
  const k = viewWidth / maxX;
  const rooms = entries.map(([id, [x, y, w, d]]) => ({
    id,
    x: x * k, y: y * k, w: w * k, h: d * k,
    cx: (x + w / 2) * k, cy: (y + d / 2) * k,
    areaText: `${Math.round(w * d * 10) / 10} m²`,
    circulation: kinds[id] === "hallway",
    tiny: w * k < 38 || d * k < 26,
  }));
  return { width: viewWidth, height: maxY * k, rooms };
}
