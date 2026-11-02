/**
 * Room results summary (pure, node-testable). Every number is read from the compiled WorldIR; nothing is defaulted.
 * Mirrors what perception/architecture/room_semantics.py writes on the room entity (`footprint_polygon_m` in the
 * plan basis, `floor_area_m2`, `ceiling_height_min_m`) and, when present, per-entity `dimensions{value,unit,state}`.
 */
import type { WorldIR } from "@/types/worldir";
import { describeScale, type ScaleStageDto } from "../api/scale-model.ts";
import { isUnplaced, type CaptureQuality } from "../viewport/structure-points-model.ts";

export type DimState = "MEASURED" | "ESTIMATED" | "INFERRED" | "UNKNOWN" | string;

export interface DimensionRow {
  key: "length" | "width" | "height" | "area";
  label: string;
  /** null = not available (no value is invented) */
  value: number | null;
  /** "m" / "m²" only for a metric world; "model units" otherwise. */
  unit: string;
  /** state badge text; null when the world is unscaled (no state can be claimed) */
  state: DimState | null;
}

export interface RoomSummary {
  roomId: string | null;
  /** true only when the world scale is metric: only then are units metres */
  metric: boolean;
  scaleKind: "relative" | "estimated" | "measured" | "unknown";
  scaleLabel: string;
  dimensions: DimensionRow[];
  /** explicit sentence shown when dimensions are not in metres */
  unitNote: string | null;
  counts: { walls: number; doors: number; windows: number; furniture: number };
  /** false when the photos could not be placed in one space: the wall/door/window counts then come from unplaced per-photo boxes and are shown as "—" */
  countsReliable: boolean;
  /** the engine's capture-quality verdict when it reported one (drives the verdict line and the capture hint) */
  captureQuality: CaptureQuality | null;
  /** walls the engine never saw (provenance UNOBSERVED / MISSING / UNSEEN), or the API's number when it gives one */
  unobservedWalls: number;
  warnings: string[];
  /** what to capture next: only derived from real signals (unobserved walls, unplaced photos, or an API hint) */
  captureHint: string | null;
}

type Props = Record<string, unknown>;
const UNOBSERVED = new Set(["UNOBSERVED", "MISSING", "UNSEEN"]);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const round2 = (v: number): number => Math.round(v * 100) / 100;

export interface SummaryExtras {
  /** current version's scale stage from `status.model.scale`; falls back to `world.metadata.scale` */
  scale?: ScaleStageDto | null;
  /** photos the model used / placed (from status.model) */
  imagesUsed?: number | null;
  imagesRegistered?: number | null;
  /** capture-coverage from the API when it provides it: its numbers and hint override the derived ones */
  coverage?: { unobserved_walls?: number | null; weak_walls?: number | null; hint?: string | null } | null;
  /** the version's capture-quality verdict (already parsed); replaces the generic unplaced-photo wording */
  captureQuality?: CaptureQuality | null;
}

/** The dimension-state shape `dimensions{<key>: {value, unit, state}}` when the compiler wrote it. */
function entityDim(cp: Props, ...keys: string[]): { value: number | null; state: DimState | null } | null {
  const d = cp.dimensions;
  if (!d || typeof d !== "object") return null;
  for (const k of keys) {
    const r = (d as Record<string, unknown>)[k];
    if (r && typeof r === "object") {
      const o = r as Props;
      return { value: num(o.value), state: typeof o.state === "string" ? o.state : null };
    }
  }
  return null;
}

export function buildRoomSummary(world: WorldIR | null | undefined, extras: SummaryExtras = {}): RoomSummary | null {
  if (!world?.entities) return null;
  const entities = Object.values(world.entities);

  const scaleStage: ScaleStageDto | null = extras.scale ?? ((world.metadata?.scale as ScaleStageDto | undefined) ?? null);
  const view = describeScale(scaleStage);
  const metric = view.kind === "estimated" || view.kind === "measured";
  const scaleLabel =
    view.kind === "relative" ? "Relative scale (model units)" : view.kind === "unknown" ? "Scale not reported" : view.label;
  const scaleDimState = scaleStage?.dimension_state ?? (view.kind === "estimated" ? "ESTIMATED" : view.kind === "measured" ? "MEASURED" : null);

  const room =
    entities.find((e) => e.type === "room" && typeof (e.custom_properties as Props | undefined)?.superseded_by !== "string") ?? null;
  const cp = (room?.custom_properties ?? {}) as Props;

  // Overall plan extent of the measured footprint (any shape): the two sides of its bounding box in the plan basis.
  const poly = Array.isArray(cp.footprint_polygon_m) ? (cp.footprint_polygon_m as unknown[]) : [];
  const xs: number[] = [];
  const ys: number[] = [];
  for (const p of poly) {
    if (Array.isArray(p) && num(p[0]) !== null && num(p[1]) !== null) {
      xs.push(p[0] as number);
      ys.push(p[1] as number);
    }
  }
  let longSide: number | null = null;
  let shortSide: number | null = null;
  if (xs.length >= 3) {
    const a = Math.max(...xs) - Math.min(...xs);
    const b = Math.max(...ys) - Math.min(...ys);
    longSide = Math.max(a, b);
    shortSide = Math.min(a, b);
  }

  const lenUnit = metric ? "m" : "model units";
  const areaUnit = metric ? "m²" : "model units²";
  const mk = (key: DimensionRow["key"], label: string, derived: number | null, unit: string, ...entityKeys: string[]): DimensionRow => {
    const own = entityDim(cp, ...entityKeys);
    const value = own?.value ?? derived;
    // A state is only claimed when the compiler wrote one, or the world is metric (then the scale's own state applies).
    const state = value === null ? null : own?.state ?? (metric ? scaleDimState ?? "UNKNOWN" : null);
    return { key, label, value: value === null ? null : round2(value), unit, state };
  };
  const dimensions: DimensionRow[] = room
    ? [
        mk("length", "Length", longSide, lenUnit, "length", "width"),
        mk("width", "Width", shortSide, lenUnit, "depth", "width"),
        mk("height", "Ceiling height", num(cp.ceiling_height_min_m), lenUnit, "height"),
        mk("area", "Floor area", num(cp.floor_area_m2), areaUnit, "area"),
      ]
    : [];

  const unitNote = metric
    ? null
    : view.kind === "relative"
      ? "Model units, not metres. Set the scale to get real dimensions."
      : "Units unknown: this version did not report a scale, so these numbers are not in metres.";

  const ofType = (t: string) => entities.filter((e) => e.type === t);
  const derivedUnobserved = ofType("wall").filter((w) => typeof w.provenance === "string" && UNOBSERVED.has(w.provenance.toUpperCase())).length;
  const counts = {
    walls: ofType("wall").length,
    doors: ofType("door").length,
    windows: ofType("window").length,
    furniture: entities.filter((e) => (e.semantic_labels ?? []).includes("furniture_candidate")).length,
  };

  const cq = extras.captureQuality ?? null;
  const countsReliable = !isUnplaced(cq);
  // Walls derived from unplaced per-photo boxes say nothing about the room, so no wall coverage claims are made then.
  const missingWalls = countsReliable ? num(extras.coverage?.unobserved_walls) ?? derivedUnobserved : 0;
  const used = extras.imagesUsed ?? null;
  const placed = extras.imagesRegistered ?? null;
  const unplaced = used !== null && placed !== null && placed < used ? used - placed : 0;
  const weakWalls = countsReliable ? num(extras.coverage?.weak_walls) ?? 0 : 0;
  const warnings: string[] = [];
  if (missingWalls > 0) warnings.push(`${missingWalls} wall${missingWalls === 1 ? " was" : "s were"} not seen in your photos.`);
  if (weakWalls > 0) warnings.push(`${weakWalls} wall${weakWalls === 1 ? " was" : "s were"} seen only weakly (from one angle or at a grazing view).`);
  // The engine's verdict (shown separately) replaces the generic unplaced-photo line.
  if (!cq && unplaced > 0) warnings.push(`${unplaced} of ${used} photos could not be placed in the model.`);

  let captureHint: string | null = countsReliable ? extras.coverage?.hint ?? null : null;
  if (!countsReliable) {
    captureHint = cq?.advice[0] ?? null;
  } else if (!captureHint && (missingWalls > 0 || weakWalls > 0)) {
    captureHint = "Photograph the unseen or weakly seen wall(s) head-on from the opposite side of the room, overlapping your earlier photos by 60% or more.";
  } else if (!captureHint && cq && cq.state !== "sufficient") {
    captureHint = cq.advice[0] ?? null;
  } else if (!captureHint && !cq && unplaced > 0) {
    captureHint = "Retake the unplaced views with more overlap and a clear shared patch (a corner, a door frame) with a placed photo.";
  }

  return { roomId: room?.id ?? null, metric, scaleKind: view.kind, scaleLabel, dimensions, unitNote, counts, countsReliable, captureQuality: cq, unobservedWalls: missingWalls, warnings, captureHint };
}
