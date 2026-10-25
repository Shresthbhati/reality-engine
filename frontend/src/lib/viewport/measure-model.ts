/**
 * Measurement view-model: pure, no React/three. Mirrors world_ir/measure.py for
 * live previews (the API recomputes every saved value and is the source of truth)
 * and decides, from the world's scale state, what unit a value may be labelled with.
 *
 * Honesty rules encoded here:
 *  - a value is labelled metres ONLY for a metric-anchored world or authored
 *    (procedural, provenance GENERATED) geometry; otherwise it is "model units";
 *  - an uncertainty exists only when the data supplied one; authored geometry is
 *    exact (0); everything else reads "not estimated".
 */

export type MeasureKind = "distance" | "height" | "area" | "elevation_difference";
export type Vec3 = [number, number, number];
export type ScaleBasis = "metric" | "authored" | "unknown";

export interface MeasureToolDef {
  kind: MeasureKind | "volume" | "road_width" | "building_footprint";
  label: string;
  /** null = available; otherwise the tooltip explaining why it is disabled. */
  unavailable: string | null;
  hint: string;
}

const NOT_YET = "Not available yet";

export const MEASURE_TOOLS: MeasureToolDef[] = [
  { kind: "distance", label: "Distance", unavailable: null, hint: "Click 2 points" },
  { kind: "height", label: "Height", unavailable: null, hint: "Click 2 points (vertical difference)" },
  { kind: "area", label: "Area", unavailable: null, hint: "Click corners on a horizontal plane, then Finish" },
  { kind: "volume", label: "Volume", unavailable: NOT_YET, hint: "" },
  { kind: "road_width", label: "Road Width", unavailable: NOT_YET, hint: "" },
  { kind: "building_footprint", label: "Building Footprint", unavailable: NOT_YET, hint: "" },
  { kind: "elevation_difference", label: "Elevation Difference", unavailable: null, hint: "Click 2 points (signed)" },
];

export const KIND_LABEL: Record<MeasureKind, string> = {
  distance: "Distance",
  height: "Height",
  area: "Area",
  elevation_difference: "Elevation Difference",
};

export interface UnitContext {
  basis: ScaleBasis;
  /** Length unit label: "m" or "model units". */
  lengthUnit: string;
  areaUnit: string;
  /** Method text recorded with a measurement. */
  method: string;
  /** Visible badge text, or null when no caveat applies. */
  badge: string | null;
  /** Known uncertainty (same unit as the value), or null = not estimated. */
  uncertainty: number | null;
}

interface WorldLike {
  metadata?: { scale?: { state?: string; method?: string } };
  entities?: Record<string, { provenance?: string }>;
}

export function resolveUnitContext(world: WorldLike | null | undefined): UnitContext {
  const state = world?.metadata?.scale?.state?.toLowerCase();
  if (state === "metric") {
    return {
      basis: "metric",
      lengthUnit: "m",
      areaUnit: "m²",
      method: "picked points, metric-anchored reconstruction",
      badge: null,
      uncertainty: null,
    };
  }
  const entities = Object.values(world?.entities ?? {});
  if (entities.length > 0 && entities.every((e) => e.provenance === "GENERATED")) {
    return {
      basis: "authored",
      lengthUnit: "m",
      areaUnit: "m²",
      method: "authored geometry (exact)",
      badge: null,
      uncertainty: 0,
    };
  }
  return {
    basis: "unknown",
    lengthUnit: "model units",
    areaUnit: "model units²",
    method: "picked points, scale unknown (model units)",
    badge: "scale unknown",
    uncertainty: null,
  };
}

export function unitFor(kind: MeasureKind, ctx: Pick<UnitContext, "lengthUnit" | "areaUnit">): string {
  return kind === "area" ? ctx.areaUnit : ctx.lengthUnit;
}

/** Unit label for a stored measurement row (server `unit` field), never inventing metres. */
export function unitLabelFromRow(unit: string): string {
  switch (unit) {
    case "m": return "m";
    case "m2": return "m²";
    case "model_units2": return "model units²";
    default: return "model units";
  }
}

export function formatValue(value: number, unit: string, digits = 3): string {
  if (!Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)} ${unit}`;
}

export function formatUncertainty(u: number | null | undefined, unit: string): string {
  if (u == null || !Number.isFinite(u)) return "Uncertainty: not estimated";
  if (u === 0) return "Uncertainty: exact (authored)";
  return `Uncertainty: ± ${u.toFixed(3)} ${unit}`;
}

// ---- preview math (mirror of world_ir/measure.py; server is authoritative) ----

export const POINTS_NEEDED: Record<MeasureKind, number> = {
  distance: 2,
  height: 2,
  elevation_difference: 2,
  area: 3,
};

export function computePreview(kind: MeasureKind, pts: Vec3[]): number | null {
  if (pts.length < POINTS_NEEDED[kind]) return null;
  const [a, b] = pts;
  if (kind === "distance") return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  if (kind === "height") return Math.abs(b[1] - a[1]);
  if (kind === "elevation_difference") return b[1] - a[1];
  // Newell polygon area (valid for the planar polygons the viewport builds).
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    nx += p[1] * q[2] - p[2] * q[1];
    ny += p[2] * q[0] - p[0] * q[2];
    nz += p[0] * q[1] - p[1] * q[0];
  }
  return Math.hypot(nx, ny, nz) / 2;
}

export function centroid(pts: Vec3[]): Vec3 {
  const n = Math.max(1, pts.length);
  return [
    pts.reduce((s, p) => s + p[0], 0) / n,
    pts.reduce((s, p) => s + p[1], 0) / n,
    pts.reduce((s, p) => s + p[2], 0) / n,
  ];
}

/** Histogram of values into `bins` equal bins (the "Measurement Profile"). */
export function histogram(values: number[], bins = 8): { lo: number; hi: number; count: number }[] {
  const v = values.filter(Number.isFinite);
  if (v.length === 0) return [];
  const min = Math.min(...v), max = Math.max(...v);
  const width = max > min ? (max - min) / bins : 1;
  const out = Array.from({ length: max > min ? bins : 1 }, (_, i) => ({ lo: min + i * width, hi: min + (i + 1) * width, count: 0 }));
  for (const x of v) out[Math.min(out.length - 1, Math.floor((x - min) / width))].count++;
  return out;
}

/** Counts of entity provenance values -- the honest "supporting evidence" for a world. */
export function provenanceSummary(world: WorldLike | null | undefined): string {
  const counts = new Map<string, number>();
  for (const e of Object.values(world?.entities ?? {})) {
    const k = e.provenance ?? "UNKNOWN";
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  if (counts.size === 0) return "no entities";
  return [...counts.entries()].map(([k, n]) => `${n} ${k}`).join(", ");
}
