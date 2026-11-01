/**
 * Pure (node-testable) model for what the viewer draws as "points" and for the capture-quality verdict.
 *
 * `structure_points` and `capture_quality` come from the reconstruction report. Both are optional and parsed
 * defensively: anything malformed is treated as absent, never an error.
 */

export type StructureKind = "corner" | "opening_corner" | "dimension";

export interface StructurePoint {
  id: string;
  kind: StructureKind;
  position: [number, number, number];
  entity_ids: string[];
  label?: string;
}

export type CaptureState = "sufficient" | "partial" | "insufficient";

export interface CaptureQuality {
  state: CaptureState;
  n_input: number;
  registered: number;
  points: number;
  duplicates_dropped: number;
  reasons: string[];
  advice: string[];
  message: string;
  /** Set when the verdict is about setup, not the photos (e.g. "workspace_path_too_long"); null otherwise. */
  cause?: string | null;
}

/** With no structure points the dense cloud is shown decimated to at most this many points. */
export const DECIMATED_CAP = 2000;

export type PointDisplayMode = "structure" | "full" | "decimated" | "none";

const KINDS: ReadonlySet<string> = new Set(["corner", "opening_corner", "dimension"]);
const STATES: ReadonlySet<string> = new Set(["sufficient", "partial", "insufficient"]);
const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const count = (v: unknown): number => (fin(v) && v >= 0 ? v : 0);
const strings = (v: unknown): string[] => (Array.isArray(v) ? v.filter((s): s is string => typeof s === "string" && s.length > 0) : []);

export function parseStructurePoints(raw: unknown): StructurePoint[] {
  if (!Array.isArray(raw)) return [];
  const out: StructurePoint[] = [];
  raw.forEach((r, i) => {
    if (!r || typeof r !== "object") return;
    const o = r as Record<string, unknown>;
    const p = o.position;
    if (!Array.isArray(p) || p.length < 3 || !fin(p[0]) || !fin(p[1]) || !fin(p[2])) return;
    out.push({
      id: typeof o.id === "string" && o.id ? o.id : `sp-${i}`,
      kind: typeof o.kind === "string" && KINDS.has(o.kind) ? (o.kind as StructureKind) : "dimension",
      position: [p[0], p[1], p[2]],
      entity_ids: strings(o.entity_ids),
      ...(typeof o.label === "string" && o.label ? { label: o.label } : {}),
    });
  });
  return out;
}

/** null when absent or without a recognised state (nothing is invented). */
export function parseCaptureQuality(raw: unknown): CaptureQuality | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.state !== "string" || !STATES.has(o.state)) return null;
  return {
    state: o.state as CaptureState,
    n_input: count(o.n_input),
    registered: count(o.registered),
    points: count(o.points),
    duplicates_dropped: count(o.duplicates_dropped),
    reasons: strings(o.reasons),
    advice: strings(o.advice),
    message: typeof o.message === "string" ? o.message : "",
    cause: typeof o.cause === "string" && o.cause ? o.cause : null,
  };
}

/** The photos could not be placed in one 3D space: per-photo geometry must not be drawn as a room. */
export const isUnplaced = (q: CaptureQuality | null | undefined): boolean => q?.state === "insufficient";

/** Multi-view reconstruction never ran for a setup reason: the photos were not judged and must not be blamed. */
export const isSetupFailure = (q: CaptureQuality | null | undefined): boolean => q?.cause === "workspace_path_too_long";

/** Headline for the "not placed" overlay. */
export const captureTitle = (q: CaptureQuality | null | undefined): string =>
  isSetupFailure(q) ? "Reconstruction could not start" : "The photos could not be placed in one 3D space";

/** Short badge label for the capture verdict. */
export function captureBadge(q: CaptureQuality): string {
  if (isSetupFailure(q)) return "Setup problem";
  return { sufficient: "Capture sufficient", partial: "Capture partial", insufficient: "Not placed in 3D" }[q.state];
}

/** Which point layer to build. The full cloud only on request; otherwise structure points, else a decimated cloud. */
export function choosePointDisplay(o: { structureCount: number; fullCloud: boolean; totalPoints: number }): PointDisplayMode {
  if (o.fullCloud && o.totalPoints > 0) return "full";
  if (o.structureCount > 0) return "structure";
  return o.totalPoints > 0 ? "decimated" : "none";
}

/** Even stride over an xyz Float32Array so at most `cap` points remain (the input is returned when already small). */
export function decimatePoints(positions: Float32Array, cap: number = DECIMATED_CAP): Float32Array {
  const total = Math.floor(positions.length / 3);
  if (total <= cap || cap <= 0) return positions;
  const stride = Math.ceil(total / cap);
  const out = new Float32Array(Math.ceil(total / stride) * 3);
  let dst = 0;
  for (let i = 0; i < total; i += stride) {
    out[dst++] = positions[i * 3];
    out[dst++] = positions[i * 3 + 1];
    out[dst++] = positions[i * 3 + 2];
  }
  return out;
}

/** Marker colours (hex) per structure kind. */
export const STRUCTURE_COLORS: Record<StructureKind, number> = {
  corner: 0xffd23f,
  opening_corner: 0x35d07f,
  dimension: 0xff6b6b,
};
