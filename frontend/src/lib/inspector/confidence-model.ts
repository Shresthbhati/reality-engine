/**
 * Confidence reading for the inspector bars. Pure (no React).
 * A reading comes only from a confidence value the entity (or its real children) records.
 * No value -> tone "unknown", ratio null: rendered as "Not assessed" in neutral grey, never a colour, never a number.
 */

export type ConfidenceTone = "high" | "medium" | "low" | "unknown";

export interface ConfidenceReading {
  label: "High" | "Medium" | "Low" | "Not assessed";
  tone: ConfidenceTone;
  /** 0..1, or null when nothing was recorded */
  ratio: number | null;
}

export const HIGH_AT = 0.75;
export const MEDIUM_AT = 0.45;

export function readConfidence(value: unknown): ConfidenceReading {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return { label: "Not assessed", tone: "unknown", ratio: null };
  }
  const ratio = Math.min(1, Math.max(0, value));
  if (ratio >= HIGH_AT) return { label: "High", tone: "high", ratio };
  if (ratio >= MEDIUM_AT) return { label: "Medium", tone: "medium", ratio };
  return { label: "Low", tone: "low", ratio };
}

/** Design token for a tone. Unknown is always the neutral token. */
export const TONE_TOKEN: Record<ConfidenceTone, string> = {
  high: "var(--conf-high)",
  medium: "var(--warning)",
  low: "var(--conf-low)",
  unknown: "var(--conf-unknown)",
};

/** Mean of the finite confidences, or null when there are none (never 0 for "no data"). */
export function meanConfidence(values: Array<unknown>): number | null {
  const nums = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  return nums.length === 0 ? null : nums.reduce((a, b) => a + b, 0) / nums.length;
}

interface EntityLike {
  id: string;
  type?: string;
  confidence?: unknown;
  parent_id?: string;
  semantic_labels?: string[];
  custom_properties?: Record<string, unknown>;
  relationships?: Array<{ target_id?: string; kind?: string; type?: string }>;
}

export interface ConfidenceFacet {
  key: "geometry" | "dimensions" | "materials" | "objects" | "openings";
  label: string;
  reading: ConfidenceReading;
}

const isOpening = (e: EntityLike) => e.type === "door" || e.type === "window";
const isFurniture = (e: EntityLike) =>
  e.type === "furniture" || e.type === "object" || (e.semantic_labels ?? []).includes("furniture_candidate");

/**
 * Entities that belong to this room, by explicit link only (room_id, parent_id, or a relationship either way).
 * Doors and windows are linked to the wall that hosts them, so they join through a member wall.
 */
export function roomMembers(room: EntityLike, entities: Record<string, EntityLike>): EntityLike[] {
  const linked = new Set<string>((room.relationships ?? []).map((r) => String(r.target_id)));
  const direct = Object.values(entities).filter(
    (e) =>
      e.id !== room.id &&
      (e.parent_id === room.id ||
        e.custom_properties?.room_id === room.id ||
        linked.has(e.id) ||
        (e.relationships ?? []).some((r) => r.target_id === room.id)),
  );
  const wallIds = new Set(direct.filter((e) => e.type === "wall").map((e) => e.id));
  const have = new Set(direct.map((e) => e.id));
  const hosted = Object.values(entities).filter(
    (e) =>
      isOpening(e) &&
      !have.has(e.id) &&
      (wallIds.has(String(e.custom_properties?.host_wall_id)) ||
        (e.relationships ?? []).some((r) => r.target_id !== undefined && wallIds.has(r.target_id))),
  );
  return [...direct, ...hosted];
}

/**
 * Per-facet confidence. Only fields that exist are read:
 *  - geometry   = the entity's own `confidence`
 *  - dimensions / materials = `dimension_confidence` / `material_confidence` if the compiler ever records them
 *  - objects / openings (rooms only) = mean of the real confidences of the room's furniture / doors+windows
 * `uncertainty.confidence` is deliberately never read: the compiler writes 0.0 there as a placeholder.
 */
export function buildConfidenceFacets(
  entity: EntityLike,
  entities: Record<string, EntityLike> | null | undefined,
): ConfidenceFacet[] {
  const cp = entity.custom_properties ?? {};
  const facets: ConfidenceFacet[] = [
    { key: "geometry", label: "Geometry", reading: readConfidence(entity.confidence) },
    { key: "dimensions", label: "Dimensions", reading: readConfidence(cp.dimension_confidence) },
    { key: "materials", label: "Materials", reading: readConfidence(cp.material_confidence) },
  ];
  if (entity.type === "room") {
    const members = entities ? roomMembers(entity, entities) : [];
    facets.push(
      { key: "objects", label: "Objects", reading: readConfidence(meanConfidence(members.filter(isFurniture).map((e) => e.confidence))) },
      { key: "openings", label: "Openings", reading: readConfidence(meanConfidence(members.filter(isOpening).map((e) => e.confidence))) },
    );
  }
  return facets;
}
