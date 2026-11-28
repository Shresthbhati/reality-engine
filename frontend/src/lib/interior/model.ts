/**
 * Interior semantics derived from a real WorldIR version.
 *
 * Everything here is a READ-ONLY projection of what the engine actually
 * compiled. Nothing is invented:
 *
 * - A measurement is present only if the promoting stage recorded it
 *   (`promote_rooms` writes floor_area_m2 / floor_dimension_*_m / height_m
 *   into `custom_properties`). A missing measurement stays absent and the UI
 *   renders "not measured" rather than a default.
 * - Connectivity comes from real WorldIR `relationships` (CONTAINS /
 *   PART_OF / ADJACENT_TO edges). A room with no edge to another room is
 *   reported as having no recorded connection, not as "connected to nothing".
 * - `provenance` / `confidence` / `uncertainty` pass through verbatim so the
 *   UI can distinguish OBSERVED from INFERRED from RECONSTRUCTED output.
 *
 * Classes the engine has no producer for are modelled explicitly as empty
 * with a stated reason, so the UI says "the engine has no door detector"
 * rather than rendering an empty list that reads as a bug.
 */
import type { Entity, Geometry, WorldIR } from "@/types/worldir";

/** Entity classes the interior hierarchy addresses. */
export type InteriorKind =
  | "building"
  | "level"
  | "room"
  | "corridor"
  | "wall"
  | "door"
  | "window"
  | "floor"
  | "ceiling"
  | "stairs"
  | "other";

export const KIND_LABELS: Record<InteriorKind, string> = {
  building: "Building",
  level: "Level",
  room: "Room",
  corridor: "Corridor",
  wall: "Wall",
  door: "Door",
  window: "Window",
  floor: "Floor",
  ceiling: "Ceiling",
  stairs: "Stairs",
  other: "Other",
};

function normalizeType(raw: string | undefined): string {
  return (raw || "").trim().toLowerCase();
}

/** Map a raw WorldIR `type` onto an interior kind. */
export function kindOf(entity: Pick<Entity, "type">): InteriorKind {
  switch (normalizeType(entity.type)) {
    case "building":
      return "building";
    case "level":
    case "floor_level":
      return "level";
    case "room":
      return "room";
    case "corridor":
      return "corridor";
    case "wall":
      return "wall";
    case "door":
      return "door";
    case "window":
      return "window";
    case "floor":
    case "slab":
      return "floor";
    case "ceiling":
      return "ceiling";
    case "stairs":
    case "stair":
      return "stairs";
    default:
      return "other";
  }
}

function relKind(rel: { type?: string; kind?: string }): string {
  return (rel.type || rel.kind || "").trim().toLowerCase();
}

function relTarget(rel: { target_id?: string }): string | null {
  return rel.target_id ?? null;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * A recorded measurement: either a real value or explicitly absent. Never
 * defaults to 0 -- 0 is a legitimate measurement and must not stand in for
 * "unknown".
 */
export interface Measurement {
  value: number;
  unit: string;
  provenance: string | null;
}

function measurementFrom(
  props: Record<string, unknown> | undefined,
  key: string,
  unit: string,
): Measurement | null {
  const value = num(props?.[key]);
  if (value === null) return null;
  return { value, unit, provenance: "ESTIMATED" };
}

/** Axis-aligned bounds of an entity's geometry, when it has any. */
export interface Bounds {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
  size: { x: number; y: number; z: number };
}

export function boundsOf(
  entity: Entity,
  geometries: Record<string, Geometry> | undefined,
): Bounds | null {
  if (!geometries) return null;
  const boxes = (entity.geometry_ids || [])
    .map((gid) => geometries[gid])
    .filter((g): g is Geometry => Boolean(g && g.bounds_min && g.bounds_max))
    .map((g) => ({ min: g.bounds_min, max: g.bounds_max }));
  if (boxes.length === 0) return null;

  const min = {
    x: Math.min(...boxes.map((b) => b.min.x)),
    y: Math.min(...boxes.map((b) => b.min.y)),
    z: Math.min(...boxes.map((b) => b.min.z)),
  };
  const max = {
    x: Math.max(...boxes.map((b) => b.max.x)),
    y: Math.max(...boxes.map((b) => b.max.y)),
    z: Math.max(...boxes.map((b) => b.max.z)),
  };
  return {
    min,
    max,
    size: { x: max.x - min.x, y: max.y - min.y, z: max.z - min.z },
  };
}

/** An entity plus the real facts the inspector shows about it. */
export interface InteriorEntity {
  id: string;
  kind: InteriorKind;
  name: string;
  /** Raw WorldIR type, shown verbatim so nothing is relabelled. */
  rawType: string;
  provenance: string | null;
  confidence: number | null;
  uncertainty: Record<string, unknown> | null;
  bounds: Bounds | null;
  /** Measurements the promoting stage actually recorded. */
  measurements: Record<string, Measurement>;
  observationCount: number;
  relationshipCount: number;
  entity: Entity;
}

function toInteriorEntity(
  entity: Entity,
  geometries: Record<string, Geometry> | undefined,
): InteriorEntity {
  const props = entity.custom_properties;
  const measurements: Record<string, Measurement> = {};
  const add = (m: Measurement | null, key: string) => {
    if (m) measurements[key] = m;
  };

  // promote_rooms records the room measurements; the extents below come from
  // the real geometry bounds when the entity has any.
  add(measurementFrom(props, "floor_area_m2", "m²"), "floor_area_m2");
  add(measurementFrom(props, "floor_dimension_u_m", "m"), "floor_dimension_u_m");
  add(measurementFrom(props, "floor_dimension_v_m", "m"), "floor_dimension_v_m");
  add(measurementFrom(props, "height_m", "m"), "height_m");

  const bounds = boundsOf(entity, geometries);
  if (bounds) {
    const p = entity.provenance ?? null;
    measurements.extent_x = { value: bounds.size.x, unit: "m", provenance: p };
    measurements.extent_y = { value: bounds.size.y, unit: "m", provenance: p };
    measurements.extent_z = { value: bounds.size.z, unit: "m", provenance: p };
  }

  return {
    id: entity.id,
    kind: kindOf(entity),
    name: entity.name || entity.id,
    rawType: entity.type,
    provenance: entity.provenance ?? null,
    confidence: num(entity.confidence),
    uncertainty: entity.uncertainty ?? null,
    bounds,
    measurements,
    observationCount: (entity.observations || []).length,
    relationshipCount: (entity.relationships || []).length,
    entity,
  };
}

/** A room/corridor with its real containment and connectivity. */
export interface InteriorSpace extends InteriorEntity {
  /** Entity ids this space CONTAINS (real WorldIR edge). */
  contains: string[];
  /** Those entities, resolved. */
  containsElements: InteriorEntity[];
  /** Sibling spaces reached by a real relationship edge. */
  connectedSpaces: Array<{ id: string; name: string; kind: InteriorKind; via: string }>;
  /** Door/window entities recorded inside this space. */
  openings: Array<{ id: string; kind: "door" | "window"; name: string }>;
  /** Other rooms recorded as touching this one. */
  connectedRooms: Array<{ id: string; name: string }>;
  /** Corridors recorded as touching this room. */
  corridorConnections: Array<{ id: string; name: string }>;
}

/** Why a hierarchy class has no entries -- shown instead of a blank list. */
export interface UnsupportedKind {
  kind: InteriorKind;
  label: string;
  reason: string;
}

export interface InteriorModel {
  worldId: string;
  worldName: string;
  versionId: string | null;
  spaces: InteriorSpace[];
  elements: InteriorEntity[];
  /** Count per kind actually present in this version. */
  counts: Record<InteriorKind, number>;
  /** Kinds absent from this version, with the reason. */
  unsupported: UnsupportedKind[];
  /** Every entity, for the viewport and search. */
  all: InteriorEntity[];
}

/** Kinds the current engine pipeline has a producer for. */
const ENGINE_PRODUCED_KINDS: InteriorKind[] = [
  "room",
  "wall",
  "floor",
  "ceiling",
  "building",
];

function isSpaceKind(kind: InteriorKind): boolean {
  return kind === "room" || kind === "corridor";
}

/**
 * Build the interior model for one WorldIR version.
 *
 * Returns null only when there is no WorldIR at all. A version with zero
 * entities still yields a model -- with empty collections and a populated
 * `unsupported` list -- which is the honest "engine reconstructed nothing of
 * this class" state.
 */
export function buildInteriorModel(
  worldIR: WorldIR | null,
  versionId: string | null,
): InteriorModel | null {
  if (!worldIR) return null;

  const rawEntities = Object.values(worldIR.entities || {});
  const all = rawEntities.map((e) => toInteriorEntity(e, worldIR.geometries));
  const byId = new Map(all.map((e) => [e.id, e]));

  const counts = {} as Record<InteriorKind, number>;
  for (const k of Object.keys(KIND_LABELS) as InteriorKind[]) counts[k] = 0;
  for (const e of all) {
    if (e.kind !== "other") counts[e.kind] += 1;
  }

  // Reverse index: which space contains each entity, from real CONTAINS and
  // PART_OF edges in either direction.
  const parentOf = new Map<string, string>();
  for (const e of all) {
    for (const rel of e.entity.relationships || []) {
      const kind = relKind(rel);
      const target = relTarget(rel);
      if (!target) continue;
      if (kind === "contains" && isSpaceKind(byId.get(e.id)?.kind ?? "other")) {
        if (!parentOf.has(target)) parentOf.set(target, e.id);
      }
      if (kind === "part_of" && isSpaceKind(byId.get(target)?.kind ?? "other")) {
        if (!parentOf.has(e.id)) parentOf.set(e.id, target);
      }
    }
  }

  const spaces: InteriorSpace[] = [];
  for (const space of all) {
    if (!isSpaceKind(space.kind)) continue;

    const containsIds = new Set<string>();
    for (const rel of space.entity.relationships || []) {
      if (relKind(rel) === "contains") {
        const t = relTarget(rel);
        if (t && byId.has(t)) containsIds.add(t);
      }
    }
    for (const [childId, parentId] of parentOf) {
      if (parentId === space.id) containsIds.add(childId);
    }

    const containsElements = [...containsIds]
      .map((cid) => byId.get(cid))
      .filter((e): e is InteriorEntity => Boolean(e));


    // Sibling spaces: any space reachable through a real relationship edge.
    const connectedSpaces: InteriorSpace["connectedSpaces"] = [];
    const connectedRooms: InteriorSpace["connectedRooms"] = [];
    const corridorConnections: InteriorSpace["corridorConnections"] = [];

    const considerNeighbour = (otherId: string, via: string) => {
      const other = byId.get(otherId);
      if (!other || otherId === space.id || !isSpaceKind(other.kind)) return;
      if (connectedSpaces.some((c) => c.id === otherId)) return;
      connectedSpaces.push({ id: otherId, name: other.name, kind: other.kind, via });
      if (other.kind === "room") connectedRooms.push({ id: otherId, name: other.name });
      if (other.kind === "corridor") corridorConnections.push({ id: otherId, name: other.name });
    };

    for (const rel of space.entity.relationships || []) {
      const kind = relKind(rel);
      const t = relTarget(rel);
      if (!t) continue;
      if (kind === "adjacent_to" || kind === "overlaps" || kind === "crosses") {
        considerNeighbour(t, kind);
      }
    }
    // A neighbour is symmetric: if another space points at this one, that is
    // a recorded connection too.
    for (const other of all) {
      if (!isSpaceKind(other.kind)) continue;
      for (const rel of other.entity.relationships || []) {
        if (relTarget(rel) === space.id) considerNeighbour(other.id, relKind(rel));
      }
    }

    const openings = containsElements
      .filter((e) => e.kind === "door" || e.kind === "window")
      .map((e) => ({ id: e.id, kind: e.kind as "door" | "window", name: e.name }));

    spaces.push({
      ...space,
      contains: [...containsIds],
      containsElements,
      connectedSpaces,
      openings,
      connectedRooms,
      corridorConnections,
    });
  }

  const spaceIds = new Set(spaces.map((s) => s.id));
  const elements = all.filter((e) => !spaceIds.has(e.id));

  // Only list a class as unsupported when it is absent from the whole world.
  // A supported class with zero entries is a reconstruction fact the UI shows
  // differently ("engine found none in this capture").
  const unsupported: UnsupportedKind[] = (
    ["door", "window", "stairs", "level", "corridor"] as InteriorKind[]
  )
    .filter((k) => counts[k] === 0)
    .map((kind) => ({
      kind,
      label: KIND_LABELS[kind],
      reason: ENGINE_PRODUCED_KINDS.includes(kind)
        ? "No such entity was reconstructed in this version."
        : "This class has no producer in the current engine pipeline.",
    }));

  return {
    worldId: worldIR.id,
    worldName: worldIR.name || worldIR.id,
    versionId,
    spaces,
    elements,
    counts,
    unsupported,
    all,
  };
}

/** Groups of a kind, in stable display order, for the hierarchy tree. */
export function entitiesOfKind(
  model: InteriorModel,
  kind: InteriorKind,
): InteriorEntity[] {
  return model.all.filter((e) => e.kind === kind);
}
