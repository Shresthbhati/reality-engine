/**
 * Scene tree derived ONLY from entities that exist in the WorldIR.
 *
 * Nothing is invented: a Building / Floor / Room row exists only when the WorldIR has a matching
 * entity, a group (Walls, Windows, ...) exists only when it has members, and an entity that cannot be
 * tied to a room lands in "Unassigned" rather than being guessed into one.
 */
import type { Entity, WorldIR } from "@/types/worldir";
import { OUTDOOR_GROUP_LABEL, OUTDOOR_GROUP_ORDER, outdoorDisplayName, outdoorGroupOf, type OutdoorGroup } from "../viewport/outdoor-model.ts";

export type SceneKind =
  | "building"
  | "level"
  | "room"
  | "floor"
  | "ceiling"
  | "wall"
  | "window"
  | "door"
  | "furniture"
  | "lighting"
  | "other";

export type SceneNodeKind = SceneKind | "group" | "unassigned";

export interface SceneNode {
  /** Unique within the tree. */
  key: string;
  kind: SceneNodeKind;
  label: string;
  /** Muted secondary text (the raw id when the label had to be synthesised). */
  hint?: string;
  /** Set when the row stands for a real entity and can be selected. */
  entityId: string | null;
  /** Group rows: member count shown in the label. */
  count?: number;
  /** Every entity id at or below this node (drives the eye toggle). */
  leafIds: string[];
  children: SceneNode[];
  /** Set only when the entity is map-imported or generated (not captured); absent otherwise. */
  badge?: OriginBadge;
}

export type EntityOrigin = "captured" | "imported" | "generated";
export type OriginBadge = "Imported from map" | "Generated (not captured)";
export const ORIGIN_BADGE: Record<Exclude<EntityOrigin, "captured">, OriginBadge> = {
  imported: "Imported from map",
  generated: "Generated (not captured)",
};

/** Where an entity came from. Generated wins over imported; everything else counts as captured. */
export function entityOrigin(e: Entity): EntityOrigin {
  const p = (e.custom_properties ?? {}) as Record<string, unknown>;
  if (p.generated_by) return "generated";
  if (p.imported === true || p.source === "OpenStreetMap") return "imported";
  return "captured";
}

type Rec = Record<string, unknown>;

const lower = (v: unknown): string => (typeof v === "string" ? v.toLowerCase() : "");
const cp = (e: Entity): Rec => (e.custom_properties ?? {}) as Rec;

/** Which scene kind an entity is, from its type, labels and opening kind. */
export function classifyEntity(e: Entity): SceneKind {
  const t = lower(e.type);
  const labels = (e.semantic_labels ?? []).map(lower);
  const kind = lower(cp(e).kind) || lower(cp(e).subtype);
  if (t === "building") return "building";
  if (t === "storey" || t === "level" || t === "floor_level") return "level";
  if (t === "room" || t === "space" || t === "corridor" || kind === "corridor" || labels.includes("corridor")) return "room";
  if (t === "floor") return "floor";
  if (t === "ceiling") return "ceiling";
  if (t === "wall") return "wall";
  if (t === "window" || kind === "window") return "window";
  if (t === "door" || t === "doorway" || kind === "door" || kind === "doorway") return "door";
  if (t === "opening") return "other";
  if (t === "furniture" || labels.includes("furniture") || labels.includes("furniture_candidate")) return "furniture";
  if (t === "light" || t === "lighting" || t === "lamp" || t === "luminaire" || labels.includes("lighting") || labels.includes("light")) return "lighting";
  return "other";
}

const BASE_NAME: Record<SceneKind, string> = {
  building: "Building",
  level: "Floor",
  room: "Room",
  floor: "Floor",
  ceiling: "Ceiling",
  wall: "Wall",
  window: "Window",
  door: "Door",
  furniture: "Furniture",
  lighting: "Light",
  other: "Object",
};

const GROUP_LABEL: Record<string, string> = {
  wall: "Walls",
  window: "Windows",
  door: "Doors",
  furniture: "Furniture",
  lighting: "Lighting",
  other: "Other",
};

/** "room-plane-000" / "struct-plane-001" are ids, not names. */
function looksLikeId(name: string, id: string): boolean {
  return name === id || /^[a-z0-9]+([-_][a-z0-9]+)+$/.test(name);
}

const pad3 = (n: number) => String(n).padStart(3, "0");

function entityLabel(e: Entity, kind: SceneKind, index: number, total: number): { label: string; hint?: string } {
  const name = (e.name ?? "").trim();
  if (name && !looksLikeId(name, e.id)) return { label: name };
  const base = kind === "other" && e.type && e.type !== "unknown" ? e.type.charAt(0).toUpperCase() + e.type.slice(1) : BASE_NAME[kind];
  const single = total === 1 && (kind === "floor" || kind === "ceiling");
  return { label: single ? base : `${base} ${pad3(index + 1)}`, hint: e.id };
}

/** Relationship kinds that mean "this belongs to / sits in that". */
const MEMBER_KINDS = new Set(["part_of", "contained_in", "belongs_to", "in", "child_of"]);

function relTargets(e: Entity, kinds?: Set<string>): string[] {
  const out: string[] = [];
  for (const r of e.relationships ?? []) {
    const k = lower((r as Rec).kind ?? (r as Rec).type);
    if (!kinds || kinds.has(k)) out.push(r.target_id);
  }
  return out;
}

const idList = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);

export interface BuildSceneTreeResult {
  tree: SceneNode[];
  /** entity id -> row, for selection / ancestor lookup. */
  byEntity: Map<string, SceneNode>;
}

export function buildSceneTree(world: WorldIR | null | undefined): SceneNode[] {
  return buildSceneTreeDetailed(world).tree;
}

export function buildSceneTreeDetailed(world: WorldIR | null | undefined): BuildSceneTreeResult {
  const all = Object.values(world?.entities ?? {}).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  // map-imported / generated outdoor entities get their own typed groups and never compete for the building/room roles
  const outdoor = all.filter((e) => outdoorGroupOf(e) !== null);
  const outdoorIds = new Set(outdoor.map((e) => e.id));
  const entities = all.filter((e) => !outdoorIds.has(e.id));
  const byId = new Map(entities.map((e) => [e.id, e]));
  const kindOf = new Map(entities.map((e) => [e.id, classifyEntity(e)]));
  const ofKind = (k: SceneKind) => entities.filter((e) => kindOf.get(e.id) === k);

  const rooms = ofKind("room");
  const roomIds = new Set(rooms.map((r) => r.id));
  const levels = ofKind("level");
  const levelIds = new Set(levels.map((l) => l.id));
  const building = ofKind("building")[0] ?? null;

  // ids a room lists as its own (walls, doors, ...), and reverse "attached_to" evidence links
  const listedIn = new Map<string, string>();
  for (const r of rooms) {
    const p = cp(r);
    for (const key of ["wall_slab_ids", "wall_ids", "door_ids", "window_ids", "boundary_element_ids", "furniture_ids", "member_ids"]) {
      for (const id of idList(p[key])) if (byId.has(id) && !listedIn.has(id)) listedIn.set(id, r.id);
    }
  }
  const attachedFrom = new Map<string, string[]>();
  for (const e of entities) {
    for (const t of relTargets(e, new Set(["attached_to"]))) {
      const arr = attachedFrom.get(t) ?? [];
      arr.push(e.id);
      attachedFrom.set(t, arr);
    }
  }

  const roomCache = new Map<string, string | null>();
  const resolveRoom = (e: Entity, seen: Set<string> = new Set()): string | null => {
    if (roomCache.has(e.id)) return roomCache.get(e.id)!;
    if (seen.has(e.id) || seen.size > 4) return null;
    seen.add(e.id);
    let found: string | null = null;
    const p = cp(e);
    if (e.parent_id && roomIds.has(e.parent_id)) found = e.parent_id;
    else if (typeof p.room_id === "string" && roomIds.has(p.room_id)) found = p.room_id;
    else if (listedIn.has(e.id)) found = listedIn.get(e.id)!;
    if (!found) {
      for (const t of relTargets(e, MEMBER_KINDS)) {
        if (roomIds.has(t)) {
          found = t;
          break;
        }
      }
    }
    if (!found) {
      const hosts = [typeof p.host_wall_id === "string" ? p.host_wall_id : null, ...relTargets(e, new Set([...MEMBER_KINDS, "attached_to"]))];
      for (const h of hosts) {
        const he = h ? byId.get(h) : null;
        if (he && kindOf.get(he.id) === "wall") {
          found = resolveRoom(he, seen);
          if (found) break;
        }
      }
    }
    if (!found) {
      for (const a of attachedFrom.get(e.id) ?? []) {
        const ae = byId.get(a);
        if (ae) {
          found = resolveRoom(ae, seen);
          if (found) break;
        }
      }
    }
    roomCache.set(e.id, found);
    return found;
  };

  // room -> level (only when both exist as entities)
  const levelOfRoom = new Map<string, string>();
  const metaLevels = ((world?.metadata?.interior_space_graph as Rec | undefined)?.levels as Rec[] | undefined) ?? [];
  for (const lvl of metaLevels) {
    if (typeof lvl.level_id === "string" && levelIds.has(lvl.level_id)) {
      for (const rid of [...idList(lvl.room_ids), ...idList(lvl.corridor_ids)]) if (roomIds.has(rid)) levelOfRoom.set(rid, lvl.level_id);
    }
  }
  for (const r of rooms) {
    if (levelOfRoom.has(r.id)) continue;
    if (r.parent_id && levelIds.has(r.parent_id)) levelOfRoom.set(r.id, r.parent_id);
    else for (const t of relTargets(r, MEMBER_KINDS)) if (levelIds.has(t)) levelOfRoom.set(r.id, t);
  }

  const byEntity = new Map<string, SceneNode>();
  const leaf = (e: Entity, kind: SceneKind, index: number, total: number): SceneNode => {
    const { label, hint } = entityLabel(e, kind, index, total);
    const n: SceneNode = { key: `e:${e.id}`, kind, label, hint, entityId: e.id, leafIds: [e.id], children: [] };
    byEntity.set(e.id, n);
    return n;
  };

  /** Floors, ceilings and the typed groups for a set of member entities (no rooms/levels/buildings). */
  const contentNodes = (members: Entity[], parentKey: string): SceneNode[] => {
    const out: SceneNode[] = [];
    for (const k of ["floor", "ceiling"] as const) {
      const list = members.filter((m) => kindOf.get(m.id) === k);
      list.forEach((m, i) => out.push(leaf(m, k, i, list.length)));
    }
    for (const k of ["wall", "window", "door", "furniture", "lighting", "other"] as const) {
      const list = members.filter((m) => kindOf.get(m.id) === k);
      if (list.length === 0) continue;
      const children = list.map((m, i) => leaf(m, k, i, list.length));
      out.push({
        key: `g:${parentKey}:${k}`,
        kind: "group",
        label: `${GROUP_LABEL[k]} (${list.length})`,
        entityId: null,
        count: list.length,
        leafIds: children.flatMap((c) => c.leafIds),
        children,
      });
    }
    return out;
  };

  const withLeafIds = (n: SceneNode): SceneNode => {
    n.leafIds = [...(n.entityId ? [n.entityId] : []), ...n.children.flatMap((c) => c.leafIds)];
    return n;
  };

  const structural = entities.filter((e) => {
    const k = kindOf.get(e.id)!;
    return k !== "building" && k !== "level" && k !== "room";
  });
  const membersOfRoom = new Map<string, Entity[]>();
  const unassigned: Entity[] = [];
  for (const e of structural) {
    const rid = resolveRoom(e);
    if (rid) (membersOfRoom.get(rid) ?? membersOfRoom.set(rid, []).get(rid)!).push(e);
    else unassigned.push(e);
  }

  const roomNode = (r: Entity, i: number): SceneNode => {
    const { label, hint } = entityLabel(r, "room", i, rooms.length);
    const n: SceneNode = { key: `e:${r.id}`, kind: "room", label, hint, entityId: r.id, leafIds: [], children: contentNodes(membersOfRoom.get(r.id) ?? [], r.id) };
    byEntity.set(r.id, n);
    return withLeafIds(n);
  };
  const roomNodes = new Map(rooms.map((r, i) => [r.id, roomNode(r, i)]));

  const levelNode = (l: Entity, i: number): SceneNode => {
    const { label, hint } = entityLabel(l, "level", i, levels.length);
    const children = rooms.filter((r) => levelOfRoom.get(r.id) === l.id).map((r) => roomNodes.get(r.id)!);
    const n: SceneNode = { key: `e:${l.id}`, kind: "level", label, hint, entityId: l.id, leafIds: [], children };
    byEntity.set(l.id, n);
    return withLeafIds(n);
  };
  const levelNodes = levels.map(levelNode);

  // Top level: [building > (levels, loose rooms)] or just (levels, loose rooms)
  const looseRooms = rooms.filter((r) => !levelOfRoom.has(r.id)).map((r) => roomNodes.get(r.id)!);
  const topChildren = [...levelNodes, ...looseRooms];

  let tree: SceneNode[];
  if (building) {
    const { label, hint } = entityLabel(building, "building", 0, 1);
    const bn: SceneNode = { key: `e:${building.id}`, kind: "building", label, hint, entityId: building.id, leafIds: [], children: topChildren };
    byEntity.set(building.id, bn);
    tree = [withLeafIds(bn)];
  } else {
    tree = topChildren;
  }

  if (unassigned.length > 0) {
    const nodes = contentNodes(unassigned, "unassigned");
    if (rooms.length === 0 && levels.length === 0 && !building) {
      tree = [...tree, ...nodes];
    } else {
      tree = [...tree, withLeafIds({ key: "unassigned", kind: "unassigned", label: "Unassigned", entityId: null, leafIds: [], children: nodes })];
    }
  }
  // Outdoor groups: Buildings, Roads & paths, Rail & metro, Bridges, Street furniture, Trees, Other map features
  const perKind = new Map<string, number>();
  for (const g of OUTDOOR_GROUP_ORDER) {
    const members = outdoor.filter((e) => outdoorGroupOf(e) === g);
    if (members.length === 0) continue;
    const children = members.map((e): SceneNode => {
      const kind = (cp(e).kind ?? cp(e).road_kind ?? cp(e).connector_kind ?? e.type ?? "") as string;
      const n = (perKind.get(kind) ?? 0) + 1;
      perKind.set(kind, n);
      const leafNode: SceneNode = { key: `e:${e.id}`, kind: g === "buildings" ? "building" : "other", label: outdoorDisplayName(e, n), hint: e.id, entityId: e.id, leafIds: [e.id], children: [] };
      byEntity.set(e.id, leafNode);
      return leafNode;
    });
    tree.push({ key: `g:outdoor:${g satisfies OutdoorGroup}`, kind: "group", label: `${OUTDOOR_GROUP_LABEL[g]} (${members.length})`, entityId: null, count: members.length, leafIds: children.flatMap((c) => c.leafIds), children });
  }
  const originOf = new Map(all.map((e) => [e.id, e]));
  const stamp = (n: SceneNode) => {
    const e = n.entityId ? originOf.get(n.entityId) : undefined;
    const o = e ? entityOrigin(e) : "captured";
    if (o !== "captured") n.badge = ORIGIN_BADGE[o];
    n.children.forEach(stamp);
  };
  tree.forEach(stamp);
  return { tree, byEntity };
}

/** Keys that should be open on first render: every structural row, but not the member groups. */
export function defaultExpandedKeys(tree: SceneNode[]): Set<string> {
  const out = new Set<string>();
  const walk = (n: SceneNode) => {
    if (n.kind !== "group" && n.children.length > 0) out.add(n.key);
    n.children.forEach(walk);
  };
  tree.forEach(walk);
  return out;
}

/** Keys of every ancestor of an entity's row (so selecting from the viewport reveals it). */
export function ancestorKeys(tree: SceneNode[], entityId: string | null): string[] {
  if (!entityId) return [];
  const path: string[] = [];
  const walk = (n: SceneNode, trail: string[]): boolean => {
    if (n.entityId === entityId) {
      path.push(...trail);
      return true;
    }
    return n.children.some((c) => walk(c, [...trail, n.key]));
  };
  tree.some((n) => walk(n, []));
  return path;
}

/** Keep rows whose label/hint/id match, or that have a matching descendant. A matching group keeps all its members. */
export function filterTree(tree: SceneNode[], query: string): SceneNode[] {
  const q = query.trim().toLowerCase();
  if (!q) return tree;
  const hit = (n: SceneNode) => n.label.toLowerCase().includes(q) || (n.hint ?? "").toLowerCase().includes(q) || (n.entityId ?? "").toLowerCase().includes(q);
  const walk = (n: SceneNode): SceneNode | null => {
    if (hit(n)) return n;
    const kids = n.children.map(walk).filter((c): c is SceneNode => c !== null);
    return kids.length > 0 ? { ...n, children: kids } : null;
  };
  return tree.map(walk).filter((n): n is SceneNode => n !== null);
}

export type Visibility = "all" | "none" | "some";

export function visibilityOf(leafIds: readonly string[], hidden: ReadonlySet<string>): Visibility {
  if (leafIds.length === 0) return "all";
  let hiddenCount = 0;
  for (const id of leafIds) if (hidden.has(id)) hiddenCount++;
  return hiddenCount === 0 ? "all" : hiddenCount === leafIds.length ? "none" : "some";
}

/** Eye toggle: if everything is hidden show it all, otherwise hide it all. Returns a new set. */
export function toggleHidden(hidden: ReadonlySet<string>, leafIds: readonly string[]): Set<string> {
  const next = new Set(hidden);
  if (visibilityOf(leafIds, hidden) === "none") for (const id of leafIds) next.delete(id);
  else for (const id of leafIds) next.add(id);
  return next;
}

/** Every entity id in the tree (for pruning stale hidden ids when the WorldIR changes). */
export function allEntityIds(tree: SceneNode[]): Set<string> {
  const out = new Set<string>();
  for (const n of tree) for (const id of n.leafIds) out.add(id);
  return out;
}
