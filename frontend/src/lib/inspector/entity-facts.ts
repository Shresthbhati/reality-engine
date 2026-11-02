/**
 * Plain-language facts for the inspector, derived only from what the entity records. Pure (no React / three.js).
 * A fact that has no data is omitted; a few key facts say "Not identified" so the UI stays honest.
 */
import type { DimensionEvidence, DimState } from "../viewport/inspector-evidence-model.ts";

export type Rgb = [number, number, number];
type Props = Record<string, unknown>;

export interface EntityLike {
  id: string;
  type?: string;
  name?: string;
  parent_id?: string;
  semantic_labels?: string[];
  relationships?: Array<{ target_id?: string; kind?: string; type?: string }>;
  custom_properties?: Props;
}

const isRec = (v: unknown): v is Props => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export type EntityKind = "room" | "wall" | "window" | "door" | "floor" | "ceiling" | "furniture" | "other";

export function entityKind(e: EntityLike): EntityKind {
  const t = String(e.type ?? "").toLowerCase();
  if (t === "room" || t === "wall" || t === "window" || t === "door" || t === "floor" || t === "ceiling") return t;
  if (t === "furniture" || t === "object" || (e.semantic_labels ?? []).includes("furniture_candidate")) return "furniture";
  return "other";
}

const KIND_LABEL: Record<EntityKind, string> = {
  room: "Room", wall: "Wall", window: "Window", door: "Door", floor: "Floor", ceiling: "Ceiling", furniture: "Furniture", other: "Item",
};
export const kindLabel = (e: EntityLike): string => {
  const k = entityKind(e);
  return k === "other" && e.type ? e.type.charAt(0).toUpperCase() + e.type.slice(1) : KIND_LABEL[k];
};

const ID_LIKE = /^[a-z0-9]+([-_][a-z0-9]+)+$/;
/** A name that is just a machine id ("room-plane-000") becomes the plain kind ("Room"); the id stays in Details. */
export function displayName(e: EntityLike): string {
  const name = (e.name ?? "").trim();
  if (!name || name === e.id || ID_LIKE.test(name)) return kindLabel(e);
  return name.charAt(0).toUpperCase() + name.slice(1);
}

/* ---------------- dimensions ---------------- */

export const STATE_WORD: Record<DimState, string> = {
  MEASURED: "Measured", RECONSTRUCTED: "Reconstructed", ESTIMATED: "Estimated", VERIFIED: "Verified",
};
export const stateWord = (s: DimState | null): string => (s ? STATE_WORD[s] : "State not recorded");

/** Units as a person reads them. "model_units" means the world has no metric scale yet. */
export function unitText(unit: string | null): string {
  if (!unit) return "";
  if (unit === "model_units") return "units";
  if (unit === "model_units2") return "units²";
  if (unit === "model_units3") return "units³";
  if (unit === "m2") return "m²";
  return unit;
}
export const isUncalibratedUnit = (unit: string | null): boolean => !!unit && unit.startsWith("model_units");

const fmtNum = (v: number): string => (Math.abs(v) < 0.0005 ? "0" : Number.isInteger(v) ? String(v) : v.toFixed(2));
export const fmtValue = (v: number, unit: string | null): string => {
  const u = unitText(unit);
  return u ? `${fmtNum(v)} ${u}` : fmtNum(v);
};

export interface DimensionRow {
  key: string;
  label: string;
  value: number;
  unit: string | null;
  state: DimState | null;
  basis: string | null;
  uncertaintyM: number | null;
  /** the 0.10 m thickness default is not a measurement */
  assumedDefault: boolean;
  /** true when a user-entered real value can replace it (it is a recorded dimension) */
  correctable: boolean;
}

const humanize = (key: string): string => {
  const base = key.replace(/_m2$|_m$/, "").replace(/_/g, " ").trim();
  return base.charAt(0).toUpperCase() + base.slice(1);
};

type Spec = Array<[string, string]>; // [dimension key, label]
const SPECS: Record<EntityKind, Spec> = {
  room: [["layout_dimension_u_m", "Width"], ["layout_dimension_v_m", "Length"], ["ceiling_height_max_m", "Ceiling height"], ["height_m", "Ceiling height"], ["floor_area_m2", "Floor area"], ["net_wall_area_m2", "Wall area (net)"]],
  wall: [["length_m", "Length"], ["height_m", "Height"], ["thickness_m", "Thickness"]],
  window: [["width_m", "Width"], ["height_m", "Height"], ["sill_height_m", "Sill height"]],
  door: [["width_m", "Width"], ["height_m", "Height"], ["sill_height_m", "Sill height"]],
  floor: [["thickness_m", "Thickness"]],
  ceiling: [["height_above_floor_m", "Height above floor"], ["thickness_m", "Thickness"]],
  furniture: [["height_m", "Height"], ["base_height_above_floor_m", "Height above floor"]],
  other: [],
};

const isAssumedDefault = (key: string, d: DimensionEvidence, cp: Props): boolean =>
  key === "thickness_m" && (/default/i.test(d.basis ?? "") || cp.thickness_basis === "default_unmeasured");

/**
 * The few dimensions a person asks about, in reading order, with plain labels.
 * Everything else the entity records is available through `allDimensionRows` (Details).
 */
export function primaryDimensionRows(e: EntityLike, dims: DimensionEvidence[]): DimensionRow[] {
  const cp = isRec(e.custom_properties) ? e.custom_properties : {};
  const kind = entityKind(e);
  const byKey = new Map(dims.map((d) => [d.key, d]));
  const rows: DimensionRow[] = [];
  const seenLabels = new Set<string>();
  for (const [key, label] of SPECS[kind]) {
    const d = byKey.get(key);
    if (!d || seenLabels.has(label)) continue;
    seenLabels.add(label);
    rows.push({ key, label, value: d.value, unit: d.unit, state: d.state, basis: d.basis, uncertaintyM: d.uncertaintyM, assumedDefault: isAssumedDefault(key, d, cp), correctable: true });
  }
  if (kind === "furniture") {
    // The footprint is recorded as a plan-view box, with no per-value state.
    const obb = cp.plan_obb_size_m;
    if (Array.isArray(obb) && obb.length === 2 && obb.every((n) => num(n) !== null)) {
      const unit = dims.find((d) => d.key === "height_m")?.unit ?? null;
      const mk = (key: string, label: string, value: number): DimensionRow => ({ key, label, value, unit, state: null, basis: "plan-view bounding box of the object's points", uncertaintyM: null, assumedDefault: false, correctable: false });
      rows.unshift(mk("plan_obb_size_m.0", "Width", obb[0] as number), mk("plan_obb_size_m.1", "Depth", obb[1] as number));
    }
  }
  return rows;
}

export function allDimensionRows(e: EntityLike, dims: DimensionEvidence[]): DimensionRow[] {
  const cp = isRec(e.custom_properties) ? e.custom_properties : {};
  return dims.map((d) => ({ key: d.key, label: humanize(d.key), value: d.value, unit: d.unit, state: d.state, basis: d.basis, uncertaintyM: d.uncertaintyM, assumedDefault: isAssumedDefault(d.key, d, cp), correctable: true }));
}

/** Text for the (i) tooltip: state, basis, uncertainty. */
export function dimensionTooltip(r: DimensionRow): string {
  const parts = [stateWord(r.state)];
  parts.push(`Basis: ${r.assumedDefault ? "assumed default - no stage measures this" : r.basis ?? "not recorded"}`);
  parts.push(`Uncertainty: ${r.uncertaintyM === null ? "not recorded" : `±${r.uncertaintyM.toFixed(3)} m`}`);
  if (isUncalibratedUnit(r.unit)) parts.push("Scale is not calibrated, so this is in model units, not metres");
  return parts.join(". ");
}

/* ---------------- overview facts ---------------- */

export interface Fact {
  label: string;
  value: string;
  /** muted: the honest "Not identified" style */
  muted?: boolean;
}

function dimOf(dims: DimensionEvidence[], ...keys: string[]): DimensionEvidence | null {
  for (const k of keys) {
    const d = dims.find((x) => x.key === k);
    if (d) return d;
  }
  return null;
}
const stateSuffix = (s: DimState | null): string => (s === "MEASURED" ? "" : s ? ` (${STATE_WORD[s].toLowerCase()})` : "");

export function roomFacts(room: EntityLike, dims: DimensionEvidence[], levelName: string | null): Fact[] {
  const cp = isRec(room.custom_properties) ? room.custom_properties : {};
  const out: Fact[] = [];
  if (levelName) out.push({ label: "Floor", value: levelName });
  const type = [cp.room_type, cp.subtype, (room.semantic_labels ?? [])[0]].find((v) => typeof v === "string" && v.trim() !== "") as string | undefined;
  out.push(type ? { label: "Room type", value: type.charAt(0).toUpperCase() + type.slice(1) } : { label: "Room type", value: "Not identified", muted: true });
  const area = dimOf(dims, "floor_area_m2");
  const height = dimOf(dims, "ceiling_height_max_m", "height_m");
  if (area) out.push({ label: `Area${stateSuffix(area.state)}`, value: fmtValue(area.value, area.unit) });
  if (area && height) {
    // Volume is a product of two recorded values, so it is always an estimate.
    const unit = area.unit?.startsWith("model_units") ? "units³" : "m³";
    out.push({ label: "Volume (estimated)", value: `${fmtNum(area.value * height.value)} ${unit}` });
  }
  return out;
}

export function hostedOpeningIds(wall: EntityLike, entities: Record<string, EntityLike>): string[] {
  const cp = isRec(wall.custom_properties) ? wall.custom_properties : {};
  const ids = new Set<string>();
  const slab = isRec(cp.wall_slab) ? cp.wall_slab : null;
  if (slab && Array.isArray(slab.openings)) for (const o of slab.openings) if (isRec(o) && typeof o.entity_id === "string") ids.add(o.entity_id);
  for (const r of wall.relationships ?? []) {
    const t = r.target_id ? entities[r.target_id] : undefined;
    if (t && (t.type === "door" || t.type === "window") && (r.kind ?? r.type) === "contains") ids.add(t.id);
  }
  for (const e of Object.values(entities)) {
    if (e.type !== "door" && e.type !== "window") continue;
    const host = isRec(e.custom_properties) ? e.custom_properties.host_wall_id : undefined;
    if (host === wall.id || (e.relationships ?? []).some((r) => r.target_id === wall.id && (r.kind ?? r.type) === "part_of")) ids.add(e.id);
  }
  return [...ids].filter((id) => entities[id]);
}

/** Wall that hosts an opening, when recorded. */
export function openingHostId(op: EntityLike, entities: Record<string, EntityLike>): string | null {
  const cp = isRec(op.custom_properties) ? op.custom_properties : {};
  const direct = typeof cp.host_wall_id === "string" ? cp.host_wall_id : null;
  if (direct && entities[direct]) return direct;
  for (const r of op.relationships ?? []) {
    const t = r.target_id ? entities[r.target_id] : undefined;
    if (t?.type === "wall" && !isEvidencePlane(t) && (r.kind ?? r.type) === "part_of") return t.id;
  }
  return null;
}

/* ---------------- colour / materials ---------------- */

export interface Swatch {
  title: string;
  rgb: Rgb;
  name: string;
}

const asRgb = (v: unknown): Rgb | null =>
  Array.isArray(v) && v.length === 3 && v.every((n) => num(n) !== null) ? (v as Rgb) : null;

/** Rough colour name from rgb: lightness + saturation + hue family. An estimate, labelled as one in the UI. */
export function colourName([r, g, b]: Rgb): string {
  const R = r / 255, G = g / 255, B = b / 255;
  const max = Math.max(R, G, B), min = Math.min(R, G, B), l = (max + min) / 2, d = max - min;
  const s = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
  if (l > 0.88) return "Off-white";
  if (l < 0.12) return "Near black";
  if (s < 0.12) return l > 0.65 ? "Light grey" : l > 0.35 ? "Grey" : "Dark grey";
  let h = 0;
  if (d !== 0) h = max === R ? ((G - B) / d) % 6 : max === G ? (B - R) / d + 2 : (R - G) / d + 4;
  h = (h * 60 + 360) % 360;
  const fam = h < 15 || h >= 345 ? "red" : h < 40 ? "orange" : h < 65 ? "yellow" : h < 160 ? "green" : h < 200 ? "teal" : h < 255 ? "blue" : h < 300 ? "purple" : "pink";
  const warmBrown = (fam === "orange" || fam === "red" || fam === "yellow") && l < 0.5 && s < 0.6;
  const tone = l > 0.7 ? "Light " : l < 0.3 ? "Dark " : s < 0.35 ? "Muted " : "";
  return `${tone}${warmBrown ? "brown" : fam}`.replace(/^./, (c) => c.toUpperCase());
}

export function materialSwatches(e: EntityLike): Swatch[] {
  const cp = isRec(e.custom_properties) ? e.custom_properties : {};
  const kind = entityKind(e);
  const out: Swatch[] = [];
  const push = (title: string, v: unknown) => {
    const rgb = asRgb(v);
    if (rgb) out.push({ title, rgb, name: colourName(rgb) });
  };
  const app = isRec(cp.appearance) ? cp.appearance : null;
  const measured = app && app.color_state === "measured" ? app.color_rgb : null;
  if (kind === "room") {
    push("Surface finish (estimated)", cp.wall_color_rgb);
    push("Floor material (estimated)", cp.floor_color_rgb);
    const layers = Array.isArray(cp.ceiling_layers) ? cp.ceiling_layers : [];
    if (isRec(layers[0])) push("Ceiling finish (estimated)", layers[0].color_rgb);
  } else if (kind === "floor") {
    push("Floor material (estimated)", measured ?? cp.mean_color_rgb);
  } else if (kind === "ceiling") {
    push("Ceiling finish (estimated)", measured ?? cp.mean_color_rgb);
  } else if (kind === "furniture") {
    push("Surface colour (estimated)", measured ?? cp.mean_color_rgb);
  } else {
    push("Surface finish (estimated)", measured ?? cp.mean_color_rgb);
  }
  return out;
}

/* ---------------- world overview ---------------- */

/** Solved shells replace the raw evidence planes they were fitted to; counting both would double every wall. */
export const isEvidencePlane = (e: EntityLike): boolean => {
  const cp = isRec(e.custom_properties) ? e.custom_properties : {};
  return e.id.startsWith("struct-plane") && (cp.slab_id !== undefined || cp.slab_wall_id !== undefined);
};

export interface WorldCounts {
  rooms: number; walls: number; doors: number; windows: number; furniture: number; other: number; total: number;
}
export function countWorld(entities: Record<string, EntityLike> | null | undefined): WorldCounts {
  const c: WorldCounts = { rooms: 0, walls: 0, doors: 0, windows: 0, furniture: 0, other: 0, total: 0 };
  for (const e of Object.values(entities ?? {})) {
    if (isEvidencePlane(e)) continue;
    c.total++;
    const k = entityKind(e);
    if (k === "room") c.rooms++;
    else if (k === "wall") c.walls++;
    else if (k === "door") c.doors++;
    else if (k === "window") c.windows++;
    else if (k === "furniture") c.furniture++;
    else c.other++;
  }
  return c;
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
export function describeWorldCounts(c: WorldCounts): string {
  const parts = [
    c.rooms && plural(c.rooms, "room"), c.walls && plural(c.walls, "wall"), c.doors && plural(c.doors, "door"),
    c.windows && plural(c.windows, "window"), c.furniture && plural(c.furniture, "piece of furniture", "pieces of furniture"),
  ].filter(Boolean) as string[];
  return parts.length ? parts.join(", ") : "Nothing recognised yet";
}

export type ScaleViewLike = { kind: "relative" | "estimated" | "measured" | "unknown" };
export function scaleSentence(v: ScaleViewLike): string {
  switch (v.kind) {
    case "relative": return "Not to scale yet. Sizes are in model units, not metres. Enter one known distance to fix this.";
    case "estimated": return "Scaled from a typical size you accepted, so distances are estimates in metres.";
    case "measured": return "In metres, from a distance you measured.";
    default: return "Scale was not recorded for this world.";
  }
}
export function anyUncalibrated(entities: Record<string, EntityLike> | null | undefined): boolean {
  for (const e of Object.values(entities ?? {})) {
    const dims = isRec(e.custom_properties) && isRec(e.custom_properties.dimensions) ? e.custom_properties.dimensions : null;
    if (dims && Object.values(dims).some((d) => isRec(d) && typeof d.unit === "string" && d.unit.startsWith("model_units"))) return true;
  }
  return false;
}
