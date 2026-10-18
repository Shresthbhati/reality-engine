/**
 * Pure logic for the District tab of the Build dialog and its Generated edit form: early
 * validation, the POST /districts body, the live count note and the PUT patch. No React, no
 * fetch. The backend (procedural/district_grammar.py) stays authoritative; its 409/422 text is
 * shown verbatim. All lengths are metres.
 */

export type DistrictLayout = "grid" | "spine";

export const MAX_PARCELS = 1000;
export const MAX_FAMILIES = 12;
export const DEFAULT_FAMILIES = 3;

export interface DistrictDraft {
  name: string;
  layout: DistrictLayout;
  blocksX: number;
  blocksY: number;
  lot: number;
  streetWidth: number;
  crossEvery: number;
  families: number;
  /** null = no cap (every parcel fully detailed). */
  maxFull: number | null;
  seed: number;
}

export function defaultDistrictDraft(): DistrictDraft {
  return {
    name: "district-1", layout: "grid", blocksX: 3, blocksY: 3, lot: 24, streetWidth: 8,
    crossEvery: 4, families: DEFAULT_FAMILIES, maxFull: 4, seed: 0,
  };
}

const isInt = (v: number) => Number.isInteger(v);
const inRange = (v: number, lo: number, hi: number) => Number.isFinite(v) && v >= lo && v <= hi;

/** Spine layouts always have one row of parcels on each side of the main street. */
export function effectiveBlocksY(d: Pick<DistrictDraft, "layout" | "blocksY">): number {
  return d.layout === "spine" ? 2 : d.blocksY;
}

export function validateDistrict(d: DistrictDraft): string[] {
  const out: string[] = [];
  const by = effectiveBlocksY(d);
  if (!d.name.trim()) out.push("Enter a district name.");
  if (!isInt(d.blocksX) || d.blocksX < 1 || !isInt(by) || by < 1) {
    out.push("Grid size must be whole numbers of at least 1 parcel in each direction.");
  } else if (d.blocksX * by > MAX_PARCELS) {
    out.push(`A district is limited to ${MAX_PARCELS} parcels.`);
  }
  if (!inRange(d.lot, 8, 200)) out.push("Lot size must be between 8 and 200 m.");
  if (!inRange(d.streetWidth, 1, 60)) out.push("Street width must be between 1 and 60 m.");
  if (d.layout === "spine" && (!isInt(d.crossEvery) || d.crossEvery < 1)) {
    out.push("Parcels between cross streets must be a whole number of at least 1.");
  }
  if (!isInt(d.families) || d.families < 1 || d.families > MAX_FAMILIES) {
    out.push(`Families must be a whole number from 1 to ${MAX_FAMILIES}.`);
  }
  if (d.maxFull !== null && (!isInt(d.maxFull) || d.maxFull < 0)) {
    out.push("Fully detailed buildings must be a whole number of 0 or more (or empty for no limit).");
  }
  if (!isInt(d.seed)) out.push("Seed must be a whole number.");
  return out;
}

/** Counts the grammar guarantees for this spec (DistrictSpec.plan). Nothing else is claimed. */
export function districtCounts(d: Pick<DistrictDraft, "layout" | "blocksX" | "blocksY" | "crossEvery">) {
  const by = effectiveBlocksY(d);
  const parcels = d.blocksX * by;
  const streets =
    d.layout === "grid"
      ? d.blocksX + 1 + (by + 1)
      : 1 + (Math.floor((d.blocksX - 1) / Math.max(d.crossEvery, 1)) + 1) + 1;
  return { parcels, streets };
}

/** Live note under the form; "" while the inputs are not valid yet. */
export function districtNote(d: DistrictDraft): string {
  if (validateDistrict(d).length > 0) return "";
  const { parcels, streets } = districtCounts(d);
  const full = d.maxFull === null ? parcels : Math.min(d.maxFull, parcels);
  const rest = parcels - full;
  return (
    `${parcels} parcel${parcels === 1 ? "" : "s"} (one building each) and ${streets} named street${streets === 1 ? "" : "s"}. ` +
    `Up to ${full} building${full === 1 ? "" : "s"} fully detailed, nearest the district centre first; ` +
    `${rest} ${rest === 1 ? "is" : "are"} one box each (footprint only).`
  );
}

export interface DistrictBody {
  name: string;
  layout: DistrictLayout;
  blocks_x: number;
  blocks_y: number;
  lot_w: number;
  lot_d: number;
  street_width: number;
  cross_every?: number;
  families?: { name: string; storeys: number; storey_height: number }[];
  detail: { max_full: number | null };
  seed: number;
}

/** Explicit families stepping evenly from 1 to 8 storeys (3 m each). A count of 3 is sent as
 * "omitted" so the server's seeded defaults apply instead. */
export function explicitFamilies(n: number): { name: string; storeys: number; storey_height: number }[] {
  return Array.from({ length: n }, (_, k) => ({
    name: `family-${k}`,
    storeys: n === 1 ? 3 : 1 + Math.round((k * 7) / (n - 1)),
    storey_height: 3,
  }));
}

export function toDistrictBody(d: DistrictDraft): DistrictBody {
  return {
    name: d.name.trim(),
    layout: d.layout,
    blocks_x: d.blocksX,
    blocks_y: effectiveBlocksY(d),
    lot_w: d.lot,
    lot_d: d.lot,
    street_width: d.streetWidth,
    ...(d.layout === "spine" ? { cross_every: d.crossEvery } : {}),
    ...(d.families !== DEFAULT_FAMILIES ? { families: explicitFamilies(d.families) } : {}),
    detail: { max_full: d.maxFull },
    seed: d.seed,
  };
}

/** Generated-panel edit form: seed and the detail policy. */
export interface DistrictEditForm {
  seed: number;
  /** NaN = empty input = no cap. */
  maxFull: number;
  /** NaN = empty input = no radius. */
  fullWithin: number;
  /** Stored focus, sent back unchanged (the PUT replaces the whole detail object). */
  focus: [number, number] | null;
}

function numOr(v: unknown, fb: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fb;
}

export function districtEditFrom(params: Record<string, unknown>): DistrictEditForm {
  const d = (params.detail ?? {}) as Record<string, unknown>;
  const f = Array.isArray(d.focus) && d.focus.length === 2 ? (d.focus as unknown[]) : null;
  return {
    seed: numOr(params.seed, 0),
    maxFull: numOr(d.max_full, NaN),
    fullWithin: numOr(d.full_within, NaN),
    focus: f ? [numOr(f[0], 0), numOr(f[1], 0)] : null,
  };
}

export function validateDistrictEdit(f: DistrictEditForm): string[] {
  const out: string[] = [];
  if (!Number.isInteger(f.seed)) out.push("Seed must be a whole number.");
  if (!Number.isNaN(f.maxFull) && (!Number.isInteger(f.maxFull) || f.maxFull < 0)) {
    out.push("Fully detailed buildings must be a whole number of 0 or more (or empty for no limit).");
  }
  if (!Number.isNaN(f.fullWithin) && !(Number.isFinite(f.fullWithin) && f.fullWithin >= 0)) {
    out.push("Full-detail radius must be 0 m or more (or empty for no radius).");
  }
  return out;
}

export function districtPatch(f: DistrictEditForm): Record<string, unknown> {
  return {
    seed: f.seed,
    detail: {
      max_full: Number.isNaN(f.maxFull) ? null : f.maxFull,
      full_within: Number.isNaN(f.fullWithin) ? null : f.fullWithin,
      focus: f.focus,
    },
  };
}

/** One-line list summary from stored params. */
export function districtSummary(p: Record<string, unknown>): string {
  const layout = p.layout === "spine" ? "spine" : "grid";
  return `${layout}, ${numOr(p.blocks_x, 0)} x ${numOr(p.blocks_y, 0)} parcels, lot ${numOr(p.lot_w, 0)} x ${numOr(p.lot_d, 0)} m`;
}
