/**
 * Pure logic for the Building tab of the Build dialog: early validation and the
 * POST /buildings body. No React, no fetch. The backend (procedural/building_grammar.py)
 * stays authoritative; its 409/422 text is shown verbatim.
 */

import {
  MAX_SIDE_M,
  validateApartment,
  type AdjacencyDraft,
  type RoomRow,
} from "./apartment-model.ts";
import { H_MAX, H_MIN, MAX_UI_STOREYS } from "./generated-model.ts";

export interface BuildingDraft {
  name: string;
  width: number;
  depth: number;
  storeys: number;
  /** One height (m) applied to every storey; per-storey heights are edited in Generated. */
  height: number;
  /** Optional floor layout: empty rows = a shell with no interior rooms. */
  rows: RoomRow[];
  entrance: string;
  adjacency: AdjacencyDraft[];
  seed: number;
}

export function validateBuilding(d: BuildingDraft): string[] {
  const out: string[] = [];
  if (!d.name.trim()) out.push("Enter a building name.");
  if (![d.width, d.depth].every((v) => Number.isFinite(v) && v > 0 && v <= MAX_SIDE_M)) {
    out.push(`Footprint sides must be between 0 and ${MAX_SIDE_M} m.`);
  }
  if (!Number.isInteger(d.storeys) || d.storeys < 1 || d.storeys > MAX_UI_STOREYS) {
    out.push(`Storeys must be a whole number from 1 to ${MAX_UI_STOREYS}.`);
  }
  if (!Number.isFinite(d.height) || d.height < H_MIN || d.height > H_MAX) {
    out.push(`Storey height must be between ${H_MIN} and ${H_MAX} m.`);
  }
  if (!Number.isInteger(d.seed)) out.push("Seed must be a whole number.");
  if (d.rows.length > 0) {
    // Reuse the apartment checks for the room list (name/footprint/seed already checked above).
    const rest = validateApartment({
      name: d.name || "x", rows: d.rows, entrance: d.entrance, footprint: null,
      adjacency: d.adjacency, seed: Number.isInteger(d.seed) ? d.seed : 0,
    });
    out.push(...rest);
  }
  return out;
}

export interface BuildingBody {
  name: string;
  footprint: [number, number];
  storeys: number;
  storey_heights: number[];
  floor_layout?: {
    rooms: { id: string; kind: string; min_area: number; max_area: number }[];
    entrance: string;
    adjacency: [string, string, string][];
    seed: number;
  };
  seed: number;
}

export function toBuildingBody(d: BuildingDraft): BuildingBody {
  const body: BuildingBody = {
    name: d.name.trim(),
    footprint: [d.width, d.depth],
    storeys: d.storeys,
    storey_heights: Array.from({ length: d.storeys }, () => d.height),
    seed: d.seed,
  };
  if (d.rows.length > 0) {
    body.floor_layout = {
      rooms: d.rows.map((r) => ({ id: r.id.trim(), kind: r.kind, min_area: r.min_area, max_area: r.max_area })),
      entrance: d.entrance,
      adjacency: d.adjacency.map((a) => [a.a, a.b, a.level]),
      seed: d.seed,
    };
  }
  return body;
}

