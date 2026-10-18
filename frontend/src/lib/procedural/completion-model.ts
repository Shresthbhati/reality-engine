/**
 * Pure logic for Complete interior (Phase D): request body, early validation, honest labels and diff
 * summaries for the /completion/proposals API. No React, no fetch (unit-tested by `npm run test:contract`).
 * The backend (apps/api/routes_completion.py) stays authoritative; its 409/422 text is shown verbatim.
 * Everything proposed is GENERATED: never observed, never reconstructed.
 */
import {
  toRequestBody,
  validateApartment,
  type AdjacencyDraft,
  type RoomRow,
} from "./apartment-model.ts";

export type CompletionProgram = "studio" | "one_bed" | "custom";
export const COMPLETION_PROGRAMS: { key: CompletionProgram; label: string }[] = [
  { key: "studio", label: "Studio (living, kitchen, bath)" },
  { key: "one_bed", label: "One bedroom (living, kitchen, bath, bed)" },
  { key: "custom", label: "Custom room list" },
];

export interface CompletionDraft {
  name: string;
  program: CompletionProgram;
  rows: RoomRow[];
  entrance: string;
  adjacency: AdjacencyDraft[];
  seed: number;
  alternatives: number;
}

export interface CompletionProposalBody {
  name: string;
  template?: "studio" | "one_bed";
  rooms?: { id: string; kind: string; min_area: number; max_area: number }[];
  entrance?: string;
  adjacency?: [string, string, string][];
  seed: number;
  alternatives: number;
}

export function validateCompletion(d: CompletionDraft): string[] {
  const out: string[] = [];
  if (!d.name.trim()) out.push("Enter a name for the generated interior.");
  if (d.name.length > 200) out.push("Name must be at most 200 characters.");
  if (!Number.isInteger(d.seed)) out.push("Seed must be a whole number.");
  if (!Number.isInteger(d.alternatives) || d.alternatives < 1 || d.alternatives > 3) {
    out.push("Alternatives must be 1 to 3.");
  }
  if (d.program === "custom") {
    // The footprint comes from the shell, never from the form: reuse the room/adjacency checks only.
    out.push(...validateApartment({ ...d, footprint: null }).filter(
      (m) => m !== "Enter an apartment name." && !/^Seed must/.test(m),
    ));
  }
  return out;
}

export function toProposalBody(d: CompletionDraft): CompletionProposalBody {
  const base = { name: d.name.trim(), seed: d.seed, alternatives: d.alternatives };
  if (d.program !== "custom") return { ...base, template: d.program };
  const b = toRequestBody({ ...d, footprint: null });
  return { ...base, rooms: b.rooms, entrance: b.entrance, adjacency: b.adjacency };
}

// ---- API shapes ------------------------------------------------------------------------------------------------

export interface ShellFootprint {
  x_min: number;
  z_min: number;
  z_max: number;
  width: number;
  depth: number;
  floor_y: number;
  method: string;
}

export interface ProgramRoom {
  id: string;
  kind: string;
  min_area: number;
  max_area: number;
}

export interface ProposalSummary {
  footprint: ShellFootprint;
  unit: string;
  shell_entity_ids: string[];
  program: { rooms: ProgramRoom[]; entrance: string };
  name: string;
}

export type ProposalStatus = "proposed" | "accepted" | "rejected";

export interface CompletionProposalRow {
  id: string;
  world_id: string;
  status: ProposalStatus | string;
  alt_index: number;
  seed: number;
  base_version_id: string | null;
  accepted_version_id: string | null;
  summary: ProposalSummary;
  label: string;
  created_at: string | null;
}

export interface CompletionDiff {
  proposal_id: string;
  status: string;
  unit: string | null;
  footprint: ShellFootprint;
  generated_label: string;
  added: string[];
  changed: string[];
  removed: string[];
  counts: { added: number; changed: number; removed: number };
  provenance: { existing: Record<string, number>; added: Record<string, number> };
}

// ---- honest labels ---------------------------------------------------------------------------------------------

/** Units exactly as the API reports them: "m" only when the world has a metric/authored measurement. */
export function unitText(unit: string | null | undefined): string {
  if (unit === "m") return "m (world has a metric or authored scale)";
  if (unit === "model_units") return "model units (not metres: this world has no metric scale)";
  return `${unit ?? "unknown"} (unit not reported by the API)`;
}

const unitShort = (unit: string | null | undefined) =>
  unit === "m" ? "m" : unit === "model_units" ? "model units" : (unit ?? "?");
const n2 = (v: number) => String(Math.round(v * 100) / 100);

/** "6 x 5 model units, axis-aligned bounding box of the reconstructed shell ...". */
export function footprintLine(fp: ShellFootprint, unit: string | null | undefined): string {
  const how = fp.method === "axis_aligned_bbox"
    ? "axis-aligned bounding box of the reconstructed shell (a rotated or L-shaped shell is approximated by its box)"
    : `measured by method ${fp.method}`;
  return `${n2(fp.width)} x ${n2(fp.depth)} ${unitShort(unit)}, ${how}`;
}

/** Badge text. Accepted proposals say what they became; none is ever called observed or reconstructed. */
export function statusBadge(status: string): string {
  if (status === "proposed") return "PROPOSED";
  if (status === "accepted") return "ACCEPTED - GENERATED";
  if (status === "rejected") return "REJECTED";
  return status.toUpperCase();
}

export function provenanceText(counts: Record<string, number>): string {
  const parts = Object.entries(counts).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => `${v} ${k}`);
  return parts.length ? parts.join(", ") : "none";
}

/** Headline for the diff: counts only, from the server's arrays. */
export function diffSummary(d: CompletionDiff): string {
  const { added, changed, removed } = d.counts;
  return `${added} added, ${changed} changed, ${removed} removed`;
}

/** The diff is only honest when every added entity is GENERATED; false means show a warning. */
export function addedAllGenerated(d: CompletionDiff): boolean {
  const kinds = Object.keys(d.provenance.added);
  return kinds.length > 0 && kinds.every((k) => k === "GENERATED");
}

export function programLine(rooms: ProgramRoom[]): string {
  return rooms.map((r) => `${r.id} (${r.kind}, ${r.min_area}-${r.max_area} m²)`).join(", ");
}

/** Pending proposals first, newest batch first (rows of one batch differ by milliseconds: compare to the second), then alternative order. */
export function sortProposals(rows: CompletionProposalRow[]): CompletionProposalRow[] {
  const rank = (s: string) => (s === "proposed" ? 0 : 1);
  return [...rows].sort((a, b) =>
    rank(a.status) - rank(b.status) ||
    (b.created_at ?? "").slice(0, 19).localeCompare((a.created_at ?? "").slice(0, 19)) ||
    a.alt_index - b.alt_index);
}
