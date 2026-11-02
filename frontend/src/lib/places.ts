/**
 * Pure view logic for Home and the new-place flow: what a card says, what a build step is, what a check found.
 * No React, no fetching: everything is derived from what the API reported, so it is unit-tested in tests/contract.
 * Consumer words come from lib/copy.ts only.
 */
import { BUILD_STEPS, CHECK, HOME, QUALITY, SCALE, VIEWER, photosPlaced } from "./copy.ts";
import type { QualityBand } from "./copy.ts";

export type BuildStepId = "waiting" | "reading" | "building" | "saving";

/** The three working steps the engine really distinguishes, in order. */
export const WORKING_STEPS: readonly BuildStepId[] = ["reading", "building", "saving"];

const STAGE_TO_STEP: Record<string, BuildStepId> = {
  verifying_artifact: "reading",
  checking_reconstruction_backend: "reading",
  resolving_evidence: "reading",
  analyzing_evidence: "reading",
  building_rough_model: "building",
  reconstructing: "building",
  refining: "building",
  committing_version: "saving",
};

/** Which step a running job is on. A queued job, or an unknown stage, is "waiting": never a step it did not report. */
export function buildStep(job: { status: string; stage: string | null }): BuildStepId {
  if (job.status !== "running") return "waiting";
  return STAGE_TO_STEP[job.stage ?? ""] ?? "reading";
}

export function stepLabel(step: BuildStepId): string {
  return BUILD_STEPS[step];
}

export type BuildOutcome = "building" | "ready" | "partial" | "failed" | "cancelled";

export function buildOutcome(job: { status: string }): BuildOutcome {
  switch (job.status) {
    case "succeeded":
      return "ready";
    case "partial":
      return "partial";
    case "failed":
      return "failed";
    case "cancelled":
      return "cancelled";
    default:
      return "building";
  }
}

export function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m} min ${String(s % 60).padStart(2, "0")} s` : `${s} s`;
}

export function relativeTime(iso: string | null | undefined, now: number): string | null {
  if (!iso) return null;
  // A timestamp without a zone designator is UTC (the API stores UTC); Date.parse would read it as local time.
  const t = Date.parse(/(Z|[+-]\d\d:?\d\d)$/i.test(iso) ? iso : `${iso}Z`);
  if (Number.isNaN(t)) return null;
  const sec = Math.max(0, Math.round((now - t) / 1000));
  if (sec < 60) return "Just now";
  const min = Math.round(sec / 60);
  if (min < 60) return `${min} min ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr} h ago`;
  const day = Math.round(hr / 24);
  return day === 1 ? "Yesterday" : `${day} days ago`;
}

export interface PlaceCardInput {
  name: string;
  photo_count?: number;
  placed_count?: number;
  quality_band?: QualityBand;
  scale_state?: string | null;
  active_job?: { status: string; stage: string | null; progress: number | null } | null;
  updated_at?: string | null;
  created_at?: string | null;
  current_version_id?: string | null;
}

export interface PlaceCardView {
  photoLine: string;
  quality: { band: QualityBand; label: string } | null;
  notRealSize: boolean;
  building: { step: string; percent: number } | null;
  updated: string | null;
}

/** What a card shows. Quality and size chips appear only once there is a model; a building card shows its step. */
export function cardView(p: PlaceCardInput, now: number): PlaceCardView {
  const photos = p.photo_count ?? 0;
  const hasModel = Boolean(p.current_version_id);
  const band = p.quality_band ?? "unknown";
  const job = p.active_job ?? null;
  return {
    photoLine: photos === 0 ? HOME.noPhotosYet : hasModel ? photosPlaced(p.placed_count ?? 0, photos) : `${photos} photo${photos === 1 ? "" : "s"}`,
    quality: hasModel && band !== "unknown" ? { band, label: QUALITY[band] } : null,
    notRealSize: hasModel && p.scale_state === "relative",
    building: job ? { step: stepLabel(buildStep(job)), percent: Math.round(job.progress ?? 0) } : null,
    updated: relativeTime(p.updated_at ?? p.created_at, now),
  };
}

export { SCALE };

// ------------------------------------------------------------------------------------------------ photo check

export interface CheckRowInput {
  name: string;
  kind: string;
  duplicate_of: string | null;
  decodable: boolean | null;
  too_small: boolean;
  relatively_soft: boolean;
  looks_like_photo: boolean | null;
  has_gps: boolean;
  problem?: string | null;
}

/** Plain-language problems for one file. Empty means nothing to flag. */
export function fileProblems(r: CheckRowInput): string[] {
  const out: string[] = [];
  if (r.kind !== "photo" || r.decodable === false) out.push(CHECK.problems.unreadable);
  else {
    if (r.too_small) out.push(CHECK.problems.tooSmall);
    if (r.duplicate_of) out.push(`${CHECK.problems.duplicate} ${r.duplicate_of}`);
    if (r.relatively_soft) out.push(CHECK.problems.soft);
    if (r.looks_like_photo === false) out.push(CHECK.problems.notPhoto);
  }
  return out;
}

export type CheckVerdictKey = keyof typeof CHECK.verdict;

export function verdictMessage(verdict: string): string {
  return (CHECK.verdict as Record<string, string>)[verdict] ?? CHECK.verdict.has_problems;
}

/** Default place name: the folder the photos came from when they share one, otherwise the date. */
export function suggestName(files: { name: string; webkitRelativePath?: string }[], today: Date): string {
  const folders = new Set(
    files.map((f) => (f.webkitRelativePath ?? "").split("/")).filter((parts) => parts.length > 1).map((parts) => parts[0]),
  );
  if (folders.size === 1) return [...folders][0];
  const pad = (n: number) => String(n).padStart(2, "0");
  return `Place ${today.getFullYear()}-${pad(today.getMonth() + 1)}-${pad(today.getDate())}`;
}

// ------------------------------------------------------------------------------------------------ place viewer

const ENTITY_LABELS: Record<string, string> = {
  building: "Building",
  structure: "Structure",
  wall: "Wall",
  floor: "Floor",
  ceiling: "Ceiling",
  roof: "Roof",
  door: "Door",
  window: "Window",
  stairs: "Stairs",
  room: "Room",
  corridor: "Corridor",
  storey: "Floor level",
  opening: "Opening",
  column: "Column",
  beam: "Beam",
  dome: "Dome",
  arch: "Arch",
  vegetation: "Plants",
  terrain: "Ground",
  vehicle: "Vehicle",
  water: "Water",
  debris: "Debris",
  unknown: "Unidentified part",
};

/** A part's kind in plain words; an unlisted kind is shown readable rather than as a code. */
export function entityLabel(type: string | null | undefined): string {
  const t = (type ?? "unknown").toLowerCase();
  if (ENTITY_LABELS[t]) return ENTITY_LABELS[t];
  const spaced = t.replace(/[_-]+/g, " ").trim();
  return spaced ? spaced.charAt(0).toUpperCase() + spaced.slice(1) : ENTITY_LABELS.unknown;
}

export type StatusChipKind = "building" | "unplaced" | "size" | "weak" | "ok";

export interface StatusChipInput {
  building: boolean;
  placed: number;
  total: number;
  scaleState: string | null;
  weakCount: number;
}

/** The single most important fact about the place, in priority order: a running build, photos not placed, not real
 *  size, weak parts, otherwise "Looks good". Never claims "good" while any earlier problem exists. */
export function statusChip(i: StatusChipInput): { kind: StatusChipKind; text: string } {
  if (i.building) return { kind: "building", text: VIEWER.statusBuilding };
  if (i.total > 0 && i.placed < i.total) return { kind: "unplaced", text: photosPlaced(i.placed, i.total) };
  if (i.scaleState === "relative") return { kind: "size", text: SCALE.relative };
  if (i.weakCount > 0) return { kind: "weak", text: i.weakCount + " weak " + (i.weakCount === 1 ? "part" : "parts") };
  return { kind: "ok", text: VIEWER.statusOk };
}

/** What the evidence trace for a selected part lets us say. Never claims more precision than the engine recorded. */
export function traceMessage(level: "observation" | "session" | "none" | null): string {
  if (level === "observation") return VIEWER.traceObserved;
  if (level === "session") return VIEWER.traceSession;
  return VIEWER.traceNone;
}
