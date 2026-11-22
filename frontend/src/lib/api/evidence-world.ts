import type { WorldIR } from "@/types/worldir";
import { apiGet, apiPost } from "./client";

/** Client for the read-only evidence-run routes (/api/worlds/{id}/evidence-world). Everything shown comes from what the server
 * persisted for that run; nothing here estimates or defaults a missing piece. */

export interface EvidenceRunSummary {
  run_id: string;
  status: string; // complete | partial | failed | cancelled | unreadable
  counts: Record<string, number> | null;
}

/** What the detection stage persisted (read from disk by the server): which engine ran, whether tiling / sub-part crops / segmentation
 * executed, and why anything did not. `mask_coverage` is null when no box was produced; it is never defaulted to 0. */
export interface EvidenceDetectionSummary {
  status: "ran" | "not_run";
  engine?: string;
  inner_detector?: string;
  images?: number;
  tiles_processed?: number;
  subparts_detected?: number;
  raw_candidates?: number;
  refined_boxes?: number;
  masks?: number;
  mask_coverage?: number | null;
  segmentation?: "ran" | "unavailable";
  segmentation_reason?: string | null;
  reason?: string;
}

/** Verdict derived server-side from persisted measurements only (engine/multiengine/quality_gate.py). */
export interface EvidenceQuality {
  verdict: "READY" | "PARTIAL" | "BLOCKED" | "FAILED";
  reasons: string[];
  gates: { name: string; measured: unknown; status: "pass" | "fail" | "not_measured"; basis: string }[];
}

export interface CoverageFraction { covered: number; of: number; fraction: number | null }

/** Separately measured coverage of the persisted world; `fraction` is null when nothing of that kind exists (never 0). */
export interface EvidenceCoverage {
  spatial_positioning: CoverageFraction;
  geometry_dimensions: CoverageFraction;
  multiview_association: CoverageFraction;
  source_traceability: CoverageFraction;
  provenance_declared: CoverageFraction;
  topology: { errors: number; warnings: number; world_validation_errors: number };
  /** Competing hypotheses the fusion kept instead of merging (never silently resolved). */
  unresolved_conflicts?: { entities: number; hypotheses: number };
}

export interface EvidenceReadiness {
  coverage?: EvidenceCoverage;
  quality?: EvidenceQuality;
  detection?: EvidenceDetectionSummary;
  stage_status: string;
  reason?: string;
  physical_accuracy: string; // "NOT_MEASURED" unless a benchmark says otherwise
  evidence_received?: { sources: number };
  stages?: { task: string; status: string; engines: Record<string, string> }[];
  missing_evidence?: { capture_requests: number; limitations: string[] };
  rejected_predictions?: number;
  scale?: { status: string; physical_metric: boolean; records_by_scale_state: Record<string, number> };
  world: {
    status: string; // persisted | not_persisted | failed | failed_verification
    state?: "empty" | "partial" | "complete";
    version_id?: string;
    entities?: number;
    geometries?: number;
    entities_with_position?: number;
    entities_with_geometry?: number;
    scale_states?: Record<string, number>;
    reason?: string;
  };
  export_ready: boolean;
  metric_export_ready?: boolean;
  export_scale_warning?: string;
  export_not_ready_reason?: string;
}

export interface EvidenceEntityRow {
  entity_id: string;
  ontology_path: string;
  members: string[];
  uncertain: unknown;
}

export interface RejectedPrediction {
  engine_id: string;
  task: string;
  reason: string;
  source_ids?: string[];
}

export interface EvidenceEntityDetail {
  entity: { id: string; name: string; transform: Record<string, unknown> | null; geometry_ids: string[] };
  /** Matrix rows as the server sends them: the untouched observation nested under `observation`. */
  observations: { status: string; observation: { observation_id: string; quantity: string; source_id: string; engine_id: string } }[];
  rejected_observations: unknown[];
  association_decisions: { a: string; b: string; outcome: string; reason: string }[];
  capture_requests: unknown[];
  trace_problems: string[];
}

const base = (worldId: string) => `/api/worlds/${encodeURIComponent(worldId)}/evidence-world`;

export async function listEvidenceRuns(worldId: string): Promise<EvidenceRunSummary[]> {
  return (await apiGet<{ runs: EvidenceRunSummary[] }>(base(worldId))).runs;
}
export const getEvidenceReadiness = (worldId: string, runId: string) =>
  apiGet<EvidenceReadiness>(`${base(worldId)}/${encodeURIComponent(runId)}/readiness`);
export async function listEvidenceEntities(worldId: string, runId: string): Promise<EvidenceEntityRow[]> {
  return (await apiGet<{ entities: EvidenceEntityRow[] }>(`${base(worldId)}/${encodeURIComponent(runId)}/entities`)).entities;
}
export const getEvidenceEntity = (worldId: string, runId: string, entityId: string) =>
  apiGet<EvidenceEntityDetail>(`${base(worldId)}/${encodeURIComponent(runId)}/entities/${encodeURIComponent(entityId)}`);
/** Export the run's persisted world. The server refuses (409) when nothing is positioned, and refuses non-metric geometry unless
 * `acknowledgeUnscaled` is set. Returns the file plus what the server said about its scale. */
export async function exportEvidenceRun(worldId: string, runId: string, format: string, acknowledgeUnscaled: boolean) {
  const res = await fetch(`${base(worldId)}/${encodeURIComponent(runId)}/export`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ format, acknowledge_unscaled: acknowledgeUnscaled }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Export refused (HTTP ${res.status}).`);
  }
  return {
    blob: await res.blob(),
    scale: res.headers.get("x-export-scale") ?? "unknown",
    exported: Number(res.headers.get("x-entities-exported") ?? 0),
    skipped: Number(res.headers.get("x-entities-skipped") ?? 0),
  };
}

export interface EvidenceCorrectionResult {
  version_id: string;
  parent_version_id: string;
  original_evidence_preserved: boolean;
}

/** Reclassify one entity of the run's persisted world. Saves a NEW version; the original evidence is untouched (server-checked). */
export function reclassifyEvidenceEntity(worldId: string, runId: string, entityId: string, newClass: string, note = "", expectedVersionId?: string) {
  return apiPost<EvidenceCorrectionResult>(
    `${base(worldId)}/${encodeURIComponent(runId)}/entities/${encodeURIComponent(entityId)}/reclassify`,
    { new_class: newClass, note, expected_version_id: expectedVersionId },
  );
}

export interface EvidenceMergeResult {
  version_id: string;
  parent_version_id: string;
  members: number;
  rebuilt_from_evidence: boolean;
  position_status: string;
  dimensions_status: string;
}

/** Identity correction: "otherEntityId is the same object as entityId". The server rebuilds the merged entity from the raw evidence of
 * both and saves a new version; it refuses (422) two entities detected in the same photograph unless acknowledged. */
export function mergeEvidenceEntities(worldId: string, runId: string, entityId: string, otherEntityId: string, acknowledgeSameImage = false, note = "", expectedVersionId?: string) {
  return apiPost<EvidenceMergeResult>(
    `${base(worldId)}/${encodeURIComponent(runId)}/entities/${encodeURIComponent(entityId)}/merge`,
    { other_entity_id: otherEntityId, note, acknowledge_same_image: acknowledgeSameImage, expected_version_id: expectedVersionId },
  );
}

export interface EvidenceProposalResult {
  added_entities: number;
  evidence_version: string;
  proposal: { proposal_id: string; summary: { supported: number; needs_review: number; conflict: number; changes: number } };
  frame_check: { basis: string; head_version: string; evidence_scale_states: string[] };
  next: string;
}

/** Propose this run's evidence entities to the world's committed HEAD through the adoption protocol. Nothing is committed: the result is a
 * pending proposal to review and accept (entity by entity). The server refuses (409) unless HEAD is the very version this run accompanied. */
export function proposeEvidenceRun(worldId: string, runId: string, note = "") {
  return apiPost<EvidenceProposalResult>(`${base(worldId)}/${encodeURIComponent(runId)}/propose`, { note });
}

export interface EvidenceSplitResult {
  version_id: string;
  new_entity_id: string;
  rebuilt_from_evidence: boolean;
}

/** Identity correction: "these detections are a different object". The server moves them (and their derived observations) to a NEW
 * entity, rebuilds both from their own raw evidence and saves a new version; it refuses to leave the original with no detection. */
export function splitEvidenceEntity(worldId: string, runId: string, entityId: string, detachObservationIds: string[], note = "", expectedVersionId?: string) {
  return apiPost<EvidenceSplitResult>(
    `${base(worldId)}/${encodeURIComponent(runId)}/entities/${encodeURIComponent(entityId)}/split`,
    { detach_observation_ids: detachObservationIds, note, expected_version_id: expectedVersionId },
  );
}

/** The run's persisted, integrity-verified canonical WorldIR (same schema the viewport already renders). 404 = no persisted world. */
export async function getEvidenceWorld(worldId: string, runId: string): Promise<{ version_id: string; world: WorldIR }> {
  return apiGet(`${base(worldId)}/${encodeURIComponent(runId)}/world`);
}
export async function listRejectedPredictions(worldId: string, runId: string): Promise<RejectedPrediction[]> {
  return (await apiGet<{ rejected: RejectedPrediction[] }>(`${base(worldId)}/${encodeURIComponent(runId)}/rejected`)).rejected;
}

/** The one sentence a reader may be told about a run. A partial or empty world is never worded as a finished reconstruction. */
export function evidenceHeadline(r: EvidenceReadiness): string {
  const w = r.world;
  if (r.stage_status === "unreadable") return `Run unreadable: ${r.reason ?? "snapshot unavailable"}`;
  if (w.status !== "persisted") return `No persisted world (${w.status}${w.reason ? `: ${w.reason}` : ""})`;
  const n = w.entities ?? 0;
  const p = w.entities_with_position ?? 0;
  if (w.state === "empty") return "Empty world: nothing was reconstructed";
  if (w.state === "partial") return `Partial world: ${p} of ${n} entities have a measured position`;
  return `${n} entities positioned; scale ${Object.keys(w.scale_states ?? {}).join(", ") || "undeclared"}; physical accuracy ${r.physical_accuracy.toLowerCase().replace("_", " ")}`;
}
