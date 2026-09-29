/**
 * The one-action product surface: photos in -> a world (and, later, a better
 * world). Mirrors apps/api/routes_reconstructions.py exactly; nothing here
 * invents state -- every field is what the API measured.
 *
 * Worlds, Sessions and Jobs stay available as advanced navigation, but the
 * user never has to create or attach them: the first upload creates a World,
 * later uploads (worldId given) join the SAME World and rebuild it from all of
 * its evidence.
 */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiGet, apiPost, apiPostFiles } from "./client";

export type ReconstructionState =
  | "CAPTURING"
  | "ANALYZING"
  | "BUILDING_ROUGH_MODEL"
  | "RECONSTRUCTING"
  | "REFINING"
  | "PARTIALLY_COMPLETE"
  | "READY_TO_INSPECT"
  | "NEEDS_MORE_EVIDENCE"
  | "FAILED"
  | "CANCELLED";

export type ModelState = "ROUGH" | "PARTIAL" | "REFINED";
export type QualityLevel = "none" | "low" | "moderate" | "good" | "unknown";

export interface CreateReconstructionResult {
  world_id: string;
  session_id: string;
  job_id: string;
  evidence: { id: string; name: string; type: string; evidence_class: string | null }[];
  rejected: { name: string; reason: string }[];
  coalesced: boolean;
  created_world: boolean;
}

export interface EvidenceStatus {
  id: string;
  name: string;
  type: string;
  size: number | null;
  session_id: string | null;
  evidence_class: string | null;
  class_basis: string | null;
  used_for_geometry: boolean | null;
  registered: boolean | null;
  /** first | new_view | redundant | disconnected | unknown | null (not analysed yet) */
  contribution: string | null;
  in_current_model: boolean;
  /** placement history: "waiting" photos are kept and retried on every rebuild */
  registration?: {
    state: "registered" | "waiting";
    attempts: number;
    ever_registered: boolean;
  } | null;
}

export interface GuidanceItem {
  kind: string;
  message: string;
  basis: string;
}

export interface QualityBar {
  name: string;
  level: QualityLevel;
  basis: string;
}

export interface VersionStatus {
  id: string;
  number: number;
  label: string;
  is_current: boolean;
  parent_version_id: string | null;
  created_at: string | null;
  level: number | null;
  model_state: ModelState | null;
  images_used: number;
}

export interface WorldStatus {
  world_id: string;
  name: string;
  state: ReconstructionState;
  state_label: string;
  in_progress: boolean;
  job: { id: string; status: string; stage: string | null; session_id: string; created_at: string | null } | null;
  has_model: boolean;
  model: {
    version_id: string;
    level: number | null;
    level_name: string | null;
    model_state: ModelState | null;
    outcome: string | null;
    attempts: { level: number; name: string; outcome: string; detail: string }[];
    scale: { state?: string; meters_per_unit?: number | null };
    images_used: number;
    images_registered: number;
    /** measured statements about what this version changed versus the previous one */
    changes?: string[];
    verdict?: "ACCEPT" | "ACCEPT_WITH_UNCERTAINTY" | null;
    uncertainties?: string[];
    /** competing estimates that are kept, with provenance, until later evidence settles them */
    conflicts?: {
      id: string;
      kind: string;
      subject: string;
      status: "unresolved" | "resolved";
      summary: string;
      hypotheses: { source: string; position: number[]; confidence: number | null; provenance: string[] }[];
      history: { version: string; event: string; detail: string }[];
    }[];
  } | null;
  /** the newest run, including one whose result was NOT adopted (the current model was kept) */
  last_run?: {
    adopted: boolean;
    verdict: string | null;
    reasons: string[];
    changes: string[];
    kept_version_id: string | null;
  } | null;
  evidence: EvidenceStatus[];
  evidence_summary: {
    count: number;
    headline: string | null;
    quality: QualityBar[];
    excluded: { evidence_id: string; name: string; kind: string; reason: string }[];
  };
  guidance: GuidanceItem[];
  versions: VersionStatus[];
  failure: { kind: string; message: string | null; job_id: string } | null;
}

/** Drop photos into the engine. No worldId => a new world; with worldId => refine that world. */
export function createReconstruction(
  files: File[],
  opts: { worldId?: string | null; evidenceClass?: string | null; latitude?: number; longitude?: number } = {},
): Promise<CreateReconstructionResult> {
  const fields: Record<string, string> = {};
  if (opts.worldId) fields.world_id = opts.worldId;
  if (opts.evidenceClass) fields.evidence_class = opts.evidenceClass;
  if (opts.latitude !== undefined && opts.longitude !== undefined) {
    fields.latitude = String(opts.latitude);
    fields.longitude = String(opts.longitude);
  }
  return apiPostFiles<CreateReconstructionResult>("/api/reconstructions", "files", files, fields);
}

/** Ask the engine to stop a running reconstruction (takes effect at the next stage boundary). */
export function cancelReconstruction(jobId: string): Promise<{ id: string; status: string }> {
  return apiPost<{ id: string; status: string }>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`);
}

/** Re-run reconstruction for a session whose run failed or was cancelled (its photos are kept). */
export function retryReconstruction(sessionId: string): Promise<{ job_id: string }> {
  return apiPost<{ job_id: string }>(`/api/sessions/${encodeURIComponent(sessionId)}/reconstruct`);
}

export function getWorldStatus(worldId: string): Promise<WorldStatus> {
  return apiGet<WorldStatus>(`/api/worlds/${encodeURIComponent(worldId)}/status`);
}

/**
 * Live world status. Polls fast while the engine is working and slowly once
 * settled; `onSettled` fires once each time an in-progress run finishes so
 * the caller can refetch geometry.
 */
export function useWorldStatus(
  worldId: string | null,
  onSettled?: (status: WorldStatus) => void,
) {
  const [rawStatus, setStatus] = useState<WorldStatus | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [tick, setTick] = useState(0);
  const wasProgressRef = useRef(false);
  const settledRef = useRef(onSettled);
  useEffect(() => {
    settledRef.current = onSettled;
  }, [onSettled]);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!worldId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const load = async () => {
      try {
        const next = await getWorldStatus(worldId);
        if (cancelled) return;
        setStatus(next);
        setError(null);
        if (wasProgressRef.current && !next.in_progress) settledRef.current?.(next);
        wasProgressRef.current = next.in_progress;
        timer = setTimeout(load, next.in_progress ? 1500 : 8000);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err : new Error(String(err)));
        timer = setTimeout(load, 5000);
      }
    };
    wasProgressRef.current = false;
    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [worldId, tick]);

  // Derived, never stored: a status belonging to a previous world (or no world)
  // must not be shown for the current one.
  const status = worldId && rawStatus?.world_id === worldId ? rawStatus : null;
  const loading = Boolean(worldId) && status === null && error === null;
  return { status, error: worldId ? error : null, loading, refresh };
}

/** Plain-language description of how much to trust a model, from what the API measured. */
export function modelStateCopy(state: ModelState | null | undefined): { label: string; detail: string } {
  switch (state) {
    case "ROUGH":
      return { label: "Rough", detail: "A first estimate. Add more views to improve it." };
    case "PARTIAL":
      return { label: "Partial", detail: "Real geometry from several views, but not everything is covered." };
    case "REFINED":
      return {
        label: "Refined",
        detail: "Built from many views with structure recovered. Still a reconstruction, not a survey.",
      };
    default:
      return { label: "No model yet", detail: "" };
  }
}
