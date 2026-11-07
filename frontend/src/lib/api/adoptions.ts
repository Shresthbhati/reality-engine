/**
 * World adoptions API client (Task U.7 / Decision 027).
 *
 * Progressive project evolution: compares candidate evidence sessions
 * against base world models, derives 5-state semantic change sets,
 * and executes atomic version commits with Rule 18 edit preservation.
 */

import { apiGet, apiPost } from "./client";

export type AlignmentTier = "HIGH" | "MODERATE" | "LOW";

export interface AdoptionChangeSet {
  matched: string[];
  new: string[];
  updated: string[];
  uncertain: string[];
  conflicting: string[];
}

export interface AdoptionSummary {
  total_candidate_entities: number;
  matched_count: number;
  new_count: number;
  updated_count: number;
  uncertain_count: number;
  conflicting_count: number;
}

export interface AdoptionProposalDto {
  adoption_id: string;
  world_id: string;
  session_id: string;
  parent_version_id: string;
  alignment_confidence: number;
  alignment_tier: AlignmentTier;
  change_set: AdoptionChangeSet;
  status: "PROPOSED" | "COMMITTED" | "REJECTED" | "ADOPTED";
  created_at: string;
  summary: AdoptionSummary;
  candidate_world?: Record<string, unknown>;
}

export interface ProposeAdoptionParams {
  sessionId: string;
  minConfidence?: number;
  candidateWorld?: Record<string, unknown>;
  scaleFactor?: number;
  overlapRatio?: number;
  featureInlierRatio?: number;
}

export interface CommitAdoptionParams {
  resolutions?: Record<string, "keep_edit" | "accept_reconstruction">;
  note?: string;
}

export interface CommitAdoptionResult {
  adoption_id: string;
  world_id: string;
  version_id: string;
  parent_version_id: string | null;
  status: string;
  stats: Record<string, number>;
  message: string;
}

/** Propose adopting an evidence capture session into a target world. */
export async function proposeAdoption(
  worldId: string,
  params: ProposeAdoptionParams,
): Promise<AdoptionProposalDto> {
  return apiPost<AdoptionProposalDto>(`/api/worlds/${encodeURIComponent(worldId)}/adoptions`, {
    session_id: params.sessionId,
    min_confidence: params.minConfidence ?? 0.60,
    candidate_world: params.candidateWorld,
    scale_factor: params.scaleFactor ?? 1.0,
    overlap_ratio: params.overlapRatio,
    feature_inlier_ratio: params.featureInlierRatio,
  });
}

/** Get adoption proposal details and current change set. */
export async function getAdoption(
  worldId: string,
  adoptionId: string,
): Promise<AdoptionProposalDto> {
  return apiGet<AdoptionProposalDto>(
    `/api/worlds/${encodeURIComponent(worldId)}/adoptions/${encodeURIComponent(adoptionId)}`,
  );
}

/** Commit an adoption proposal atomically to produce a new world version. */
export async function commitAdoption(
  worldId: string,
  adoptionId: string,
  params: CommitAdoptionParams = {},
): Promise<CommitAdoptionResult> {
  return apiPost<CommitAdoptionResult>(
    `/api/worlds/${encodeURIComponent(worldId)}/adoptions/${encodeURIComponent(adoptionId)}/commit`,
    {
      resolutions: params.resolutions ?? {},
      note: params.note,
    },
  );
}

/** Reject / cancel an adoption proposal. */
export async function rejectAdoption(
  worldId: string,
  adoptionId: string,
  reason?: string,
): Promise<{ adoption_id: string; status: "REJECTED"; message: string }> {
  return apiPost(
    `/api/worlds/${encodeURIComponent(worldId)}/adoptions/${encodeURIComponent(adoptionId)}/reject`,
    {
      reason: reason ?? "Rejected by user",
    },
  );
}
