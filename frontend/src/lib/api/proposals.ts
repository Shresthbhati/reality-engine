/**
 * Change proposals resource (apps/api/routes_proposals.py).
 *
 * A proposal compares a candidate world with the world's current version and returns per-entity assessments. Nothing
 * changes until it is accepted, and accepting creates a NEW version (409 when HEAD has moved since the proposal).
 */
import { apiGet, apiPost } from "./client";

export type ProposalVerdict = "supported" | "needs_review" | "conflict";
export type ProposalStatus = "pending" | "accepted" | "rejected";
export type ProposalChangeKind = "added" | "removed" | "modified";

export interface ProposalAssessmentDto {
  entity_id: string;
  kind: ProposalChangeKind;
  verdict: ProposalVerdict;
  reasons: string[];
  categories: string[];
  new_evidence: string[];
  old_confidence: number | null;
  new_confidence: number | null;
}

export interface ProposalSummaryDto {
  supported: number;
  needs_review: number;
  conflict: number;
  changes: number;
  diff: Record<string, number>;
  categories: Record<string, number>;
}

export interface ProposalDto {
  proposal_id: string;
  base_version_id: string;
  candidate_artifact_uri: string;
  candidate_artifact_hash: string;
  assessments: ProposalAssessmentDto[];
  summary: ProposalSummaryDto;
  candidate_validation: { valid: boolean; errors: string[]; warnings: string[] };
  source_session_ids: string[];
  status: ProposalStatus;
  /** Present once decided: version_id, applied, overridden_conflicts, skipped, reason. */
  decision: Record<string, unknown>;
}

export interface ProposalListItemDto {
  proposal_id: string;
  base_version_id: string;
  status: ProposalStatus;
  summary: ProposalSummaryDto;
}

export interface AcceptProposalBody {
  /** Exactly these changes are applied; needs_review and conflict changes must be named here. */
  entity_ids: string[];
  /** Conflicting changes being applied anyway; each needs the shared `reason`. */
  override_conflicts: string[];
  reason: string;
}

const base = (worldId: string) => `/api/worlds/${encodeURIComponent(worldId)}/proposals`;

export async function listProposals(worldId: string): Promise<ProposalListItemDto[]> {
  return (await apiGet<{ proposals: ProposalListItemDto[] }>(base(worldId))).proposals;
}

export function getProposal(worldId: string, proposalId: string): Promise<ProposalDto> {
  return apiGet<ProposalDto>(`${base(worldId)}/${encodeURIComponent(proposalId)}`);
}

/** `candidate` is a WorldIR document (WorldIR.to_dict() shape). 422 when it is not one. */
export function createProposal(
  worldId: string,
  candidate: Record<string, unknown>,
  sourceSessionIds: string[] = [],
): Promise<ProposalDto> {
  return apiPost<ProposalDto>(base(worldId), { candidate, source_session_ids: sourceSessionIds });
}

export function acceptProposal(
  worldId: string,
  proposalId: string,
  body: AcceptProposalBody,
): Promise<{ version_id: string; proposal: ProposalDto }> {
  return apiPost(`${base(worldId)}/${encodeURIComponent(proposalId)}/accept`, body);
}

export function rejectProposal(worldId: string, proposalId: string, reason: string): Promise<ProposalDto> {
  return apiPost<ProposalDto>(`${base(worldId)}/${encodeURIComponent(proposalId)}/reject`, { reason });
}
