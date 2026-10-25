import { apiGet, apiPost } from "./client";
import type { CompletionDiff, CompletionProposalBody, CompletionProposalRow } from "@/lib/procedural/completion-model";

const base = (worldId: string) => `/api/worlds/${encodeURIComponent(worldId)}/completion/proposals`;

export interface CompletionCreated {
  proposals: CompletionProposalRow[];
  found: number;
  requested: number;
  note: string;
}

/** POST .../completion/proposals -- stores 1-3 GENERATED alternatives; commits nothing. 422 text (e.g. "no usable shell") is carried by the ApiError. */
export function createCompletionProposals(worldId: string, body: CompletionProposalBody): Promise<CompletionCreated> {
  return apiPost<CompletionCreated>(base(worldId), body);
}

export function listCompletionProposals(worldId: string): Promise<CompletionProposalRow[]> {
  return apiGet<CompletionProposalRow[]>(base(worldId));
}

/** GET .../{pid}/diff -- read-only existing-vs-proposed. */
export function getCompletionDiff(worldId: string, pid: string): Promise<CompletionDiff> {
  return apiGet<CompletionDiff>(`${base(worldId)}/${encodeURIComponent(pid)}/diff`);
}

export interface CompletionAccepted {
  version_id: string;
  entity_ids_added: string[];
  proposal: CompletionProposalRow;
  provenance: "GENERATED";
}

/** POST .../{pid}/accept -- commits as a new version. 409 text (shell changed / name taken / already decided) is carried by the ApiError. */
export function acceptCompletionProposal(worldId: string, pid: string): Promise<CompletionAccepted> {
  return apiPost<CompletionAccepted>(`${base(worldId)}/${encodeURIComponent(pid)}/accept`);
}

export function rejectCompletionProposal(worldId: string, pid: string): Promise<CompletionProposalRow> {
  return apiPost<CompletionProposalRow>(`${base(worldId)}/${encodeURIComponent(pid)}/reject`);
}
