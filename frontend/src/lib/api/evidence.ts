/**
 * Evidence + uploads resources.
 *
 * Uploads are real multipart POSTs to `/api/uploads`; the server computes the
 * sha256, stores the artifact content-addressed, creates the Evidence row and
 * enqueues PROCESS_EVIDENCE. The returned ids are the only source of truth
 * about what happened.
 */
import { apiDelete, apiGet, apiPostForm, withQuery } from "./client";
import { toEvidenceRow } from "./adapters";
import type { ApiList, EvidenceDto, LocationDto, UploadResult } from "./types";
import type { EvidenceRow } from "@/lib/types";

export async function listEvidence(options: { sessionId?: string | null } = {}): Promise<{
  rows: EvidenceRow[];
  dtos: EvidenceDto[];
}> {
  const data = await apiGet<ApiList<EvidenceDto>>(
    withQuery("/api/evidence", { session_id: options.sessionId }),
  );
  return { rows: data.items.map((d) => toEvidenceRow(d)), dtos: data.items };
}

export async function getEvidence(
  id: string,
): Promise<{ row: EvidenceRow; dto: EvidenceDto; location: LocationDto | null }> {
  const dto = await apiGet<EvidenceDto & { location: LocationDto | null }>(
    `/api/evidence/${encodeURIComponent(id)}`,
  );
  return { row: toEvidenceRow(dto, null, null, null), dto, location: dto.location ?? null };
}

/** Absolute URL for the stored artifact; the browser streams it, nothing is copied. */
export function evidenceArtifactUrl(id: string): string {
  const base = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8100").replace(/\/+$/, "");
  return `${base}/api/evidence/${encodeURIComponent(id)}/artifact`;
}

export function uploadEvidence(file: File | Blob, sessionId: string | null): Promise<UploadResult> {
  return apiPostForm<UploadResult>(withQuery("/api/uploads", { session_id: sessionId }), { file });
}

export function deleteEvidence(id: string): Promise<void> {
  return apiDelete<void>(`/api/evidence/${encodeURIComponent(id)}`);
}
