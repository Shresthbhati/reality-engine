/**
 * Evidence + uploads resources.
 *
 * Uploads are real multipart POSTs to `/api/uploads`; the server computes the
 * sha256, stores the artifact content-addressed, creates the Evidence row and
 * enqueues PROCESS_EVIDENCE. The returned ids are the only source of truth
 * about what happened.
 */
import { ApiError, apiGet, apiPostForm, withQuery } from "./client";
import { toEvidenceRow } from "./adapters";
import type { ApiList, EvidenceDto, LocationDto, UploadResult } from "./types";
import type { EvidenceRow } from "@/lib/types";

export async function listEvidence(options: { sessionId?: string | null; signal?: AbortSignal } = {}): Promise<{
  rows: EvidenceRow[];
  dtos: EvidenceDto[];
}> {
  const data = await apiGet<ApiList<EvidenceDto>>(
    withQuery("/api/evidence", { session_id: options.sessionId }),
    { signal: options.signal },
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

/**
 * Uploads are real multipart POSTs to `/api/uploads`; the server computes the
 * sha256, stores the artifact content-addressed, creates the Evidence row and
 * enqueues PROCESS_EVIDENCE.
 *
 * Success proof: HTTP 201 carrying an `evidence_id`. A 2xx without one means no
 * row was created, so it is rejected rather than shown as ingested.
 */
export function uploadEvidence(file: File | Blob, sessionId: string | null): Promise<UploadResult> {
  return apiPostForm<UploadResult>(withQuery("/api/uploads", { session_id: sessionId }), { file });
}

/**
 * Delete an Evidence item.
 *
 * The backend answers 204 No Content on success (apps/api/routes_misc.py,
 * `status_code=204`). A 404 means it was already gone. Anything else throws
 * an ApiError so the caller keeps the row visible -- an item is never removed
 * from the UI on a failure.
 */
export async function deleteEvidence(id: string): Promise<void> {
  const res = await fetch(`/api/evidence/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (res.status === 204) return;

  const data = (await res.json().catch(() => null)) as
    | { detail?: string; error?: string }
    | null;

  if (res.ok) {
    // A 2xx that is not the documented 204 is an unexpected contract; do not
    // treat it as a completed deletion.
    throw new Error(
      `The API answered ${res.status} instead of 204; the deletion could not be confirmed.`,
    );
  }

  throw new ApiError({
    message:
      data?.detail ??
      data?.error ??
      (res.status === 404
        ? "That evidence no longer exists."
        : `Delete failed with HTTP ${res.status}. The item was not removed.`),
    status: res.status,
    code:
      res.status === 404
        ? "not_found"
        : res.status >= 500
          ? "server_error"
          : "invalid_request",
    retryable: res.status === 0 || res.status === 408 || res.status === 429 || res.status >= 500,
    requestId: res.headers.get("x-request-id"),
    details: null,
  });
}
