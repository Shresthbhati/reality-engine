/** Jobs, notifications, activity, status — the platform's own state. */
import { apiGet, withQuery } from "./client";
import { toActivityRow, type ActivityRow } from "./adapters";
import type { ActivityDto, ApiList, JobDto, NotificationDto, StatusDto } from "./types";

export async function listJobs(options: { status?: string | null } = {}): Promise<JobDto[]> {
  const data = await apiGet<ApiList<JobDto>>(withQuery("/api/jobs", { status: options.status }));
  return data.items;
}

export function getJob(id: string): Promise<JobDto> {
  return apiGet<JobDto>(`/api/jobs/${encodeURIComponent(id)}`);
}

/**
 * Outcome of a cancel request.
 *
 * `outcome` deliberately distinguishes three things the backend separates:
 *  - "cancelled"    the job was queued and is now terminal-cancelled (200)
 *  - "cancelling"   a running job was flagged; the worker will stop at its
 *                   next boundary (202). NOT yet cancelled -- the UI must
 *                   keep polling rather than claim the job is done.
 *  - "rejected"     the job was already terminal (409) or unknown (404).
 */
export interface CancelJobResult {
  outcome: "cancelled" | "cancelling" | "rejected";
  jobId: string;
  status: string;
  cancelRequested: boolean;
  /** Present when the backend refused (409/404). */
  reason: string | null;
  /** True when polling `getJob` can still show a change. */
  pending: boolean;
}

/**
 * Request cancellation of a reconstruction job.
 *
 * A 202 is an acknowledgement that cancellation was REQUESTED, not that the job
 * stopped. Treating 202 as "cancelled" would display a terminal state the
 * backend has not committed; the caller must keep polling until the job itself
 * reports `cancelled`.
 */
export async function cancelJob(id: string): Promise<CancelJobResult> {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
    const data = (await res.json().catch(() => null)) as {
      id?: string;
      status?: string;
      cancel_requested?: boolean;
      detail?: string;
      error?: string;
    } | null;

    if (res.status === 409 || res.status === 404) {
      return {
        outcome: "rejected",
        jobId: id,
        status: data?.status ?? "unknown",
        cancelRequested: false,
        reason:
          data?.detail ??
          data?.error ??
          (res.status === 404
            ? "That job no longer exists."
            : "This job has already finished and cannot be cancelled."),
        pending: false,
      };
    }

    if (!res.ok) {
      throw new Error(
        data?.detail ?? data?.error ?? `Cancel request failed with HTTP ${res.status}.`,
      );
    }

    const status = data?.status ?? "unknown";
    const cancelRequested = data?.cancel_requested === true;

    // 202 means "asked to stop, winding down". Anything else 2xx with a
    // terminal status means it is already stopped.
    if (res.status === 202 || cancelRequested) {
      return {
        outcome: "cancelling",
        jobId: data?.id ?? id,
        status,
        cancelRequested: true,
        reason: null,
        pending: true,
      };
    }

    return {
      outcome: "cancelled",
      jobId: data?.id ?? id,
      status,
      cancelRequested: false,
      reason: null,
      pending: false,
    };
  } catch (err) {
    return {
      outcome: "rejected",
      jobId: id,
      status: "unknown",
      cancelRequested: false,
      reason: err instanceof Error ? err.message : "Cancel request failed.",
      pending: false,
    };
  }
}

export interface NotificationRow {
  id: string;
  type: string;
  title: string;
  body: string | null;
  entityType: string | null;
  entityId: string | null;
  readAt: string | null;
  createdAt: string | null;
}

export async function listNotifications(): Promise<NotificationRow[]> {
  const data = await apiGet<ApiList<NotificationDto>>("/api/notifications");
  return data.items.map((n) => ({
    id: n.id,
    type: n.type,
    title: n.title,
    body: n.body,
    entityType: n.entity_type,
    entityId: n.entity_id,
    readAt: n.read_at,
    createdAt: n.created_at,
  }));
}

export async function listActivity(): Promise<ActivityRow[]> {
  const data = await apiGet<ApiList<ActivityDto>>("/api/activity");
  return data.items.map(toActivityRow);
}

export function getStatus(): Promise<StatusDto> {
  return apiGet<StatusDto>("/api/status", { timeoutMs: 8000 });
}

export function getHealth(): Promise<{ status: string; time: string }> {
  return apiGet<{ status: string; time: string }>("/api/health", { timeoutMs: 5000 });
}
