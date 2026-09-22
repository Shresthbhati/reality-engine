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
