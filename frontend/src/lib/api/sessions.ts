/**
 * Sessions resource — real capture Sessions from `apps/api`.
 *
 * A Session is first-class and may exist with no World; `worldId: null` is a
 * valid, common state and is never replaced with a placeholder.
 */
import { apiGet, apiPost, withQuery } from "./client";
import { toSessionRow } from "./adapters";
import type {
  ApiList,
  CreateSessionInput,
  CreateSessionResult,
  JobDto,
  SessionDto,
  SessionTrajectoryDto,
} from "./types";
import type { SessionRow } from "@/lib/types";

export async function listSessions(options: { worldId?: string | null } = {}): Promise<{
  rows: SessionRow[];
  dtos: SessionDto[];
}> {
  const data = await apiGet<ApiList<SessionDto>>(withQuery("/api/sessions", { world_id: options.worldId }));
  return { rows: data.items.map((d) => toSessionRow(d)), dtos: data.items };
}

export async function getSession(id: string): Promise<{ row: SessionRow; dto: SessionDto }> {
  const dto = await apiGet<SessionDto>(`/api/sessions/${encodeURIComponent(id)}`);
  return { row: toSessionRow(dto), dto };
}

export function createSession(input: CreateSessionInput): Promise<CreateSessionResult> {
  return apiPost<CreateSessionResult>("/api/sessions", input);
}

export function getSessionLocation(id: string) {
  return apiGet<import("./types").LocationDto>(`/api/sessions/${encodeURIComponent(id)}/location`);
}

export function getSessionTrajectory(id: string): Promise<SessionTrajectoryDto> {
  return apiGet<SessionTrajectoryDto>(`/api/sessions/${encodeURIComponent(id)}/trajectory`);
}

/** Enqueues a real reconstruction job; the job id is returned, not a promise of success. */
export function reconstructSession(id: string): Promise<{ job_id: string; type: string; status: string }> {
  return apiPost<{ job_id: string; type: string; status: string }>(
    `/api/sessions/${encodeURIComponent(id)}/reconstruct`,
  );
}

export async function jobsForSession(id: string): Promise<JobDto[]> {
  const data = await apiGet<ApiList<JobDto>>("/api/jobs");
  return data.items.filter((j) => j.entity_type === "session" && j.entity_id === id);
}

/** Alias used by the Sessions detail page; identical to `jobsForSession`. */
export const getSessionJobs: typeof jobsForSession = jobsForSession;

