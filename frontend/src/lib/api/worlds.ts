/**
 * Worlds resource.
 *
 * The application World record carries application state; WorldStore stays
 * authoritative for immutable computational versions, referenced here by
 * `current_version_id` only.
 */
import { apiGet, apiPost } from "./client";
import { toWorldRow } from "./adapters";
import type {
  ApiList,
  CreateWorldInput,
  WorldCoverageDto,
  WorldDto,
  WorldIRDto,
  CamerasPayload,
  WorldDiffDto,
  CommitRequest,
  CommitResponse,
} from "./types";
import type { WorldRow } from "@/lib/types";

export async function listWorlds(): Promise<{ rows: WorldRow[]; dtos: WorldDto[] }> {
  const data = await apiGet<ApiList<WorldDto>>("/api/worlds");
  return { rows: data.items.map((d) => toWorldRow(d)), dtos: data.items };
}

/** Throws `ApiError{code:"not_found"}` for an unknown id — never a blank world. */
export async function getWorld(id: string): Promise<{ row: WorldRow; dto: WorldDto }> {
  const dto = await apiGet<WorldDto>(`/api/worlds/${encodeURIComponent(id)}`);
  return { row: toWorldRow(dto), dto };
}

export async function listWorldVersions(
  worldId: string,
): Promise<import("./types").WorldVersionDto[]> {
  const data = await apiGet<ApiList<import("./types").WorldVersionDto>>(
    `/api/worlds/${encodeURIComponent(worldId)}/versions`,
  );
  return data.items;
}

export function createWorld(input: CreateWorldInput): Promise<{ id: string; name: string }> {
  return apiPost<{ id: string; name: string }>("/api/worlds", input);
}

export function attachSessionToWorld(worldId: string, sessionId: string) {
  return apiPost<{ session_id: string; world_id: string }>(
    `/api/worlds/${encodeURIComponent(worldId)}/attach/${encodeURIComponent(sessionId)}`,
  );
}

export function getWorldCoverage(worldId: string): Promise<WorldCoverageDto> {
  return apiGet<WorldCoverageDto>(`/api/worlds/${encodeURIComponent(worldId)}/coverage`);
}

/** Fetch full WorldIR for a world's current version. */
export async function getWorldIR(worldId: string): Promise<WorldIRDto> {
  return apiGet<WorldIRDto>(`/api/worlds/${encodeURIComponent(worldId)}/worldir`);
}

/** Fetch camera poses for a world's current version. */
export async function getWorldCameras(worldId: string): Promise<CamerasPayload> {
  return apiGet<CamerasPayload>(`/api/worlds/${encodeURIComponent(worldId)}/cameras`);
}

/** Fetch world diff between two versions. */
export async function getWorldDiff(
  worldId: string,
  baseVersion?: string,
  headVersion?: string,
): Promise<WorldDiffDto> {
  const q = new URLSearchParams();
  if (baseVersion) q.set("base", baseVersion);
  if (headVersion) q.set("head", headVersion);
  return apiGet<WorldDiffDto>(`/api/worlds/${encodeURIComponent(worldId)}/diff?${q.toString()}`);
}

/** Commit a correction to a world entity. */
export async function commitWorldCorrection(
  worldId: string,
  payload: CommitRequest,
): Promise<CommitResponse> {
  return apiPost<CommitResponse>(`/api/worlds/${encodeURIComponent(worldId)}/commit`, payload);
}

