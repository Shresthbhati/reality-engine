/**
 * Worlds resource.
 *
 * The application World record carries application state; WorldStore stays
 * authoritative for immutable computational versions, referenced here by
 * `current_version_id` only.
 */
import { apiDelete, apiGet, apiPatch, apiPost } from "./client";
import { toWorldRow } from "./adapters";
import type {
  ApiList,
  CreateWorldInput,
  WorldCoverageDto,
  WorldDto,
  WorldIRDto,
  WorldDiffDto,
  CommitRequest,
  CommitResponse,
} from "./types";
import type { WorldRow } from "@/lib/types";
// the real /cameras shape (position_m / rotation_wxyz), not a second hand-written DTO
import type { CamerasPayload } from "@/types/worldir";

export async function listWorlds(status: "active" | "archived" | "trash" = "active"): Promise<{ rows: WorldRow[]; dtos: WorldDto[] }> {
  const url = status === "active" ? "/api/worlds" : `/api/worlds?status=${encodeURIComponent(status)}`;
  const data = await apiGet<ApiList<WorldDto>>(url);
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


export function renameWorld(worldId: string, name: string): Promise<{ id: string; name: string }> {
  return apiPatch<{ id: string; name: string }>(`/api/worlds/${encodeURIComponent(worldId)}`, { name });
}

/** Refused with a 409 ApiError while a build is running. */
export function duplicateWorld(worldId: string): Promise<{ id: string; name: string }> {
  return apiPost<{ id: string; name: string }>(`/api/worlds/${encodeURIComponent(worldId)}/duplicate`);
}

export function archiveWorld(worldId: string): Promise<{ id: string; status: string }> {
  return apiPost<{ id: string; status: string }>(`/api/worlds/${encodeURIComponent(worldId)}/archive`);
}

export function restoreWorld(worldId: string): Promise<{ id: string; status: string }> {
  return apiPost<{ id: string; status: string }>(`/api/worlds/${encodeURIComponent(worldId)}/restore`);
}

export function deleteWorld(worldId: string): Promise<void> {
  return apiDelete<void>(`/api/worlds/${encodeURIComponent(worldId)}`);
}

export function purgeWorld(worldId: string): Promise<void> {
  return apiDelete<void>(`/api/worlds/${encodeURIComponent(worldId)}?permanent=true`);
}


export interface ExtractedAssetDto {
  id: string;
  name: string;
  category: string;
  source_world_id: string;
  source_entity_id: string;
  dimensions: {
    width_m: number;
    depth_m: number;
    height_m: number;
    bounding_box?: { min: number[]; max: number[] };
  };
  provenance: Record<string, unknown>;
  download_url: string;
  created_at: string;
}

export function extractAsset(
  worldId: string,
  entityId: string,
  name?: string,
  category?: string,
): Promise<ExtractedAssetDto> {
  return apiPost<ExtractedAssetDto>(
    `/api/worlds/${encodeURIComponent(worldId)}/extract-asset`,
    { entity_id: entityId, name, category },
  );
}
