/**
 * World Intent resource: persisted intent per world + the honest resource estimate.
 * Errors propagate as ApiError — a failed estimate is shown as an error, never as an empty profile.
 */
import { apiGet, apiPost, apiPut } from "./client";
import type { CapabilityId, ResourceProfileDto, ScaleId } from "@/lib/intent/model";

export interface WorldIntentDto {
  world_id: string;
  scale: ScaleId;
  capabilities: CapabilityId[];
  created_at: string | null;
  updated_at: string | null;
}

export interface EstimateRequest {
  scale: ScaleId;
  capabilities: CapabilityId[];
  file_count: number;
  total_bytes: number;
  video_count: number;
}

export function putWorldIntent(worldId: string, scale: ScaleId, capabilities: CapabilityId[]) {
  return apiPut<WorldIntentDto>(`/api/worlds/${encodeURIComponent(worldId)}/intent`, { scale, capabilities });
}

export function getWorldIntent(worldId: string) {
  return apiGet<WorldIntentDto>(`/api/worlds/${encodeURIComponent(worldId)}/intent`);
}

export function estimateIntent(req: EstimateRequest, signal?: AbortSignal) {
  return apiPost<ResourceProfileDto>("/api/intent/estimate", req, signal ? { signal } : {});
}
