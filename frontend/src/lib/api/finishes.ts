import { apiGet, apiPost } from "./client";
import type { FinishOptions } from "@/lib/design/finish-model";

export * from "@/lib/design/finish-model";

export interface FinishResult { version_id: string; world_id: string; operation?: string }

const surface = (worldId: string, entityId: string) =>
  `/api/worlds/${encodeURIComponent(worldId)}/surfaces/${encodeURIComponent(entityId)}`;

export function getFinishOptions(): Promise<FinishOptions> {
  return apiGet<FinishOptions>("/api/finishes");
}

/** Exactly one of `paint`, `material` (floors) or `colour_rgb`; `also` finishes more surfaces in the same saved version. */
export function setFinish(
  worldId: string, entityId: string,
  choice: { paint?: string; material?: string; colour_rgb?: [number, number, number] },
  also: string[] = [], parent?: string,
): Promise<FinishResult> {
  return apiPost<FinishResult>(`${surface(worldId, entityId)}/finish`, { ...choice, also_entity_ids: also, parent_version_id: parent ?? null });
}

export function clearFinish(worldId: string, entityId: string, also: string[] = [], parent?: string): Promise<FinishResult> {
  return apiPost<FinishResult>(`${surface(worldId, entityId)}/clear-finish`, { also_entity_ids: also, parent_version_id: parent ?? null });
}
