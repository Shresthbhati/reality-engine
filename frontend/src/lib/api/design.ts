import { apiGet, apiPost } from "./client";
import type { CatalogItem } from "@/lib/design/design-model";

export * from "@/lib/design/design-model";

export interface DesignResult {
  version_id: string;
  world_id: string;
  operation?: string;
  duplicate?: boolean;
}

const furniture = (worldId: string, entityId: string) =>
  `/api/worlds/${encodeURIComponent(worldId)}/furniture/${encodeURIComponent(entityId)}`;

export async function getFurnitureCatalog(): Promise<CatalogItem[]> {
  return (await apiGet<{ items: CatalogItem[] }>("/api/furniture-catalog")).items;
}

/** `parent` is the version the designer is looking at; the server answers 409 if the world has moved on since. */
export function moveFurniture(worldId: string, entityId: string, x: number, z: number, parent?: string): Promise<DesignResult> {
  return apiPost<DesignResult>(`${furniture(worldId, entityId)}/move`, { x, z, parent_version_id: parent ?? null });
}

export function rotateFurniture(worldId: string, entityId: string, yawRad: number, parent?: string): Promise<DesignResult> {
  return apiPost<DesignResult>(`${furniture(worldId, entityId)}/rotate`, { yaw_rad: yawRad, parent_version_id: parent ?? null });
}

export function replaceFurniture(worldId: string, entityId: string, catalogId: string, parent?: string): Promise<DesignResult> {
  return apiPost<DesignResult>(`${furniture(worldId, entityId)}/replace`, { catalog_id: catalogId, parent_version_id: parent ?? null });
}

export function deleteFurniture(worldId: string, entityId: string, parent?: string): Promise<DesignResult> {
  return apiPost<DesignResult>(`${furniture(worldId, entityId)}/delete`, { parent_version_id: parent ?? null });
}
