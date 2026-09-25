import { apiGet } from "./client";
import type { Entity } from "@/types/worldir";

export interface QueryResult extends Entity {
  distance_m?: number;
}

export async function queryNearest(
  worldId: string,
  point: { x: number; y: number; z: number },
  k: number,
  type?: string,
): Promise<QueryResult[]> {
  const q = new URLSearchParams({
    x: String(point.x), y: String(point.y), z: String(point.z), k: String(k),
  });
  if (type) q.set("type", type);
  const res = await apiGet<{ results: QueryResult[] }>(
    `/api/worlds/${encodeURIComponent(worldId)}/query/nearest?${q.toString()}`,
  );
  return res.results;
}

export async function queryWithinRadius(
  worldId: string,
  point: { x: number; y: number; z: number },
  radius: number,
  type?: string,
): Promise<QueryResult[]> {
  const q = new URLSearchParams({
    x: String(point.x), y: String(point.y), z: String(point.z), radius: String(radius),
  });
  if (type) q.set("type", type);
  const res = await apiGet<{ results: QueryResult[] }>(
    `/api/worlds/${encodeURIComponent(worldId)}/query/within_radius?${q.toString()}`,
  );
  return res.results;
}

export async function queryContents(worldId: string, entityId: string): Promise<Entity[]> {
  const res = await apiGet<{ contents: Entity[] }>(
    `/api/worlds/${encodeURIComponent(worldId)}/query/contents/${encodeURIComponent(entityId)}`,
  );
  return res.contents;
}

export async function queryContainer(worldId: string, entityId: string): Promise<Entity | null> {
  const res = await apiGet<{ container: Entity | null }>(
    `/api/worlds/${encodeURIComponent(worldId)}/query/container/${encodeURIComponent(entityId)}`,
  );
  return res.container;
}
