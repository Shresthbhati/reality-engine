import type { CamerasPayload, WorldIR } from "@/types/worldir";
import { apiGet } from "./client";

export async function fetchWorldIR(worldId: string): Promise<WorldIR | null> {
  try {
    return await apiGet<WorldIR>(`/api/worlds/${encodeURIComponent(worldId)}/worldir`);
  } catch {
    return null;
  }
}

export async function fetchWorldPoints(worldId: string): Promise<ArrayBuffer | null> {
  try {
    const res = await fetch(`/api/worlds/${encodeURIComponent(worldId)}/points`);
    if (!res.ok) return null;
    return await res.arrayBuffer();
  } catch {
    return null;
  }
}

export async function fetchWorldCameras(worldId: string): Promise<CamerasPayload | null> {
  try {
    return await apiGet<CamerasPayload>(`/api/worlds/${encodeURIComponent(worldId)}/cameras`);
  } catch {
    return null;
  }
}
