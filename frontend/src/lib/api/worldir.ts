import type { CamerasPayload, PipelineReport, WorldIR } from "@/types/worldir";
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

/** Real compile-pipeline stage facts for the world's current version
 * (cameras registered, scale state, entity counts, …) -- the CLI's
 * report.json, bridged through WorldStore. Null (not fabricated data)
 * when no version has been compiled yet. */
export async function fetchWorldReport(worldId: string): Promise<PipelineReport | null> {
  try {
    return await apiGet<PipelineReport>(`/api/worlds/${encodeURIComponent(worldId)}/report`);
  } catch {
    return null;
  }
}
