import type { CamerasPayload, PipelineReport, WorldIR } from "@/types/worldir";
import { apiGet } from "./client";
import type { ApiList, JobDto } from "./types";

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

/**
 * The world's most recent reconstruction job, when one exists.
 *
 * Jobs are the only record of WHY a version is missing or partial, so the
 * failure-state UI needs them. Returns null when the world has no job or the
 * API is unreachable -- the caller then says "unknown", not "succeeded".
 */
export async function fetchLatestReconstructJob(
  worldId: string,
): Promise<JobDto | null> {
  try {
    const res = await apiGet<ApiList<JobDto>>("/api/jobs");
    const jobs = res.items.filter(
      (j) => j.type === "RECONSTRUCT_SESSION" && j.status !== "completed",
    );
    if (jobs.length === 0) return null;
    // Newest first; the API already orders by created_at desc.
    return jobs[0] ?? null;
  } catch {
    return null;
  }
}