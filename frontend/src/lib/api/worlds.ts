/**
 * Worlds resource.
 *
 * The application World record carries application state; WorldStore stays
 * authoritative for immutable computational versions, referenced here by
 * `current_version_id` only.
 */
import { apiGet, apiPost } from "./client";
import { toWorldRow } from "./adapters";
import type { ApiList, CreateWorldInput, WorldCoverageDto, WorldDto } from "./types";
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

export async function fetchWorldDiff(worldId: string, baseVersion?: string, headVersion?: string) {
  const q = new URLSearchParams();
  if (baseVersion) q.set("base", baseVersion);
  if (headVersion) q.set("head", headVersion);
  const res = await fetch(`/api/worlds/${encodeURIComponent(worldId)}/diff?${q.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch world diff");
  return res.json();
}

export async function commitWorldCorrection(worldId: string, payload: {
  entityId: string;
  changes: Record<string, any>;
  parentVersionId?: string;
  commitMessage?: string;
}) {
  const res = await fetch(`/api/worlds/${encodeURIComponent(worldId)}/commit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to commit world correction");
  return res.json();
}

