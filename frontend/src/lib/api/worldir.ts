/**
 * WorldIR / points / cameras / report — the world's OWN reconstruction facts.
 *
 * Contract: these functions THROW on failure. They never return `null` to mean
 * "could not ask the backend".
 *
 * `null` is a legitimate value for exactly one reason — a world with no
 * committed WorldStore version has nothing to return — and the backend already
 * models that as a 404, which the client surfaces as `ApiError{code:
 * "not_found"}`. Callers that want to render "not reconstructed yet" catch that
 * one code; they must not be handed `null` for a 503, a 500, a timeout or a
 * parse failure, because `null` is what the UI renders as an empty scene.
 *
 * The previous revision of this file wrapped every call in `try { ... } catch {
 * return null }`. That collapsed "this world has never been reconstructed"
 * (honest, expected) and "the API is down" (an outage) into one identical
 * `null`, so a total backend failure rendered as a perfectly normal empty
 * viewport with no error, no retry, and no way for a user to tell the
 * difference.
 */
import type { CamerasPayload, PipelineReport, WorldIR } from "@/types/worldir";
import { apiGet, apiGetBinary, isApiError } from "./client";

/**
 * The world has no reconstruction version yet. This is a real, expected state
 * with an honest empty UI — not an error and not an outage.
 */
export function isNotReconstructed(err: unknown): boolean {
  return isApiError(err) && err.code === "not_found";
}

/** Full WorldIR for a world's current WorldStore version. Throws on failure. */
export function fetchWorldIR(worldId: string, signal?: AbortSignal): Promise<WorldIR> {
  return apiGet<WorldIR>(`/api/worlds/${encodeURIComponent(worldId)}/worldir`, { signal });
}

/**
 * Binary point/mesh stream for the world's current version.
 *
 * Uses the same transport (`apiGet` + the ApiError contract) as the JSON
 * endpoints rather than a bare `fetch`, so a 404 ("no geometry") and a 503
 * ("API down") stay distinguishable instead of both collapsing to `null`.
 */
export async function fetchWorldPoints(worldId: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  const data = await apiGetBinary(`/api/worlds/${encodeURIComponent(worldId)}/points`, { signal });
  if (!(data instanceof ArrayBuffer)) {
    throw new Error("Point cloud response was not a binary stream.");
  }
  return data;
}

/** Camera poses registered on the world's current version. Throws on failure. */
export function fetchWorldCameras(worldId: string, signal?: AbortSignal): Promise<CamerasPayload> {
  return apiGet<CamerasPayload>(`/api/worlds/${encodeURIComponent(worldId)}/cameras`, { signal });
}

/** Real compile-pipeline stage facts (the CLI's report.json) for the world's
 *  current version. Throws on failure. */
export function fetchWorldReport(worldId: string, signal?: AbortSignal): Promise<PipelineReport> {
  return apiGet<PipelineReport>(`/api/worlds/${encodeURIComponent(worldId)}/report`, { signal });
}
