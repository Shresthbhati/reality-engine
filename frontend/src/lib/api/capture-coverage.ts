import { apiGet } from "./client";
import type { CaptureCoverageDto } from "./capture-coverage-model";

export * from "./capture-coverage-model";

/** Plan-view wall coverage for the world's current model (read-only; extra photos use createReconstruction). */
export function getCaptureCoverage(worldId: string) {
  return apiGet<CaptureCoverageDto>(`/api/worlds/${encodeURIComponent(worldId)}/capture-coverage`);
}
