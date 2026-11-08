import { apiGet } from "./client";
import type {
  TemporalDiffSummary,
  TemporalEpochSummary,
} from "@/lib/studio/temporal-model";

export interface WorldEpochsResponse {
  world_id: string;
  current_epoch_id: string | null;
  epochs: TemporalEpochSummary[];
}

export interface TemporalCompareOptions {
  fromEpoch?: string;
  toEpoch?: string;
  toleranceM?: number;
  includeSpatialClusters?: boolean;
}

/**
 * List chronological capture epochs/versions available for a world.
 */
export async function fetchWorldEpochs(worldId: string): Promise<WorldEpochsResponse> {
  return apiGet<WorldEpochsResponse>(`/api/worlds/${encodeURIComponent(worldId)}/epochs`);
}

/**
 * Compute 4D temporal as-built delta between two capture epochs of a world.
 */
export async function fetchTemporalCompare(
  worldId: string,
  options: TemporalCompareOptions = {},
): Promise<TemporalDiffSummary> {
  const params = new URLSearchParams();
  if (options.fromEpoch) params.set("from_epoch", options.fromEpoch);
  if (options.toEpoch) params.set("to_epoch", options.toEpoch);
  if (options.toleranceM !== undefined) params.set("tolerance_m", String(options.toleranceM));
  if (options.includeSpatialClusters) params.set("include_spatial_clusters", "true");

  const queryString = params.toString() ? `?${params.toString()}` : "";
  return apiGet<TemporalDiffSummary>(
    `/api/worlds/${encodeURIComponent(worldId)}/temporal/compare${queryString}`,
  );
}
