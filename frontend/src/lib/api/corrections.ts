/**
 * Client API for user corrections: reclassify, split, merge.
 * Blueprint Task 8.5 & Blueprint 14 (Frontend UX).
 */

import { apiPost } from "./client";
import {
  buildMergePayload,
  buildReclassifyPayload,
  buildSplitPayload,
  type CorrectionResult,
  type MergeParams,
  type ReclassifyParams,
  type SplitParams,
} from "./corrections-model";

export * from "./corrections-model";

/**
 * Reclassify an entity to a new ontology class.
 * Records a user-basis decision and mints a new WorldStore version.
 */
export async function reclassifyEntity(params: ReclassifyParams): Promise<CorrectionResult> {
  const payload = buildReclassifyPayload(params);
  return apiPost<CorrectionResult>(`/api/worlds/${params.worldId}/corrections/reclassify`, payload);
}

/**
 * Split an entity into two along an axis.
 * Records a user-basis decision and mints a new WorldStore version.
 */
export async function splitEntity(params: SplitParams): Promise<CorrectionResult> {
  const payload = buildSplitPayload(params);
  return apiPost<CorrectionResult>(`/api/worlds/${params.worldId}/corrections/split`, payload);
}

/**
 * Merge secondary entities into a primary entity.
 * Records a user-basis decision and mints a new WorldStore version.
 */
export async function mergeEntities(params: MergeParams): Promise<CorrectionResult> {
  const payload = buildMergePayload(params);
  return apiPost<CorrectionResult>(`/api/worlds/${params.worldId}/corrections/merge`, payload);
}
