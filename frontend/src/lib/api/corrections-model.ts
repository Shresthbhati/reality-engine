/**
 * User corrections presentation model & payload contracts (pure, no network imports).
 * Blueprint Task 8.5 & Blueprint 14 (Frontend UX).
 */

export interface CorrectionResult {
  version_id: string;
  world_id: string;
  entity_id: string;
  changed_fields: string[];
  duplicate: boolean;
  operation?: string;
}

export interface ReclassifyParams {
  worldId: string;
  entityId: string;
  newClass: string;
  note?: string;
  parentVersionId?: string;
}

export interface SplitParams {
  worldId: string;
  entityId: string;
  splitAxis?: "x" | "y" | "z";
  splitRatio?: number;
  note?: string;
  parentVersionId?: string;
}

export interface MergeParams {
  worldId: string;
  primaryEntityId: string;
  secondaryEntityIds: string[];
  newName?: string;
  note?: string;
  parentVersionId?: string;
}

export function buildReclassifyPayload(params: ReclassifyParams) {
  if (!params.entityId || !params.newClass) {
    throw new Error("entityId and newClass are required for reclassify");
  }
  return {
    entity_id: params.entityId,
    new_class: params.newClass.trim(),
    note: params.note?.trim() || undefined,
    parent_version_id: params.parentVersionId || undefined,
  };
}

export function buildSplitPayload(params: SplitParams) {
  if (!params.entityId) {
    throw new Error("entityId is required for split");
  }
  const splitAxis = params.splitAxis || "x";
  if (!["x", "y", "z"].includes(splitAxis)) {
    throw new Error(`Invalid splitAxis '${splitAxis}', must be 'x', 'y', or 'z'`);
  }
  const splitRatio = params.splitRatio ?? 0.5;
  if (splitRatio < 0.05 || splitRatio > 0.95) {
    throw new Error(`Invalid splitRatio ${splitRatio}, must be between 0.05 and 0.95`);
  }
  return {
    entity_id: params.entityId,
    split_axis: splitAxis,
    split_ratio: splitRatio,
    note: params.note?.trim() || undefined,
    parent_version_id: params.parentVersionId || undefined,
  };
}

export function buildMergePayload(params: MergeParams) {
  if (!params.primaryEntityId) {
    throw new Error("primaryEntityId is required for merge");
  }
  if (!params.secondaryEntityIds || params.secondaryEntityIds.length === 0) {
    throw new Error("At least one secondaryEntityId is required for merge");
  }
  return {
    primary_entity_id: params.primaryEntityId,
    secondary_entity_ids: [...params.secondaryEntityIds],
    new_name: params.newName?.trim() || undefined,
    note: params.note?.trim() || undefined,
    parent_version_id: params.parentVersionId || undefined,
  };
}
