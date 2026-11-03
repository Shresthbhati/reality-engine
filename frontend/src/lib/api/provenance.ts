import { apiGet, isApiError } from "./client";

export interface EntityEvidenceRef {
  evidence_id: string;
  evidence_name: string;
  evidence_type: string;
  observation_id?: string;
  session_id?: string;
  mask_uri?: string | null;
  bbox_px?: number[] | null;
}

export interface AssociationDecisionRef {
  observation_id: string;
  state: string;
  entity_id: string | null;
  probability: number;
  reason: string;
  dist_m?: number | null;
  mahalanobis_sq?: number | null;
  normal_diff_deg?: number | null;
  iou3d?: number | null;
  geom_verified?: boolean | null;
  candidates?: [string, number][];
}

export interface EntityObservationRef {
  observation_id: string;
  image_id: string;
  label: string;
  mask_uri?: string | null;
  bbox_px?: number[] | null;
  confidence?: number | null;
  visibility?: string | null;
  depth_source?: string | null;
}

export interface EntityProvenance {
  entity_id: string;
  version_id: string;
  provenance: string | null;
  trace_level: "observation" | "session" | "none";
  evidence: EntityEvidenceRef[];
  source_session_ids?: string[];
  reason?: string;
  association_decisions?: AssociationDecisionRef[];
  observations?: EntityObservationRef[];
}

/** Traces an entity back to the real Evidence it was compiled from --
 * GET /api/worlds/{worldId}/entities/{entityId}/provenance. Null (not
 * fabricated data) when the world/entity has no version yet. */
export async function fetchEntityProvenance(
  worldId: string,
  entityId: string,
): Promise<EntityProvenance | null> {
  try {
    return await apiGet<EntityProvenance>(
      `/api/worlds/${encodeURIComponent(worldId)}/entities/${encodeURIComponent(entityId)}/provenance`,
    );
  } catch (err) {
    // 404 is the only honest "nothing to trace". Network failures and 5xx must reach the caller: swallowing them
    // rendered an API outage as "no evidence linked to this entity".
    if (isApiError(err) && err.status === 404) return null;
    throw err;
  }
}
