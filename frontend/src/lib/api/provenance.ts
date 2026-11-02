import { apiGet, isApiError } from "./client";

export interface EntityEvidenceRef {
  evidence_id: string;
  evidence_name: string;
  evidence_type: string;
  observation_id?: string;
  session_id?: string;
}

export interface EntityProvenance {
  entity_id: string;
  version_id: string;
  provenance: string | null;
  trace_level: "observation" | "session" | "none";
  evidence: EntityEvidenceRef[];
  source_session_ids?: string[];
  reason?: string;
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
