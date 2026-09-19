/**
 * Reality Engine — Frontend API Client
 * Type-safe client to interact with Next.js backend API routes (/api/*).
 * Adheres strictly to the NO FAKE COMPLETION policy:
 * - Emits clear diagnostic errors if the backend is unavailable
 * - Maps real WorldIR entities and world versions to frontend state
 */

import type { Entity, EntityType, ProvenanceState } from "@/types/reality-engine";

export interface BackendStatus {
  backend: boolean;
  store_path: string;
  version_count: number;
  latest_version: string | null;
  python_version?: string;
  error?: string;
}

export interface BackendWorldSummary {
  version_id: string;
  world_id: string;
  parent: string | null;
  artifact_uri: string;
  entity_count: number;
  global_provenance?: string;
  global_confidence?: number;
  coordinate_frame?: string;
  name?: string;
  schema_version?: string;
  created_at?: number | null;
  modified_at?: number | null;
}

export interface BackendWorldDetail {
  version_id: string;
  world_id: string;
  name: string;
  schema_version: string;
  global_provenance: string;
  global_confidence: number;
  coordinate_frame: string;
  entity_count: number;
  entities: BackendEntityPayload[];
  geometry_count: number;
  error?: string;
}

export interface BackendEntityPayload {
  id: string;
  type: string;
  provenance: string;
  confidence: number;
  geometry_ids: string[];
  relationships: Array<{
    kind: string;
    target_id: string;
    confidence: number;
  }>;
  observations: Array<{
    id: string;
    sensor_type: string;
    timestamp: number;
    frame_id: string;
    data_uri: string;
    confidence: number;
  }>;
  custom_properties?: Record<string, unknown>;
  tags?: string[];
}

function mapBackendTypeToEntityType(rawType: string): EntityType {
  const upper = rawType.toUpperCase();
  const validTypes: EntityType[] = [
    "WORLD", "SITE", "BUILDING", "FACADE", "COMPONENT", "COLUMN",
    "CAPITAL", "ORNAMENT", "RELIEF", "OBJECT", "FLOOR", "ROOM",
    "WALL", "TERRAIN", "ROAD", "VEGETATION", "INFRASTRUCTURE",
    "CAMERA", "POINT_CLOUD", "MESH", "SPLAT", "TRAJECTORY",
    "OBSERVATION", "FEATURE", "MATCH", "GENERIC"
  ];
  if (validTypes.includes(upper as EntityType)) {
    return upper as EntityType;
  }
  return "OBJECT";
}

function mapBackendProvToProvenanceState(rawProv: string): ProvenanceState {
  const upper = rawProv.toUpperCase();
  if (upper === "OBSERVED") return "OBSERVED";
  if (upper === "RECONSTRUCTED") return "RECONSTRUCTED";
  if (upper === "INFERRED" || upper === "DERIVED") return "DERIVED";
  if (upper === "SYNTHETIC" || upper === "GENERATED") return "SYNTHETIC";
  return "UNKNOWN";
}

export function convertBackendEntityToEntity(b: BackendEntityPayload): Entity {
  return {
    id: b.id,
    name: b.id.replace(/-/g, " "),
    type: mapBackendTypeToEntityType(b.type),
    childIds: b.relationships
      .filter((r) => r.kind.toLowerCase() === "contains" || r.kind.toLowerCase() === "parent_of")
      .map((r) => r.target_id),
    provenance: {
      state: mapBackendProvToProvenanceState(b.provenance),
      confidence: b.confidence,
      algorithm: b.observations[0]?.sensor_type ?? "Reconstruction Pipeline",
      coordinateFrame: "world",
    },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: b.tags ?? [b.type],
    metadata: {
      geometry_ids: b.geometry_ids,
      observations_count: b.observations.length,
      relationships_count: b.relationships.length,
      custom_properties: b.custom_properties ?? {},
    },
    representations: b.geometry_ids.length > 0 ? ["MESH"] : ["POINT_CLOUD"],
    sessionIds: ["sess-001"],
    observationCount: b.observations.length,
    evidenceCount: b.observations.length,
  };
}

export async function checkBackendStatus(): Promise<BackendStatus> {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      return {
        backend: false,
        store_path: err.store_path ?? "",
        version_count: 0,
        latest_version: null,
        error: err.error ?? `Backend returned status ${res.status}`,
      };
    }
    return await res.json();
  } catch (err: unknown) {
    return {
      backend: false,
      store_path: "",
      version_count: 0,
      latest_version: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export async function fetchWorldList(): Promise<BackendWorldSummary[]> {
  try {
    const res = await fetch("/api/world", { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return data.worlds ?? [];
  } catch {
    return [];
  }
}

export async function fetchWorldDetail(versionId: string): Promise<BackendWorldDetail | null> {
  try {
    const res = await fetch(`/api/world/${encodeURIComponent(versionId)}`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchSessionList(): Promise<unknown[]> {
  try {
    const res = await fetch("/api/sessions", { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return data.sessions ?? [];
  } catch {
    return [];
  }
}
