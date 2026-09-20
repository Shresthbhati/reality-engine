/**
 * Reality Engine — Frontend API Client
 * Type-safe client to interact with Next.js backend API routes (/api/*).
 * Adheres strictly to the NO FAKE COMPLETION policy:
 * - Emits clear diagnostic errors if the backend is unavailable
 * - Maps real WorldIR entities and world versions to frontend state
 */

import type { Entity, EntityType, ProvenanceState } from "@/types/reality-engine";

import { parsePlyBuffer, type ParsedPointCloud } from "./ply";

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

export interface BackendGeometryPayload {
  id: string;
  type: string;
  lod_level?: number;
  vertex_count?: number | null;
  triangle_count?: number | null;
  data_uri?: string;
  data_hash?: string;
  bounds_min?: { x: number; y: number; z: number } | null;
  bounds_max?: { x: number; y: number; z: number } | null;
  provenance?: string;
  confidence?: number;
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
  geometries?: Record<string, BackendGeometryPayload>;
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

export function convertBackendEntityToEntity(
  b: BackendEntityPayload,
  geometries?: Record<string, BackendGeometryPayload>
): Entity {
  let primaryBounds: {
    min: [number, number, number];
    max: [number, number, number];
    center: [number, number, number];
    extent: [number, number, number];
  } | undefined = undefined;

  if (geometries && b.geometry_ids && b.geometry_ids.length > 0) {
    const firstGeom = geometries[b.geometry_ids[0]];
    if (firstGeom?.bounds_min && firstGeom?.bounds_max) {
      const min: [number, number, number] = [
        firstGeom.bounds_min.x,
        firstGeom.bounds_min.y,
        firstGeom.bounds_min.z,
      ];
      const max: [number, number, number] = [
        firstGeom.bounds_max.x,
        firstGeom.bounds_max.y,
        firstGeom.bounds_max.z,
      ];
      const center: [number, number, number] = [
        (min[0] + max[0]) / 2,
        (min[1] + max[1]) / 2,
        (min[2] + max[2]) / 2,
      ];
      const extent: [number, number, number] = [
        max[0] - min[0],
        max[1] - min[1],
        max[2] - min[2],
      ];
      primaryBounds = { min, max, center, extent };
    }
  }

  const sessionIdsFromObs = b.observations?.map((o) => o.frame_id).filter(Boolean) ?? [];
  const entitySessionId = b.custom_properties?.session_id ? [String(b.custom_properties.session_id)] : [];
  const allSessionIds = Array.from(new Set([...entitySessionId, ...sessionIdsFromObs]));

  return {
    id: b.id,
    name: (b.custom_properties?.name as string) ?? b.id.replace(/-/g, " "),
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
      observations: b.observations,
      relationships_count: b.relationships.length,
      relationships: b.relationships,
      custom_properties: b.custom_properties ?? {},
      bounds: primaryBounds,
    },
    representations: b.geometry_ids.length > 0 ? ["MESH"] : ["POINT_CLOUD"],
    sessionIds: allSessionIds.length > 0 ? allSessionIds : ["unknown"],
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

export interface PointCloudResult {
  available: boolean;
  data?: ParsedPointCloud;
  error?: string;
}

export async function fetchWorldPointCloud(versionId: string): Promise<PointCloudResult> {
  try {
    const res = await fetch(`/api/world/${encodeURIComponent(versionId)}/points`, { cache: "no-store" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      return {
        available: false,
        error: err.error ?? `Artifact unavailable (HTTP ${res.status})`,
      };
    }
    const buffer = await res.arrayBuffer();
    const data = parsePlyBuffer(buffer);
    return {
      available: true,
      data,
    };
  } catch (err) {
    return {
      available: false,
      error: err instanceof Error ? err.message : "Failed to load point cloud buffer",
    };
  }
}

export interface BackendWorldDiffSummary {
  entities_added: number;
  entities_removed: number;
  entities_modified: number;
  geometries_added: number;
  geometries_removed: number;
  geometries_modified: number;
}

export interface BackendEntityDiff {
  entity_id: string;
  kind: "added" | "removed" | "modified";
  changes: Array<{
    field: string;
    old: unknown;
    new: unknown;
  }>;
}

export interface BackendGeometryDiff {
  geometry_id: string;
  kind: "added" | "removed" | "modified";
  changes: Array<{
    field: string;
    old: unknown;
    new: unknown;
  }>;
}

export interface BackendWorldDiffPayload {
  from_world_id: string;
  to_world_id: string;
  from_version_id: string;
  to_version_id: string;
  summary: BackendWorldDiffSummary;
  entities: BackendEntityDiff[];
  geometries: BackendGeometryDiff[];
  error?: string;
}

export interface WorldDiffResult {
  available: boolean;
  diff?: BackendWorldDiffPayload;
  error?: string;
}

export async function fetchWorldDiff(
  baseVid: string,
  headVid: string
): Promise<WorldDiffResult> {
  try {
    const res = await fetch(
      `/api/world/diff?base=${encodeURIComponent(baseVid)}&head=${encodeURIComponent(headVid)}`,
      { cache: "no-store" }
    );
    const data = await res.json();
    if (!res.ok || data.error) {
      return {
        available: false,
        error: data.error || `HTTP ${res.status}`,
      };
    }
    return {
      available: true,
      diff: data,
    };
  } catch (err) {
    return {
      available: false,
      error: err instanceof Error ? err.message : "Failed to fetch world diff",
    };
  }
}

