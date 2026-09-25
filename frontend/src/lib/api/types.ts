/**
 * DTOs exactly as `apps/api` returns them (FastAPI, snake_case).
 *
 * These mirror the server contract, not the UI model: `adapters.ts` converts
 * them into the row types the screens render. Keeping the two apart is what
 * stops a server rename from silently changing what a page shows.
 */

export interface ApiList<T> {
  items: T[];
}

export interface LocationDto {
  latitude: number;
  longitude: number;
  altitude: number | null;
  accuracy: number | null;
  heading: number | null;
  speed: number | null;
  source: string;
  captured_at: string | null;
}

export interface SessionDto {
  id: string;
  name: string;
  status: string;
  world_id: string | null;
  coordinate_reference_system: string;
  created_at: string | null;
  captured_at: string | null;
  uploaded_at: string | null;
  processing_started_at: string | null;
  processing_completed_at: string | null;
  location: LocationDto | null;
  evidence_count: number;
}

export interface EvidenceDto {
  id: string;
  name: string;
  type: string;
  session_id: string | null;
  processing_state: string;
  mime_type: string | null;
  size: number | null;
  checksum: string | null;
  created_at: string | null;
  processed_at: string | null;
  metadata: Record<string, unknown>;
  location?: LocationDto | null;
}

export interface WorldDto {
  id: string;
  name: string;
  description: string | null;
  status: string;
  latitude: number | null;
  longitude: number | null;
  current_version_id: string | null;
  session_count: number;
  created_at: string | null;
}

export interface WorldCoverageDto {
  available: boolean;
  reason?: string;
  coverage?: Record<string, unknown> | null;
  session_points?: { lat: number; lon: number; accuracy: number | null }[];
}

export interface WorldVersionDto {
  id: string;
  world_id: string;
  parent_version_id: string | null;
  artifact_uri: string | null;
  artifact_hash: string | null;
  source_session_ids: string[];
  changed_entity_ids: string[];
  created_at: string | null;
  is_current: boolean;
}

export interface JobDto {
  id: string;
  type: string;
  entity_type: string;
  entity_id: string;
  status: string;
  stage: string | null;
  attempts: number;
  error: string | null;
  worker_id: string | null;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface NotificationDto {
  id: string;
  type: string;
  title: string;
  body: string | null;
  entity_type: string | null;
  entity_id: string | null;
  read_at: string | null;
  created_at: string | null;
}

export interface ActivityDto {
  id: string;
  type: string;
  entity_type: string | null;
  entity_id: string | null;
  summary: string;
  created_at: string | null;
}

export interface StatusDto {
  database: string;
  worker: string;
  counts: { sessions: number; evidence: number; jobs: number };
  backends: { reconstruction: boolean };
}

export interface SessionTrajectoryDto {
  available: boolean;
  reason?: string;
  points: { latitude: number; longitude: number; altitude?: number | null; timestamp?: string | null }[];
}

export interface CreateSessionInput {
  name: string;
  device_metadata?: Record<string, unknown> | null;
  coordinate_reference_system?: string;
  captured_at?: string | null;
  location?: {
    latitude: number;
    longitude: number;
    altitude?: number | null;
    accuracy?: number | null;
    heading?: number | null;
    speed?: number | null;
    source?: string;
    captured_at?: string | null;
  } | null;
}

export interface CreateWorldInput {
  name: string;
  description?: string | null;
  latitude?: number | null;
  longitude?: number | null;
}

export interface CreateSessionResult {
  id: string;
  name: string;
  status: string;
  world_id: string | null;
  location: LocationDto | null;
}

export interface UploadResult {
  upload_id: string;
  evidence_id: string;
  job_id: string;
  checksum: string;
}

/** WorldIR DTO for /api/worlds/{id}/worldir */
export interface WorldIRDto {
  schema_version: number;
  id: string;
  name: string;
  version: number;
  created_at: number;
  modified_at: number;
  entities: Record<string, WorldIREntityDto>;
  geometries: Record<string, WorldIRGeometryDto>;
  materials: Record<string, unknown>;
  surfaces: Record<string, unknown>;
  components: Record<string, unknown>;
  temporal_state: unknown;
  temporal_events: Record<string, unknown>;
  causal_relations: unknown[];
  main_branch_id: string;
  branches: Record<string, unknown>;
  scenarios: Record<string, unknown>;
  coordinate_frame: string;
  transforms: Record<string, WorldIRTransformDto>;
  observations: Record<string, unknown>;
  global_provenance: string;
  global_confidence: number;
  global_uncertainty: unknown;
  metadata: Record<string, unknown>;
}

export interface WorldIREntityDto {
  id: string;
  type: string;
  name: string;
  geometry_ids: string[];
  transform?: WorldIRTransformDto;
  provenance: string;
  confidence: number;
  metadata?: Record<string, unknown>;
  statement_state?: string;
}

export interface WorldIRGeometryDto {
  id: string;
  type: string;
  lod_level: number;
  vertex_count: number;
  triangle_count?: number;
  data_uri: string | null;
  data_hash: string | null;
  bounds_min: { x: number; y: number; z: number };
  bounds_max: { x: number; y: number; z: number };
  provenance: string;
  confidence: number;
  quality_metrics?: Record<string, unknown>;
  observations?: WorldIRObservationDto[];
}

export interface WorldIRTransformDto {
  position: { x: number; y: number; z: number };
  rotation: { w: number; x: number; y: number; z: number };
}

export interface WorldIRObservationDto {
  id: string;
  sensor_type: string;
  confidence: number;
  metadata?: Record<string, unknown>;
}

/** Cameras payload for /api/worlds/{id}/cameras */
export interface CamerasPayload {
  cameras: Array<{
    entity_id: string;
    position: { x: number; y: number; z: number };
    rotation: { w: number; x: number; y: number; z: number };
    provenance: string;
    confidence: number;
  }>;
}

/** A single field's before/after value, as world_ir.diff.FieldChange.to_dict() emits it. */
export interface FieldChangeDto {
  field: string;
  old: unknown;
  new: unknown;
}

export interface EntityDiffDto {
  entity_id: string;
  kind: "added" | "removed" | "modified";
  changes: FieldChangeDto[];
}

export interface GeometryDiffDto {
  geometry_id: string;
  kind: "added" | "removed" | "modified";
  changes: FieldChangeDto[];
}

/** World diff DTO for /api/worlds/{id}/diff -- mirrors world_ir.diff.WorldDiff.to_dict()
 * exactly (summary() only counts entities; there is no geometry summary). */
export interface WorldDiffDto {
  from_world_id: string;
  to_world_id: string;
  summary: {
    entities_added: number;
    entities_removed: number;
    entities_modified: number;
  };
  entity_diffs: EntityDiffDto[];
  geometry_diffs: GeometryDiffDto[];
}

/** Commit request/response for /api/worlds/{id}/commit */
export interface CommitRequest {
  entity_id: string;
  changes: Record<string, unknown>;
  parent_version_id?: string | null;
  commit_message?: string | null;
}

export interface CommitResponse {
  version_id: string;
  world_id: string;
  entity_id: string;
  changed_fields: string[];
}
