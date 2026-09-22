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
