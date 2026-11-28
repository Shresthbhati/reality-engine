/** Typed contracts matching Reality Engine FastAPI WorldIR / WorldStore endpoints. */

export type WorkflowStage =
  | 'project'
  | 'upload'
  | 'reconstruct'
  | 'inspect'
  | 'correct'
  | 'compare'
  | 'export';

export interface Vector3Like {
  x: number;
  y: number;
  z: number;
}

export interface WorldGeometry {
  id: string;
  type: string;
  bounds_min?: Vector3Like | null;
  bounds_max?: Vector3Like | null;
  provenance?: string;
}

export interface WorldRelationship {
  kind: string;
  target_id: string;
  confidence?: number | null;
  provenance?: string;
  metadata?: Record<string, unknown>;
}

export interface WorldEntity {
  id: string;
  type: string;
  name: string;
  geometry_ids?: string[];
  relationships?: WorldRelationship[];
  semantic_labels?: string[];
  provenance?: string;
  confidence?: number | null;
  confidence_basis?: string | null;
  uncertainty?: Record<string, unknown> | null;
  custom_properties?: Record<string, unknown>;
}

export interface WorldIRPayload {
  id: string;
  entities: Record<string, WorldEntity>;
  geometries: Record<string, WorldGeometry>;
  metadata?: Record<string, unknown>;
}

export interface WorldSummary {
  id: string;
  name: string;
  current_version_id?: string | null;
  status?: string;
  created_at?: string;
}

export interface WorldVersionInfo {
  id: string;
  parent_version_id?: string | null;
  created_at?: string;
  commit_message?: string | null;
}

export interface PointCloudResponse {
  points: number[][];
  colors?: number[][];
  count?: number;
}

export interface CameraStation {
  id: string;
  position: [number, number, number];
  look_at?: [number, number, number];
  source_image?: string;
}

export interface EntityProvenanceResponse {
  entity_id: string;
  version_id?: string;
  trace_level: 'observation' | 'session' | 'none';
  observations?: Array<Record<string, unknown>>;
  decisions?: Array<Record<string, unknown>>;
  source_sessions?: string[];
  evidence?: Array<Record<string, unknown>>;
  conflicting_evidence?: Array<Record<string, unknown>>;
  missing_evidence?: string[];
}

export interface EntityConfidenceResponse {
  entity_id: string;
  confidence: number | null;
  confidence_basis?: string | null;
  uncertainty?: Record<string, unknown> | null;
}

export interface WorldDiffResponse {
  base_version_id: string;
  head_version_id: string;
  added_entities: string[];
  removed_entities: string[];
  changed_entities: Array<{
    entity_id: string;
    changes: Record<string, unknown>;
  }>;
  summary?: Record<string, unknown>;
}

export interface WorldQualityResponse {
  world_id: string;
  overall_score?: number | null;
  coverage_ratio?: number | null;
  missing_regions?: Array<Record<string, unknown>>;
  warnings?: string[];
}

export interface ExportFormatInfo {
  id: string;
  label: string;
  extension: string;
}
