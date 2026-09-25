export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface Transform {
  position?: Vector3;
  rotation?: [number, number, number, number]; // wxyz or xyzw
  scale?: Vector3;
}

export interface BoundingBox {
  bounds_min: Vector3;
  bounds_max: Vector3;
}

export interface ObservationMetadata {
  role?: string;
  inlier_count?: number;
  extent_m?: number;
  thickness_m?: number | null;
  plane_id?: string;
  inlier_rms_distance_m?: number;
  classification_note?: string;
  [key: string]: unknown;
}

export interface Observation {
  id: string;
  sensor_type: string;
  timestamp?: number;
  frame_id?: string;
  data_uri?: string;
  data_hash?: string;
  metadata?: ObservationMetadata;
  confidence?: number;
  uncertainty?: {
    confidence?: number;
    [key: string]: unknown;
  };
}

export type ProvenanceType = "OBSERVED" | "INFERRED" | "RECONSTRUCTED" | "SYNTHESIZED" | string;

export interface Entity {
  id: string;
  type: "floor" | "wall" | "ceiling" | "object" | string;
  name?: string;
  parent_id?: string;
  transform?: Transform;
  geometry_ids?: string[];
  material_ids?: string[];
  surface_ids?: string[];
  component_ids?: string[];
  relationships?: Array<{
    target_id: string;
    type: string;
    [key: string]: unknown;
  }>;
  semantic_labels?: string[];
  observations?: Observation[];
  temporal_events?: unknown[];
  provenance?: ProvenanceType;
  confidence?: number;
  uncertainty?: {
    confidence?: number;
    [key: string]: unknown;
  };
  custom_properties?: Record<string, unknown>;
}

export interface Geometry extends BoundingBox {
  id: string;
  type: string;
  lod_level?: number;
  vertex_count?: number;
  triangle_count?: number | null;
  data_uri?: string;
  data_hash?: string;
  provenance?: ProvenanceType;
  confidence?: number;
  observations?: Observation[];
}

export interface DepthStageFact {
  status?: string;
  model?: string;
  maps?: number;
  metricized?: number;
  dense_points?: number;
  per_view?: Array<{
    evidence_id: string;
    residual_median_m?: number;
    inlier_fraction?: number;
    aligned_pixels?: number;
    scale?: number;
    note?: string;
  }>;
  note?: string;
}

export interface WorldMetadata {
  scale?: {
    state?: string;
    meters_per_unit?: number;
    note?: string;
    method?: string;
  };
  frame?: {
    note?: string;
    gravity_aligned?: boolean;
  };
  reconstruction?: {
    backend?: string;
    registration_status?: string;
    cameras_registered?: number;
    cameras_input?: number;
    points?: number;
  };
  depth?: DepthStageFact;
  compile?: {
    entities?: number;
    measurements?: number;
    relationships?: number;
  };
  [key: string]: unknown;
}

export interface WorldIR {
  schema_version?: number;
  id: string;
  name?: string;
  version?: number;
  created_at?: number;
  modified_at?: number;
  coordinate_system?: string;
  entities: Record<string, Entity>;
  geometries?: Record<string, Geometry>;
  metadata?: WorldMetadata;
  [key: string]: unknown;
}

export interface CameraPose {
  id?: string;
  position_m: [number, number, number];
  rotation_wxyz: [number, number, number, number];
  evidence_id?: string;
  timestamp?: number;
}

export interface CamerasPayload {
  image_size?: [number, number];
  cameras: CameraPose[];
}

export interface PipelineReport {
  dataset?: string;
  images_ingested?: number;
  scale_references?: number;
  status?: string;
  runtime_s?: number;
  stages?: Record<string, unknown>;
  outputs?: Record<string, string>;
  world_id?: string;
}

export interface StoredVersion {
  version_id: string;
  world_id: string;
  parent: string | null;
  artifact_uri?: string;
  artifact_hash?: string;
  changed_entity_ids?: string[];
  changed_geometry_ids?: string[];
  source_session_ids?: string[];
  created_at?: string;
}
