/**
 * Reality Engine — Core TypeScript Types
 * All domain types used across the UI. These map to backend data models.
 * WIRE: Each interface maps to a corresponding Python dataclass in world_ir/schema.py
 */

// ─── Identity ─────────────────────────────────────────────────────────────────

export type EntityId = string;
export type SessionId = string;
export type WorldId = string;
export type ProjectId = string;
export type BuildId = string;
export type ArtifactId = string;
export type VersionId = string;

// ─── Provenance ───────────────────────────────────────────────────────────────

export type ProvenanceState =
  | "OBSERVED"        // directly measured from real source
  | "RECONSTRUCTED"   // inferred by algorithm
  | "DERIVED"         // computed from other entities
  | "SYNTHETIC"       // procedurally generated
  | "IMPORTED"        // from external dataset
  | "UNKNOWN"
  | "PENDING"         // awaiting validation
  | "FAILED_VALIDATION"; // failed quality gates

export type ConfidenceLevel = "HIGH" | "MID" | "LOW" | "NONE";

export interface Provenance {
  state: ProvenanceState;
  confidence: number;         // 0.0 – 1.0
  uncertainty?: number;       // geometric uncertainty in meters
  algorithm?: string;         // e.g. "COLMAP", "OpenMVS", "SAM"
  timestamp?: string;         // ISO 8601
  coordinateFrame?: string;   // e.g. "WGS84", "LOCAL_ENU"
  version?: string;
  sourceIds?: EntityId[];     // contributing source entities
}

// ─── Geometry ─────────────────────────────────────────────────────────────────

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface Transform {
  position: Vec3;
  rotation: Quaternion;
  scale: Vec3;
}

export interface BoundingBox {
  min: Vec3;
  max: Vec3;
}

export interface GeoPosition {
  latitude: number;
  longitude: number;
  altitude?: number;
  accuracy?: number;
}

// ─── Entity ───────────────────────────────────────────────────────────────────

export type EntityType =
  | "WORLD"
  | "SITE"
  | "BUILDING"
  | "FACADE"
  | "COMPONENT"
  | "COLUMN"
  | "CAPITAL"
  | "ORNAMENT"
  | "RELIEF"
  | "OBJECT"
  | "FLOOR"
  | "ROOM"
  | "WALL"
  | "TERRAIN"
  | "ROAD"
  | "VEGETATION"
  | "INFRASTRUCTURE"
  | "CAMERA"
  | "POINT_CLOUD"
  | "MESH"
  | "SPLAT"
  | "TRAJECTORY"
  | "OBSERVATION"
  | "FEATURE"
  | "MATCH"
  | "GENERIC";

export type EntityVisibility = "VISIBLE" | "HIDDEN" | "SOLO";
export type EntityLock = "UNLOCKED" | "LOCKED" | "READ_ONLY";

export interface Entity {
  id: EntityId;
  name: string;
  type: EntityType;
  parentId?: EntityId;
  childIds: EntityId[];
  transform?: Transform;
  boundingBox?: BoundingBox;
  geoPosition?: GeoPosition;
  provenance: Provenance;
  visibility: EntityVisibility;
  lock: EntityLock;
  tags: string[];
  metadata: Record<string, unknown> & {
    bounds?: {
      min: [number, number, number];
      max: [number, number, number];
      center: [number, number, number];
      extent: [number, number, number];
    };
    observations?: Array<{
      id: string;
      sensor_type: string;
      timestamp: number;
      frame_id: string;
      data_uri: string;
      confidence: number;
    }>;
    relationships?: Array<{
      kind: string;
      target_id: string;
      confidence: number;
    }>;
  };
  representations: RepresentationType[];
  sessionIds: SessionId[];   // which sessions contributed to this entity
  observationCount: number;
  evidenceCount: number;
  scaleLevel?: ScaleLevel;
}

export type RepresentationType =
  | "POINT_CLOUD"
  | "MESH"
  | "GAUSSIAN_SPLAT"
  | "BIM"
  | "LOD0"
  | "LOD1"
  | "LOD2"
  | "SEMANTIC_MASK"
  | "DEPTH_MAP";

// ─── Session ──────────────────────────────────────────────────────────────────

export type SessionStatus =
  | "CREATED"
  | "CAPTURING"
  | "UPLOADING"
  | "QUEUED"
  | "PROCESSING"
  | "RECONSTRUCTED"
  | "FAILED"
  | "ARCHIVED";

export type SessionSourceType =
  | "PHONE"
  | "DRONE"
  | "CAMERA_RIG"
  | "LIDAR"
  | "RGB_D"
  | "STEREO"
  | "VIDEO"
  | "EXISTING_DATASET";

export interface SessionSource {
  type: SessionSourceType;
  deviceId?: string;
  imageCount?: number;
  videoFiles?: number;
  lidarScans?: number;
  hasGNSS: boolean;
  hasIMU: boolean;
  calibrated: boolean;
}

export interface Session {
  id: SessionId;
  projectId: ProjectId;
  name: string;
  status: SessionStatus;
  sources: SessionSource[];
  capturedAt?: string;
  processedAt?: string;
  imageCount: number;
  registeredCount: number;
  reprojectionError?: number;
  pointCount?: number;
  coverage?: GeoPosition[];  // bounding polygon
  quality?: number;          // 0.0 – 1.0
  worldFragmentId?: string;
  buildIds: BuildId[];
  // Extended fields for health dashboard
  gnssMode?: "RTK_FIX" | "RTK_FLOAT" | "PPK" | "NONE";
  bundleAdjustmentResidual?: number;
  imuCalibrated?: boolean;
  cameraCalibrated?: boolean;
  captureDurationSec?: number;
}

// ─── World ────────────────────────────────────────────────────────────────────

export type WorldStatus = "INITIALIZING" | "READY" | "PROCESSING" | "OUTDATED" | "ERROR";

export interface World {
  id: WorldId;
  projectId: ProjectId;
  name: string;
  status: WorldStatus;
  entityCount: number;
  sessionIds: SessionId[];
  versions: VersionId[];
  currentVersion: VersionId;
  bounds?: BoundingBox;
  geoBounds?: {
    north: number;
    south: number;
    east: number;
    west: number;
  };
  createdAt: string;
  updatedAt: string;
}

// ─── Project ──────────────────────────────────────────────────────────────────

export interface Project {
  id: ProjectId;
  name: string;
  description?: string;
  worldId?: WorldId;
  sessionIds: SessionId[];
  buildIds: BuildId[];
  createdAt: string;
  updatedAt: string;
  thumbnail?: string;
}

// ─── Build / Pipeline ─────────────────────────────────────────────────────────

export type PipelineStage =
  | "CAPTURE"
  | "TIME_SYNC"
  | "CALIBRATION"
  | "LOCALIZATION"
  | "REGISTRATION"
  | "SPARSE"
  | "DENSE"
  | "FUSION"
  | "PERCEPTION"
  | "COMPILATION";

export type StageStatus =
  | "PENDING"
  | "QUEUED"
  | "RUNNING"
  | "COMPLETE"
  | "FAILED"
  | "SKIPPED"
  | "WARNING";

export interface PipelineStageInfo {
  stage: PipelineStage;
  status: StageStatus;
  progress: number;          // 0–100
  startedAt?: string;
  completedAt?: string;
  durationMs?: number;
  backend?: string;          // e.g. "COLMAP", "OpenMVS"
  gpuUsage?: number;         // 0–100
  ramUsageMb?: number;
  warnings: string[];
  errors: string[];
  metrics: Record<string, number | string>;
  inputArtifacts: ArtifactId[];
  outputArtifacts: ArtifactId[];
}

export interface Build {
  id: BuildId;
  projectId: ProjectId;
  sessionIds: SessionId[];
  status: StageStatus;
  stages: PipelineStageInfo[];
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  logs: LogEntry[];
  // Extended fields for health dashboard
  name?: string;
  errorMessage?: string;
  worldVersionId?: string;
}

// ─── Evidence ─────────────────────────────────────────────────────────────────

export type EvidenceType =
  | "IMAGE"
  | "FEATURE"
  | "MATCH"
  | "CAMERA_POSE"
  | "DEPTH"
  | "POINT"
  | "OBSERVATION"
  | "LIDAR_SCAN"
  | "IMU_READING"
  | "GNSS_FIX";

export interface Evidence {
  id: EntityId;
  type: EvidenceType;
  entityId: EntityId;
  sessionId?: SessionId;
  confidence: number;
  timestamp?: string;
  parentIds: EntityId[];   // evidence this was derived from
  childIds: EntityId[];    // evidence derived from this
  metadata: Record<string, unknown>;
}

export interface EvidenceChain {
  rootId: EntityId;
  nodes: Evidence[];
  edges: Array<{ from: EntityId; to: EntityId }>;
}

// ─── Diagnostics ──────────────────────────────────────────────────────────────

export interface ComputeMetrics {
  gpuUsage: number;       // 0–100
  gpuVramUsed: number;    // MB
  gpuVramTotal: number;   // MB
  cpuUsage: number;       // 0–100
  ramUsed: number;        // MB
  ramTotal: number;       // MB
  diskRead: number;       // MB/s
  diskWrite: number;      // MB/s
  networkIn: number;      // KB/s
  networkOut: number;     // KB/s
}

export type LogLevel = "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";

export interface LogEntry {
  id: string;
  level: LogLevel;
  timestamp: string;
  module: string;
  message: string;
  data?: Record<string, unknown>;
}

// ─── UI State ─────────────────────────────────────────────────────────────────

export type DensityMode = "comfortable" | "compact" | "dense";

export type WorkspaceId =
  | "capture"
  | "loader"
  | "studio"
  | "benchmarks"
  | "build"
  | "city"
  | "evidence"
  | "diagnostics"
  | "settings";

// ─── Mobile Capture Types ──────────────────────────────────────────────────

export interface MobileSensorStatus {
  cameraReady: boolean;
  gnssFix: boolean;
  gnssAccuracyM: number;
  gnssHz: number;
  imuCalibrated: boolean;
  imuHz: number;
  ptpTimeSynced: boolean;
  ptpDriftMs: number;
  storageAvailableGb: number;
  storageTotalGb: number;
  lensCalibrated: boolean;
}

export interface LiveCaptureQuality {
  overall: "GOOD" | "WARNING" | "CRITICAL";
  motionBlurScore: number;
  lightingScore: number;
  focusScore: number;
  coveragePercentage: number;
  recentWarnings: string[];
  fps: number;
  droppedFrames: number;
  recordedSeconds: number;
  capturedFrames: number;
}

// ─── Validation Types ───────────────────────────────────────────────────────

export type ValidationSeverity = "PASS" | "WARNING" | "ERROR";

export interface ValidationIssue {
  id: string;
  severity: ValidationSeverity;
  title: string;
  description: string;
  affectedCount: number;
  affectedUnit: string;
  evidence: string;
  recommendedAction: string;
}

// ─── Measurements with Uncertainty ──────────────────────────────────────────

export type MeasurementType =
  | "POINT_TO_POINT"
  | "HEIGHT"
  | "WIDTH"
  | "DEPTH"
  | "AREA"
  | "VOLUME"
  | "WALL_THICKNESS"
  | "ROOM_DIMENSION";

export interface Measurement {
  id: string;
  name: string;
  type: MeasurementType;
  value: number;
  unit: "m" | "m²" | "m³" | "mm";
  uncertainty: number; // ± uncertainty in given unit
  entityId?: EntityId;
  points?: [Vec3, Vec3];
  confidence: number;
  timestamp: string;
  method?: string;
  calibrationStatus?: "CALIBRATED" | "UNVERIFIED" | "UNAVAILABLE";
  calibrationAvailable?: boolean;
}

// ─── Benchmark Corpus Types ─────────────────────────────────────────────────

export type BenchmarkCategory =
  | "SKYSCRAPER"
  | "INDIAN_FORT"
  | "WORLD_WONDER"
  | "EUROPEAN_HERITAGE"
  | "OTHER_STRUCTURE";

export type BenchmarkReconstructionStatus =
  | "NOT_STARTED"
  | "EVIDENCE_AVAILABLE"
  | "INGESTED"
  | "RECONSTRUCTION_RUNNING"
  | "RECONSTRUCTED"
  | "PERCEPTION_COMPLETE"
  | "WORLDIR_COMPILED"
  | "VALIDATED"
  | "FAILED"
  | "PARTIAL";

export type CaptureAvailability =
  | "LOCAL_DATASET"
  | "AERIAL_ARCHIVE"
  | "PARTIAL_CAPTURE"
  | "NOT_YET_CAPTURED";

export type GroundTruthAvailability =
  | "CAD_SURVEY"
  | "TERRESTRIAL_LIDAR"
  | "SATELLITE_DEM"
  | "NOT_AVAILABLE";

export interface BenchmarkStructure {
  id: string;
  name: string;
  category: BenchmarkCategory;
  location: string;
  country: string;
  scale: string;
  complexity: "EXTREME" | "VERY_HIGH" | "HIGH" | "MODERATE";
  captureTypes: string[];
  expectedImages: number;
  reconstructionStatus: BenchmarkReconstructionStatus;
  captureAvailability: CaptureAvailability;
  groundTruthAvailability: GroundTruthAvailability;
  worldIRStatus: "COMPILED" | "PARTIAL" | "PENDING";
  architectureTypology: string;
  era?: string;
  heightMeters?: number;
  footprintSqMeters?: number;
  structuralNotes: string;
  keyChallenges: string[];
  benchmarkObjectives: string[];
  evidenceSummary?: {
    photos?: number;
    lidarScans?: number;
    gcpCount?: number;
    datasetSizeGb?: number;
    sourceUri?: string;
  };
  metrics?: {
    registeredCameras?: number;
    sparsePoints?: number;
    densePoints?: number;
    reprojectionErrorPx?: number;
    meanResidualMm?: number;
    uncertaintyMm?: number;
    detectedObjects?: number;
    detectedPlanes?: number;
    detectedRooms?: number;
    processingTimeMin?: number;
  };
}

export interface WorkspaceLayout {
  id: WorkspaceId;
  panels: PanelConfig[];
}

export interface PanelConfig {
  id: string;
  type: string;
  size?: number;
  collapsed?: boolean;
  position?: "left" | "right" | "bottom" | "center";
}

export interface SelectionState {
  selectedEntityIds: EntityId[];
  hoveredEntityId?: EntityId;
  focusedEntityId?: EntityId;
}

// ─── Artifacts ────────────────────────────────────────────────────────────────

export type ArtifactType =
  | "POINT_CLOUD_PLY"
  | "MESH_OBJ"
  | "MESH_USD"
  | "MESH_IFC"
  | "MESH_GLTF"
  | "GAUSSIAN_SPLAT"
  | "CAMERA_TRAJECTORY"
  | "SEMANTIC_MAP"
  | "DEPTH_MAP"
  | "REPORT_PDF";

export interface Artifact {
  id: ArtifactId;
  type: ArtifactType;
  name: string;
  sessionId?: SessionId;
  buildId?: BuildId;
  sizeBytes: number;
  createdAt: string;
  downloadUrl?: string;
}

// ─── Multi-Scale Reality Hierarchy & Detail Levels ────────────────────────────

export type ScaleLevel =
  | 'WORLD'
  | 'SITE'
  | 'BUILDING'
  | 'STRUCTURE'
  | 'FACADE'
  | 'COMPONENT'
  | 'DETAIL'
  | 'MICRO_DETAIL';

// ─── 8 Reconstruction View Modes ──────────────────────────────────────────────

export type ReconstructionViewMode =
  | 'WORLD'
  | 'GEOMETRY'
  | 'DETAIL'
  | 'TEXTURE'
  | 'SEMANTICS'
  | 'EVIDENCE'
  | 'CONFIDENCE'
  | 'PROVENANCE';

// ─── Capture Multi-Pass Guidance ─────────────────────────────────────────────

export type CapturePassType =
  | 'SITE_PASS'
  | 'STRUCTURE_PASS'
  | 'FACADE_PASS'
  | 'DETAIL_PASS'
  | 'MICRO_DETAIL_PASS';

export interface CaptureGuidanceCue {
  id: string;
  level: 'info' | 'warning' | 'critical' | 'success';
  message: string;
  targetPass: CapturePassType;
  suggestedAction: string;
}

// ─── Spatial Detail Coverage Model ───────────────────────────────────────────

export interface DetailCoverageNode {
  id: string;
  name: string;
  path: string;
  scaleLevel: ScaleLevel;
  coveragePercent: number;
  resolutionMm: number;
  evidenceObservations: number;
  registeredCameras: number;
  status: 'HIGH' | 'MED' | 'LOW' | 'UNOBSERVED' | 'INFERRED';
  children?: DetailCoverageNode[];
}

// ─── Evidence-to-Detail Provenance Model ─────────────────────────────────────

export interface DetailProvenanceRecord {
  detailId: string;
  detailName: string;
  parentComponent: string;
  scaleLevel: ScaleLevel;
  observationsCount: number;
  registeredCamerasCount: number;
  passes: CapturePassType[];
  depthMethod: string;
  fusionMethod: string;
  groundResolutionMm: number;
  confidenceScore: number;
  sampleCameraIds: string[];
  geometryLOD: 'MACRO_MASSING' | 'SURFACE_FACET' | 'HIGH_FREQ_RELIEF' | 'MICRO_DISPLACEMENT';
}

// ─── Global Modes (Visual Source of Truth) ───────────────────────────────────

export type GlobalMode =
  | 'EXPLORE'
  | 'INSPECT'
  | 'EDIT'
  | 'CAPTURE'
  | 'BUILD'
  | 'REVIEW';

// ─── Global Spatial Scales (10 Continuous LOD Scales) ─────────────────────────

export type SpatialScale =
  | 'ROOM'
  | 'BUILDING'
  | 'STREET'
  | 'PLOT'
  | 'BLOCK'
  | 'MULTI-BLOCK'
  | 'LOCALITY'
  | 'WARD'
  | 'DISTRICT'
  | 'CITY';

export interface SpatialScaleInfo {
  scale: SpatialScale;
  label: string;
  approxDistance: string;
  typicalUnits: string;
  defaultLOD: number;
}

export const SPATIAL_SCALES: SpatialScaleInfo[] = [
  { scale: 'ROOM', label: 'Room', approxDistance: '0 – 10 m', typicalUnits: 'mm / cm', defaultLOD: 4 },
  { scale: 'BUILDING', label: 'Building', approxDistance: '10 – 50 m', typicalUnits: 'cm', defaultLOD: 3 },
  { scale: 'STREET', label: 'Street', approxDistance: '50 – 200 m', typicalUnits: 'm', defaultLOD: 3 },
  { scale: 'PLOT', label: 'Plot', approxDistance: '100 – 300 m', typicalUnits: 'm', defaultLOD: 2 },
  { scale: 'BLOCK', label: 'Block', approxDistance: '200 – 1,000 m', typicalUnits: 'm', defaultLOD: 2 },
  { scale: 'MULTI-BLOCK', label: 'Multi-Block', approxDistance: '1 – 3 km', typicalUnits: 'm', defaultLOD: 2 },
  { scale: 'LOCALITY', label: 'Locality', approxDistance: '3 – 8 km', typicalUnits: 'm / km', defaultLOD: 1 },
  { scale: 'WARD', label: 'Ward', approxDistance: '5 – 15 km', typicalUnits: 'km', defaultLOD: 1 },
  { scale: 'DISTRICT', label: 'District', approxDistance: '15 – 40 km', typicalUnits: 'km', defaultLOD: 1 },
  { scale: 'CITY', label: 'City', approxDistance: '40+ km', typicalUnits: 'km', defaultLOD: 0 },
];

// ─── UI Notifications & Toast System ──────────────────────────────────────────

export type NotificationType = 'info' | 'success' | 'warning' | 'error' | 'pipeline';

export interface UINotification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: string;
  durationMs?: number;
  actionLabel?: string;
  onAction?: () => void;
  metadata?: Record<string, unknown>;
}

// ─── World Navigation Categories (Left Navigation Panel) ──────────────────────

export interface WorldItem {
  id: string;
  name: string;
  versionTag: string;
  status: 'ACTIVE' | 'ARCHIVED' | 'PROCESSING';
  coverageKm2?: number;
  lastUpdated?: string;
}

export interface CaptureSessionItem {
  id: string;
  title: string;
  sensorType: 'LASER' | 'DRONE' | 'MOBILE' | 'SATELLITE' | 'GROUND';
  date: string;
  frameCount?: number;
  status: 'ALIGNED' | 'PENDING' | 'WARNING';
}

export interface EvidenceCategoryItem {
  id: string;
  title: string;
  iconName: string;
  count: number;
  unit: string;
}

export interface PlaceItem {
  id: string;
  name: string;
  category: string;
  scale: SpatialScale;
  coordinates: [number, number];
}

export interface BookmarkItem {
  id: string;
  title: string;
  timestamp: string;
  viewState?: {
    scale: SpatialScale;
    position: Vec3;
  };
}

export interface WorldVersionItem {
  id: string;
  label: string;
  tag: string;
  isCurrent: boolean;
  date: string;
}

