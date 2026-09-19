"use client";
/**
 * Reality Engine — Global Application Store (Zustand)
 * Single source of truth for cross-workspace state, engineering measurements,
 * mobile capture, loader ingestion, spatial reconstruction studio, and benchmark corpus.
 */
import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import type {
  Entity,
  EntityId,
  Project,
  Session,
  World,
  Build,
  SelectionState,
  WorkspaceId,
  ComputeMetrics,
  SimulationScenario,
  LogEntry,
  DensityMode,
  MobileSensorStatus,
  LiveCaptureQuality,
  ValidationIssue,
  Measurement,
  MeasurementType,
  BenchmarkStructure,
  BenchmarkCategory,
  ScaleLevel,
  ReconstructionViewMode,
  CapturePassType,
  DetailCoverageNode,
  DetailProvenanceRecord,
  CaptureGuidanceCue,
  GlobalMode,
  SpatialScale,
  UINotification,
} from "@/types/reality-engine";
import { BENCHMARK_STRUCTURES } from "./benchmark-corpus";

export * from "@/types/reality-engine";
export type * from "@/types/reality-engine";

import {
  checkBackendStatus,
  fetchWorldList,
  fetchWorldDetail,
  fetchWorldPointCloud,
  convertBackendEntityToEntity,
  type BackendWorldSummary,
  type BackendGeometryPayload,
} from "@/lib/api";

// ─── Mock Data ────────────────────────────────────────────────────────────────

const MOCK_PROJECT: Project = {
  id: "proj-kolkata-001",
  name: "Victoria Memorial Monument Complex",
  description: "Multi-modal precision reality capture of historical monument and grounds",
  worldId: "world-001",
  sessionIds: ["sess-001", "sess-002", "sess-003"],
  buildIds: ["build-001"],
  createdAt: "2026-09-14T08:00:00Z",
  updatedAt: "2026-09-15T14:30:00Z",
};

const MOCK_ENTITIES: Entity[] = [
  {
    id: "ent-world",
    name: "Victoria Memorial World",
    type: "WORLD",
    childIds: ["ent-site"],
    provenance: { state: "OBSERVED", confidence: 0.98 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["world", "georeferenced"],
    metadata: { crs: "EPSG:32645 (WGS 84 / UTM zone 45N)", datum: "WGS84" },
    representations: ["POINT_CLOUD", "MESH"],
    sessionIds: ["sess-001", "sess-002"],
    observationCount: 38450,
    evidenceCount: 14200,
  },
  {
    id: "ent-site",
    name: "Monument Grounds and Water Basin",
    type: "SITE",
    parentId: "ent-world",
    childIds: ["ent-terrain", "ent-structure-main", "ent-roads"],
    provenance: { state: "OBSERVED", confidence: 0.96 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["site", "heritage"],
    metadata: { area_m2: 260000, bounds_lat: [22.5448, 22.5492] },
    representations: ["POINT_CLOUD", "MESH"],
    sessionIds: ["sess-001", "sess-002"],
    observationCount: 32000,
    evidenceCount: 11500,
  },
  {
    id: "ent-terrain",
    name: "Topographic Ground Surface",
    type: "TERRAIN",
    parentId: "ent-site",
    childIds: [],
    provenance: { state: "RECONSTRUCTED", confidence: 0.92, algorithm: "OpenMVS Poisson Surface" },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["terrain"],
    metadata: { area_m2: 185000, elevation_delta_m: 3.4 },
    representations: ["MESH", "POINT_CLOUD"],
    sessionIds: ["sess-001"],
    observationCount: 9200,
    evidenceCount: 4100,
  },
  {
    id: "ent-structure-main",
    name: "Victoria Memorial Main Edifice",
    type: "BUILDING",
    parentId: "ent-site",
    childIds: ["ent-floor-plinth", "ent-dome-central", "ent-facade-north"],
    provenance: { state: "RECONSTRUCTED", confidence: 0.98, algorithm: "COLMAP + OpenMVS Fusion", uncertainty: 0.012 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["structure", "marble", "indo-saracenic"],
    metadata: { height_m: 56.0, length_m: 103.0, width_m: 69.0, material: "Makrana Marble" },
    representations: ["MESH", "POINT_CLOUD"],
    sessionIds: ["sess-001", "sess-002"],
    observationCount: 18450,
    evidenceCount: 6800,
  },
  {
    id: "ent-facade-north",
    name: "North Facade",
    type: "WALL",
    parentId: "ent-structure-main",
    childIds: ["ent-central-bay"],
    provenance: { state: "RECONSTRUCTED", confidence: 0.98, algorithm: "Dense Multi-view Stereo", uncertainty: 0.008 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["facade", "north", "exterior"],
    metadata: { height_m: 28.4, span_m: 86.2, material: "Makrana Marble" },
    representations: ["MESH", "POINT_CLOUD"],
    sessionIds: ["sess-001", "sess-002"],
    observationCount: 8400,
    evidenceCount: 310,
  },
  {
    id: "ent-central-bay",
    name: "North Central Bay & Portico",
    type: "ROOM",
    parentId: "ent-facade-north",
    childIds: ["ent-column-ionic-04"],
    provenance: { state: "RECONSTRUCTED", confidence: 0.99, algorithm: "Sub-millimeter Photogrammetry", uncertainty: 0.004 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["portico", "colonnade"],
    metadata: { height_m: 14.8, depth_m: 6.2, column_count: 6 },
    representations: ["MESH"],
    sessionIds: ["sess-002"],
    observationCount: 4200,
    evidenceCount: 180,
  },
  {
    id: "ent-column-ionic-04",
    name: "Ionic Fluted Column #4",
    type: "WALL",
    parentId: "ent-central-bay",
    childIds: ["ent-capital-ionic-04"],
    provenance: { state: "RECONSTRUCTED", confidence: 0.99, uncertainty: 0.003 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["column", "ionic", "fluted"],
    metadata: { height_m: 11.2, diameter_m: 1.15, flute_count: 24 },
    representations: ["MESH"],
    sessionIds: ["sess-002"],
    observationCount: 2400,
    evidenceCount: 120,
  },
  {
    id: "ent-capital-ionic-04",
    name: "Carved Ionic Capital & Volutes",
    type: "OBJECT",
    parentId: "ent-column-ionic-04",
    childIds: ["ent-ornament-floral"],
    provenance: { state: "RECONSTRUCTED", confidence: 0.97, algorithm: "Dense Macro-Photogrammetry", uncertainty: 0.002 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["capital", "volute", "ornament", "detail"],
    metadata: { height_m: 1.25, width_m: 1.6, style: "Ionic Order" },
    representations: ["MESH"],
    sessionIds: ["sess-002"],
    observationCount: 1850,
    evidenceCount: 94,
  },
  {
    id: "ent-ornament-floral",
    name: "Carved Leaf & Floral Relief",
    type: "OBJECT",
    parentId: "ent-capital-ionic-04",
    childIds: ["ent-micro-weathering"],
    provenance: { state: "RECONSTRUCTED", confidence: 0.96, algorithm: "Sub-pixel Disparity Fusion", uncertainty: 0.001 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["relief", "floral", "carving", "micro_detail"],
    metadata: { relief_depth_mm: 45, feature_resolution_mm: 0.35, motif: "Acanthus & Lotus" },
    representations: ["MESH"],
    sessionIds: ["sess-002"],
    observationCount: 890,
    evidenceCount: 48,
  },
  {
    id: "ent-micro-weathering",
    name: "Surface Weathering & Fine Marble Veins",
    type: "OBJECT",
    parentId: "ent-ornament-floral",
    childIds: [],
    provenance: { state: "OBSERVED", confidence: 0.94, algorithm: "Photometric Normal Mapping", uncertainty: 0.0005 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["micro_detail", "texture", "weathering", "cracks"],
    metadata: { crack_width_mm: 0.8, micro_roughness: 0.12 },
    representations: ["MESH"],
    sessionIds: ["sess-002"],
    observationCount: 620,
    evidenceCount: 32,
  },
  {
    id: "ent-dome-central",
    name: "Central Queen's Dome and Angel of Victory",
    type: "WALL",
    parentId: "ent-structure-main",
    childIds: [],
    provenance: { state: "RECONSTRUCTED", confidence: 0.97, algorithm: "Dense Multi-view Stereo", uncertainty: 0.018 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["dome", "rotunda"],
    metadata: { diameter_m: 21.4, dome_height_m: 19.2, statue_height_m: 4.9 },
    representations: ["MESH"],
    sessionIds: ["sess-001"],
    observationCount: 4800,
    evidenceCount: 2200,
  },
  {
    id: "ent-floor-plinth",
    name: "Plinth and Colonnaded Gallery",
    type: "FLOOR",
    parentId: "ent-structure-main",
    childIds: ["ent-room-royal", "ent-room-portrait"],
    provenance: { state: "RECONSTRUCTED", confidence: 0.95, uncertainty: 0.014 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["gallery", "colonnade"],
    metadata: { plinth_elevation_m: 2.1, area_m2: 7100 },
    representations: ["MESH"],
    sessionIds: ["sess-002"],
    observationCount: 6800,
    evidenceCount: 2800,
  },
  {
    id: "ent-room-royal",
    name: "Royal Gallery Chamber",
    type: "ROOM",
    parentId: "ent-floor-plinth",
    childIds: [],
    provenance: { state: "RECONSTRUCTED", confidence: 0.93, algorithm: "Indoor SLAM + LiDAR" },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["interior"],
    metadata: { ceiling_height_m: 12.8, area_m2: 540 },
    representations: ["MESH"],
    sessionIds: ["sess-002"],
    observationCount: 2100,
    evidenceCount: 940,
  },
  {
    id: "ent-room-portrait",
    name: "Portrait and Archive Hall",
    type: "ROOM",
    parentId: "ent-floor-plinth",
    childIds: [],
    provenance: { state: "RECONSTRUCTED", confidence: 0.91 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["interior"],
    metadata: { ceiling_height_m: 10.4, area_m2: 420 },
    representations: ["MESH"],
    sessionIds: ["sess-002"],
    observationCount: 1650,
    evidenceCount: 710,
  },
  {
    id: "ent-roads",
    name: "Paved Perimeter Paths and Moat Embankment",
    type: "ROAD",
    parentId: "ent-site",
    childIds: [],
    provenance: { state: "OBSERVED", confidence: 0.96 },
    visibility: "VISIBLE",
    lock: "UNLOCKED",
    tags: ["infrastructure"],
    metadata: { total_path_length_m: 3400 },
    representations: ["MESH", "POINT_CLOUD"],
    sessionIds: ["sess-001"],
    observationCount: 4200,
    evidenceCount: 1800,
  },
];

const MOCK_COVERAGE_TREE: DetailCoverageNode[] = [
  {
    id: "cov-bld",
    name: "Victoria Memorial Main Edifice",
    path: "World > Site > Main Building",
    scaleLevel: "BUILDING",
    coveragePercent: 87,
    resolutionMm: 4.2,
    evidenceObservations: 18450,
    registeredCameras: 680,
    status: "HIGH",
    children: [
      {
        id: "cov-fac-n",
        name: "North Facade",
        path: "World > Site > Main Building > North Facade",
        scaleLevel: "FACADE",
        coveragePercent: 94,
        resolutionMm: 2.1,
        evidenceObservations: 8400,
        registeredCameras: 310,
        status: "HIGH",
        children: [
          {
            id: "cov-comp-ent",
            name: "Central Entrance & Colonnade",
            path: "World > Site > Main Building > North Facade > Central Entrance",
            scaleLevel: "COMPONENT",
            coveragePercent: 99,
            resolutionMm: 1.2,
            evidenceObservations: 4200,
            registeredCameras: 180,
            status: "HIGH",
            children: [
              {
                id: "cov-det-cap",
                name: "Ionic Column Capitals & Volutes",
                path: "World > Site > Main Building > North Facade > Central Entrance > Capitals",
                scaleLevel: "DETAIL",
                coveragePercent: 96,
                resolutionMm: 0.6,
                evidenceObservations: 1850,
                registeredCameras: 94,
                status: "HIGH",
                children: [
                  {
                    id: "cov-micro-floral",
                    name: "Carved Leaf & Floral Relief",
                    path: "World > Site > Main Building > North Facade > Central Entrance > Capitals > Floral Relief",
                    scaleLevel: "MICRO_DETAIL",
                    coveragePercent: 91,
                    resolutionMm: 0.35,
                    evidenceObservations: 890,
                    registeredCameras: 48,
                    status: "HIGH",
                  },
                ],
              },
            ],
          },
          {
            id: "cov-det-nw",
            name: "Northwest Corner Ornament & Balustrade",
            path: "World > Site > Main Building > North Facade > Northwest Ornament",
            scaleLevel: "DETAIL",
            coveragePercent: 42,
            resolutionMm: 8.5,
            evidenceObservations: 340,
            registeredCameras: 16,
            status: "LOW",
          },
        ],
      },
      {
        id: "cov-fac-s",
        name: "South Facade & Terrace",
        path: "World > Site > Main Building > South Facade",
        scaleLevel: "FACADE",
        coveragePercent: 78,
        resolutionMm: 3.8,
        evidenceObservations: 5200,
        registeredCameras: 220,
        status: "MED",
      },
    ],
  },
];

const MOCK_DETAIL_PROVENANCE: DetailProvenanceRecord = {
  detailId: "det-ionic-capital-04",
  detailName: "Ionic Column Capital #4 (North Portico)",
  parentComponent: "North Central Bay & Portico",
  scaleLevel: "DETAIL",
  observationsCount: 48,
  registeredCamerasCount: 22,
  passes: ["STRUCTURE_PASS", "FACADE_PASS", "DETAIL_PASS", "MICRO_DETAIL_PASS"],
  depthMethod: "Dense Patch-Match MVS with Sub-pixel Disparity",
  fusionMethod: "Screened Poisson Surface Reconstruction with Normal Constraints",
  groundResolutionMm: 0.42,
  confidenceScore: 0.964,
  sampleCameraIds: ["CAM_0428", "CAM_0429", "CAM_0433", "CAM_0440", "CAM_0452", "CAM_0461"],
  geometryLOD: "HIGH_FREQ_RELIEF",
};

const MOCK_SESSIONS: Session[] = [
  {
    id: "sess-001",
    projectId: "proj-kolkata-001",
    name: "Drone_Aerial_Orbit_2026_09_14_001",
    status: "RECONSTRUCTED",
    sources: [{ type: "DRONE", imageCount: 4821, hasGNSS: true, hasIMU: true, calibrated: true }],
    imageCount: 4821,
    registeredCount: 4792,
    reprojectionError: 0.67,
    pointCount: 2940000,
    quality: 0.96,
    buildIds: ["build-001"],
  },
  {
    id: "sess-002",
    projectId: "proj-kolkata-001",
    name: "iPhone_Pro_Walkthrough_2026_09_14_002",
    status: "RECONSTRUCTED",
    sources: [{ type: "PHONE", imageCount: 2840, hasGNSS: true, hasIMU: true, calibrated: true }],
    imageCount: 2840,
    registeredCount: 2798,
    reprojectionError: 0.89,
    pointCount: 1420000,
    quality: 0.92,
    buildIds: ["build-001"],
  },
  {
    id: "sess-003",
    projectId: "proj-kolkata-001",
    name: "Terrestrial_LiDAR_Scan_2026_09_15_001",
    status: "PROCESSING",
    sources: [{ type: "LIDAR", lidarScans: 8, hasGNSS: true, hasIMU: true, calibrated: true }],
    imageCount: 0,
    registeredCount: 0,
    pointCount: 18500000,
    quality: 0.98,
    buildIds: [],
  },
  {
    id: "sess-004",
    projectId: "proj-kolkata-001",
    name: "Camera_Rig_North_Facade_2026_09_16_001",
    status: "QUEUED",
    sources: [{ type: "CAMERA_RIG", imageCount: 1650, hasGNSS: true, hasIMU: true, calibrated: true }],
    imageCount: 1650,
    registeredCount: 0,
    quality: undefined,
    buildIds: [],
  },
];

const MOCK_BUILD: Build = {
  id: "build-001",
  projectId: "proj-kolkata-001",
  sessionIds: ["sess-001", "sess-002"],
  status: "COMPLETE",
  stages: [
    { stage: "CAPTURE", status: "COMPLETE", progress: 100, durationMs: 0, backend: "Hardware Sensors", gpuUsage: 0, ramUsageMb: 0, warnings: [], errors: [], metrics: { images: 7661 }, inputArtifacts: [], outputArtifacts: [] },
    { stage: "TIME_SYNC", status: "COMPLETE", progress: 100, durationMs: 12400, backend: "RTE Hardware Clock PTP", gpuUsage: 0, ramUsageMb: 240, warnings: [], errors: [], metrics: { drift_ms: 1.8 }, inputArtifacts: [], outputArtifacts: [] },
    { stage: "CALIBRATION", status: "COMPLETE", progress: 100, durationMs: 42000, backend: "Brown-Conrady Intrinsics", gpuUsage: 12, ramUsageMb: 520, warnings: [], errors: [], metrics: { reprojection_px: 0.48 }, inputArtifacts: [], outputArtifacts: [] },
    { stage: "LOCALIZATION", status: "COMPLETE", progress: 100, durationMs: 145000, backend: "COLMAP GPU Matcher", gpuUsage: 74, ramUsageMb: 3800, warnings: [], errors: [], metrics: { registered: 7590, total: 7661 }, inputArtifacts: [], outputArtifacts: [] },
    { stage: "REGISTRATION", status: "COMPLETE", progress: 100, durationMs: 98000, backend: "Bundle Adjustment (Ceres)", gpuUsage: 82, ramUsageMb: 4200, warnings: [], errors: [], metrics: { mean_residual: 0.62 }, inputArtifacts: [], outputArtifacts: [] },
    { stage: "SPARSE", status: "COMPLETE", progress: 100, durationMs: 290000, backend: "COLMAP CUDA Triangulation", gpuUsage: 94, ramUsageMb: 6800, warnings: [], errors: [], metrics: { sparse_points: 684200 }, inputArtifacts: [], outputArtifacts: [] },
    { stage: "DENSE", status: "COMPLETE", progress: 100, durationMs: 1680000, backend: "OpenMVS CUDA PatchMatch", gpuUsage: 99, ramUsageMb: 11400, warnings: [], errors: [], metrics: { dense_points: 4360000 }, inputArtifacts: [], outputArtifacts: [] },
    { stage: "FUSION", status: "COMPLETE", progress: 100, durationMs: 380000, backend: "Volumetric Signed Distance Field", gpuUsage: 78, ramUsageMb: 8200, warnings: [], errors: [], metrics: { triangles: 2150000 }, inputArtifacts: [], outputArtifacts: [] },
    { stage: "PERCEPTION", status: "COMPLETE", progress: 100, durationMs: 510000, backend: "Segment Anything 2 (SAM2) 3D", gpuUsage: 92, ramUsageMb: 9400, warnings: [], errors: [], metrics: { segmented_entities: 184 }, inputArtifacts: [], outputArtifacts: [] },
    { stage: "COMPILATION", status: "COMPLETE", progress: 100, durationMs: 72000, backend: "Reality Engine WorldIR Compiler", gpuUsage: 24, ramUsageMb: 2400, warnings: [], errors: [], metrics: { worldir_nodes: 184, observations: 38450 }, inputArtifacts: [], outputArtifacts: [] },
  ],
  createdAt: "2026-09-14T09:00:00Z",
  startedAt: "2026-09-14T09:01:00Z",
  completedAt: "2026-09-14T15:30:00Z",
  logs: [],
};

const MOCK_MEASUREMENTS: Measurement[] = [
  {
    id: "meas-001",
    name: "North Portico Colonnade Width",
    type: "WIDTH",
    value: 48.32,
    unit: "m",
    uncertainty: 0.04,
    entityId: "ent-structure-main",
    confidence: 0.98,
    timestamp: "2026-09-15T10:14:22Z",
  },
  {
    id: "meas-002",
    name: "Main Edifice Structural Height",
    type: "HEIGHT",
    value: 56.14,
    unit: "m",
    uncertainty: 0.06,
    entityId: "ent-structure-main",
    confidence: 0.97,
    timestamp: "2026-09-15T10:15:08Z",
  },
  {
    id: "meas-003",
    name: "Central Dome Internal Diameter",
    type: "POINT_TO_POINT",
    value: 21.38,
    unit: "m",
    uncertainty: 0.03,
    entityId: "ent-dome-central",
    confidence: 0.96,
    timestamp: "2026-09-15T10:18:45Z",
  },
  {
    id: "meas-004",
    name: "Exterior Granite Wall Thickness",
    type: "WALL_THICKNESS",
    value: 1.42,
    unit: "m",
    uncertainty: 0.015,
    entityId: "ent-room-royal",
    confidence: 0.94,
    timestamp: "2026-09-15T10:20:10Z",
  },
  {
    id: "meas-005",
    name: "Royal Gallery Floor Area",
    type: "AREA",
    value: 540.2,
    unit: "m²",
    uncertainty: 1.4,
    entityId: "ent-room-royal",
    confidence: 0.95,
    timestamp: "2026-09-15T10:22:00Z",
  },
];

const MOCK_VALIDATION_ISSUES: ValidationIssue[] = [
  {
    id: "val-001",
    severity: "WARNING",
    title: "Timestamp drift detected between sensors",
    description: "Hardware PTP drift exceeded sync threshold between Drone aerial telemetry and mobile handheld camera stream.",
    affectedCount: 1238,
    affectedUnit: "frames",
    evidence: "Estimated drift: 42 ms across pass 2 (2026-09-14 11:24).",
    recommendedAction: "Apply cubic B-spline temporal re-interpolation or sync to GNSS PPS reference clock.",
  },
  {
    id: "val-002",
    severity: "PASS",
    title: "IMU Gravitational Bias Calibration Valid",
    description: "Tri-axial accelerometer and gyroscope zero-velocity bias converged with covariance < 1e-4.",
    affectedCount: 7661,
    affectedUnit: "samples",
    evidence: "Allan variance residuals within standard aerospace bounds.",
    recommendedAction: "No action required. Sensor calibration certified.",
  },
  {
    id: "val-003",
    severity: "ERROR",
    title: "Insufficient feature matches on East facade glazed marble",
    description: "COLMAP feature matcher could not associate tie points due to reflective specular glare from wet Makrana marble panels.",
    affectedCount: 14,
    affectedUnit: "keyframes",
    evidence: "Session 2026-09-14-002: Inlier ratio fell below 0.08 threshold.",
    recommendedAction: "Add supplementary oblique views around east facade or inject cross-polarized captures.",
  },
  {
    id: "val-004",
    severity: "WARNING",
    title: "GNSS Carrier Phase Multi-path Warning",
    description: "Satellite signal reflection occurred under the deep colonnade overhang, degrading positional covariance.",
    affectedCount: 64,
    affectedUnit: "epochs",
    evidence: "RTK fixed integer ambiguity temporarily dropped to float mode.",
    recommendedAction: "Constrain visual-inertial bundle adjustment to high-confidence LiDAR point landmarks.",
  },
];

const DEFAULT_MOBILE_SENSORS: MobileSensorStatus = {
  cameraReady: true,
  gnssFix: true,
  gnssAccuracyM: 0.016,
  gnssHz: 10,
  imuCalibrated: true,
  imuHz: 200,
  ptpTimeSynced: true,
  ptpDriftMs: 0.38,
  storageAvailableGb: 194.5,
  storageTotalGb: 256.0,
  lensCalibrated: true,
};

const DEFAULT_CAPTURE_QUALITY: LiveCaptureQuality = {
  overall: "GOOD",
  motionBlurScore: 0.03,
  lightingScore: 0.94,
  focusScore: 0.97,
  coveragePercentage: 74.2,
  recentWarnings: [],
  fps: 60.0,
  droppedFrames: 0,
  recordedSeconds: 168,
  capturedFrames: 1008,
};

// ─── Store Interface ──────────────────────────────────────────────────────────

interface RealityEngineStore {
  // ── Workstation Preferences and Density
  densityMode: DensityMode;
  setDensityMode: (mode: DensityMode) => void;

  // ── Project / World context
  project: Project | null;
  world: World | null;
  sessions: Session[];
  builds: Build[];
  entities: Map<EntityId, Entity>;
  entityTree: EntityId[];

  // ── Synchronized Selection across Outliner, Viewport, Inspector, Evidence
  selection: SelectionState;

  // ── Active Workspace
  activeWorkspace: WorkspaceId;

  // ── Viewport 3D State
  viewportMode: "PERSPECTIVE" | "ORTHOGRAPHIC" | "TOP" | "FRONT" | "SIDE";
  shadingMode: "RGB" | "NORMALS" | "WIREFRAME" | "POINT_CLOUD" | "SPLATS" | "DEPTH" | "CONFIDENCE" | "COVERAGE";
  showGrid: boolean;
  showAxes: boolean;
  showCameras: boolean;
  showPointCloud: boolean;
  showMesh: boolean;
  showSplats: boolean;
  showConfidence: boolean;
  showProvenance: boolean;

  // ── Measurements with Uncertainty
  measurements: Measurement[];
  activeMeasurementTool: MeasurementType | null;

  // ── Studio Workstation Panels (Collapsible & Progressive)
  outlinerCollapsed: boolean;
  inspectorCollapsed: boolean;
  bottomDrawerOpen: boolean;
  bottomDrawerTab: "pipeline" | "jobs" | "evidence" | "diagnostics" | "console";
  studioBottomTab: "pipeline" | "jobs" | "evidence" | "diagnostics" | "console";

  // ── Benchmark Comparison & Loading
  comparisonBenchmarkIds: string[];

  // ── Global Modes & Global Spatial Scale (Visual Source of Truth)
  globalMode: GlobalMode;
  setGlobalMode: (mode: GlobalMode) => void;
  spatialScale: SpatialScale;
  setSpatialScale: (scale: SpatialScale) => void;

  // ── UI Notifications & Toast System
  notifications: UINotification[];
  addNotification: (notification: Omit<UINotification, 'id' | 'timestamp'> & { id?: string }) => void;
  dismissNotification: (id: string) => void;
  clearNotifications: () => void;

  // ── Master World Navigation
  leftNavCollapsed: boolean;
  toggleLeftNav: () => void;
  leftNavSearch: string;
  setLeftNavSearch: (query: string) => void;
  selectedWorldId: string;
  setSelectedWorldId: (id: string) => void;
  selectedPlaceId: string | null;
  setSelectedPlaceId: (id: string | null) => void;
  selectedBookmarkId: string | null;
  setSelectedBookmarkId: (id: string | null) => void;
  selectedVersionId: string;
  setSelectedVersionId: (id: string) => void;

  // ── Extreme-Fidelity Multi-Scale & Reconstruction View Modes
  activeScaleLevel: ScaleLevel;
  setScaleLevel: (level: ScaleLevel) => void;
  reconstructionViewMode: ReconstructionViewMode;
  setReconstructionViewMode: (mode: ReconstructionViewMode) => void;
  activeCapturePass: CapturePassType;
  setActiveCapturePass: (pass: CapturePassType) => void;
  loaderDeviceMode: "computer" | "phone";
  setLoaderDeviceMode: (mode: "computer" | "phone") => void;
  spatialCoverageNodes: DetailCoverageNode[];
  selectedDetailProvenance: DetailProvenanceRecord | null;
  setSelectedDetailProvenance: (record: DetailProvenanceRecord | null) => void;

  // ── Mobile Capture State
  captureScreen: "home" | "project" | "prep" | "camera" | "review";
  mobileSensors: MobileSensorStatus;
  liveCaptureQuality: LiveCaptureQuality;
  cameraLens: "0.5x" | "1x" | "3x";
  isRecording: boolean;
  simulateMobileDevice: boolean;
  showCoverageHud: boolean;
  showQualityDetails: boolean;

  // ── Loader / Ingest State
  validationIssues: ValidationIssue[];
  selectedSourceTypeFilter: string | null;
  selectedSessionId: string | null;
  timelineScrubSec: number;

  // ── Benchmark Explorer
  benchmarks: BenchmarkStructure[];
  selectedBenchmarkId: string | null;
  benchmarkCategoryFilter: BenchmarkCategory | "ALL";

  // ── Compute and Diagnostics
  computeMetrics: ComputeMetrics;

  // ── Simulation (Downstream Consumer)
  activeScenario: SimulationScenario | null;
  simulationStatus: string;

  // ── Logs
  logs: LogEntry[];

  // ── Command Palette
  commandPaletteOpen: boolean;

  // ── Actions
  setProject: (project: Project) => void;
  setWorld: (world: World) => void;
  setActiveWorkspace: (id: WorkspaceId) => void;
  selectEntity: (id: EntityId, multi?: boolean) => void;
  deselectAll: () => void;
  hoverEntity: (id: EntityId | undefined) => void;
  toggleEntityVisibility: (id: EntityId) => void;
  toggleEntityLock: (id: EntityId) => void;
  isolateEntity: (id: EntityId) => void;
  showAllEntities: () => void;
  addSession: (session: Session) => void;
  setViewportMode: (mode: "PERSPECTIVE" | "ORTHOGRAPHIC" | "TOP" | "FRONT" | "SIDE") => void;
  setShadingMode: (mode: "RGB" | "NORMALS" | "WIREFRAME" | "POINT_CLOUD" | "SPLATS" | "DEPTH" | "CONFIDENCE" | "COVERAGE") => void;
  toggleViewportOption: (option: "showGrid" | "showAxes" | "showCameras" | "showPointCloud" | "showMesh" | "showSplats" | "showConfidence" | "showProvenance") => void;
  setCommandPaletteOpen: (open: boolean) => void;
  setActiveMeasurementTool: (tool: MeasurementType | null) => void;
  setStudioBottomTab: (tab: "pipeline" | "jobs" | "evidence" | "diagnostics" | "console") => void;
  // ── Studio Panel Visibility
  toggleOutliner: () => void;
  setOutlinerCollapsed: (collapsed: boolean) => void;
  toggleInspector: () => void;
  setInspectorCollapsed: (collapsed: boolean) => void;
  toggleBottomDrawer: (tab?: "pipeline" | "jobs" | "evidence" | "diagnostics" | "console") => void;
  setBottomDrawerTab: (tab: "pipeline" | "jobs" | "evidence" | "diagnostics" | "console") => void;

  // ── Benchmark Comparison & Loading
  toggleComparisonBenchmark: (id: string) => void;
  clearComparisonBenchmarks: () => void;
  loadBenchmarkIntoStudio: (benchmarkId: string) => void;

  setCaptureScreen: (screen: "home" | "project" | "prep" | "camera" | "review") => void;
  setCameraLens: (lens: "0.5x" | "1x" | "3x") => void;
  toggleRecording: () => void;
  triggerPhotoCapture: () => void;
  toggleSimulateMobileDevice: () => void;
  toggleCoverageHud: () => void;
  toggleQualityDetails: () => void;
  setSelectedSourceTypeFilter: (filter: string | null) => void;
  setSelectedSessionId: (id: string | null) => void;
  setTimelineScrubSec: (sec: number) => void;
  setSelectedBenchmarkId: (id: string | null) => void;
  setBenchmarkCategoryFilter: (cat: BenchmarkCategory | "ALL") => void;
  loadMockData: () => void;

  // ── Measurements actions
  addMeasurement: (measurement: Measurement) => void;
  clearMeasurements: () => void;

  // ── Real Backend State & Actions (NO FAKE COMPLETION)
  backendConnected: boolean;
  backendError: string | null;
  loadedFromBackend: boolean;
  activeWorldVersion: string | null;
  worldVersions: BackendWorldSummary[];
  loadWorldFromBackend: (versionId?: string) => Promise<boolean>;
  refreshWorldVersions: () => Promise<void>;

  // ── Point Cloud / Mesh Artifacts (Real Streaming)
  pointCloudPositions: Float32Array | null;
  pointCloudColors: Float32Array | null;
  pointCloudCount: number;
  pointCloudStatus: "IDLE" | "LOADING" | "AVAILABLE" | "UNAVAILABLE";
  pointCloudError: string | null;
  geometries: Record<string, BackendGeometryPayload>;
}

export const useREStore = create<RealityEngineStore>()(
  subscribeWithSelector((set, get) => ({
    // ── Workstation Preferences
    densityMode: "compact",
    setDensityMode: (densityMode) => {
      if (typeof document !== "undefined") {
        document.documentElement.setAttribute("data-density", densityMode);
      }
      set({ densityMode });
    },

    // ── Initial State
    project: null,
    world: null,
    sessions: [],
    builds: [],
    entities: new Map(),
    entityTree: [],
    selection: { selectedEntityIds: [] },
    activeWorkspace: "studio",

    // ── Viewport
    viewportMode: "PERSPECTIVE",
    shadingMode: "RGB",
    showGrid: true,
    showAxes: true,
    showCameras: true,
    showPointCloud: true,
    showMesh: true,
    showSplats: false,
    showConfidence: false,
    showProvenance: false,

    // ── Measurements
    measurements: MOCK_MEASUREMENTS,
    activeMeasurementTool: null,

    // ── Studio Workstation Panels (Collapsible & Progressive)
    outlinerCollapsed: false,
    inspectorCollapsed: false,
    bottomDrawerOpen: false, // Default: COLLAPSED! Viewport is hero
    bottomDrawerTab: "pipeline",
    studioBottomTab: "pipeline",

    // ── Benchmark Comparison
    comparisonBenchmarkIds: ["skysc-001", "fort-001", "wonder-007"],

    // ── Global Modes & Global Spatial Scale (Visual Source of Truth)
    globalMode: "INSPECT",
    setGlobalMode: (globalMode) => {
      set({ globalMode });
      // Map global mode to workspace if appropriate
      if (globalMode === "CAPTURE") set({ activeWorkspace: "capture" });
      else if (globalMode === "BUILD") set({ activeWorkspace: "build" });
      else if (globalMode === "REVIEW") set({ activeWorkspace: "evidence" });
      else set({ activeWorkspace: "studio" });
    },
    spatialScale: "BLOCK",
    setSpatialScale: (spatialScale) => {
      // Map spatial scale to legacy scaleLevel for backward compatibility
      const scaleMap: Record<SpatialScale, ScaleLevel> = {
        ROOM: "MICRO_DETAIL",
        BUILDING: "BUILDING",
        STREET: "STRUCTURE",
        PLOT: "FACADE",
        BLOCK: "SITE",
        "MULTI-BLOCK": "SITE",
        LOCALITY: "WORLD",
        WARD: "WORLD",
        DISTRICT: "WORLD",
        CITY: "WORLD",
      };
      set({ spatialScale, activeScaleLevel: scaleMap[spatialScale] ?? "STRUCTURE" });
    },

    // ── UI Notifications & Toast System
    notifications: [
      {
        id: "notif-init",
        type: "info",
        title: "Workstation Initialized",
        message: "Spatial Reality Engine v2.3.1 ready. CUDA daemon active.",
        timestamp: new Date().toISOString(),
        durationMs: 4000,
      },
    ],
    addNotification: (notification) =>
      set((s) => ({
        notifications: [
          ...s.notifications,
          {
            id: notification.id ?? `notif-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
            timestamp: new Date().toISOString(),
            durationMs: notification.durationMs ?? 4500,
            ...notification,
          },
        ],
      })),
    dismissNotification: (id) =>
      set((s) => ({ notifications: s.notifications.filter((n) => n.id !== id) })),
    clearNotifications: () => set({ notifications: [] }),

    // ── Master World Navigation
    leftNavCollapsed: false,
    toggleLeftNav: () => set((s) => ({ leftNavCollapsed: !s.leftNavCollapsed })),
    leftNavSearch: "",
    setLeftNavSearch: (leftNavSearch) => set({ leftNavSearch }),
    selectedWorldId: "world-middletown",
    setSelectedWorldId: (selectedWorldId) => set({ selectedWorldId }),
    selectedPlaceId: null,
    setSelectedPlaceId: (selectedPlaceId) => set({ selectedPlaceId }),
    selectedBookmarkId: null,
    setSelectedBookmarkId: (selectedBookmarkId) => set({ selectedBookmarkId }),
    selectedVersionId: "v7.3",
    setSelectedVersionId: (selectedVersionId) => set({ selectedVersionId }),

    // ── Extreme-Fidelity Multi-Scale & 8 Reconstruction Modes
    activeScaleLevel: "STRUCTURE",
    setScaleLevel: (activeScaleLevel) => set({ activeScaleLevel }),
    reconstructionViewMode: "GEOMETRY",
    setReconstructionViewMode: (reconstructionViewMode) => set({ reconstructionViewMode }),
    activeCapturePass: "FACADE_PASS",
    setActiveCapturePass: (activeCapturePass) => set({ activeCapturePass }),
    loaderDeviceMode: "computer",
    setLoaderDeviceMode: (loaderDeviceMode) => set({ loaderDeviceMode }),
    spatialCoverageNodes: MOCK_COVERAGE_TREE,
    selectedDetailProvenance: MOCK_DETAIL_PROVENANCE,
    setSelectedDetailProvenance: (selectedDetailProvenance) => set({ selectedDetailProvenance }),

    // ── Mobile Capture State
    captureScreen: "camera",
    mobileSensors: DEFAULT_MOBILE_SENSORS,
    liveCaptureQuality: DEFAULT_CAPTURE_QUALITY,
    cameraLens: "1x",
    isRecording: false,
    simulateMobileDevice: true,
    showCoverageHud: true,
    showQualityDetails: false,

    // ── Loader State
    validationIssues: MOCK_VALIDATION_ISSUES,
    selectedSourceTypeFilter: null,
    selectedSessionId: "sess-001",
    timelineScrubSec: 42,

    // ── Benchmark Explorer
    benchmarks: BENCHMARK_STRUCTURES,
    selectedBenchmarkId: "skysc-001",
    benchmarkCategoryFilter: "ALL",

    // ── Compute
    computeMetrics: {
      gpuUsage: 78,
      gpuVramUsed: 6300,
      gpuVramTotal: 8192,
      cpuUsage: 42,
      ramUsed: 14200,
      ramTotal: 32768,
      diskRead: 0,
      diskWrite: 0,
      networkIn: 0,
      networkOut: 0,
    },
    activeScenario: null,
    simulationStatus: "IDLE",
    logs: [
      { id: "log-1", level: "INFO", timestamp: "10:14:02", module: "COLMAP", message: "Bundle adjustment converged with 0.62 px mean reprojection error." },
      { id: "log-2", level: "INFO", timestamp: "10:14:18", module: "OpenMVS", message: "Dense point cloud fused: 4,360,000 spatial samples." },
      { id: "log-3", level: "WARNING", timestamp: "10:15:00", module: "TimeSyncer", message: "PTP timestamp drift of 42ms detected in session pass 2." },
      { id: "log-4", level: "INFO", timestamp: "10:16:22", module: "WorldIR", message: "Successfully compiled persistent hierarchical world graph (184 nodes)." },
    ],
    commandPaletteOpen: false,

    // ── Real Backend State (NO FAKE COMPLETION)
    backendConnected: false,
    backendError: null,
    loadedFromBackend: false,
    activeWorldVersion: null,
    worldVersions: [],
    pointCloudPositions: null,
    pointCloudColors: null,
    pointCloudCount: 0,
    pointCloudStatus: "IDLE",
    pointCloudError: null,
    geometries: {},

    // ── Actions
    setProject: (project) => set({ project }),
    setWorld: (world) => set({ world }),
    setActiveWorkspace: (activeWorkspace) => set({ activeWorkspace }),

    selectEntity: (id, multi = false) => set((state) => {
      const current = state.selection.selectedEntityIds;
      if (multi) {
        const next = current.includes(id) ? current.filter((x) => x !== id) : [...current, id];
        return { selection: { ...state.selection, selectedEntityIds: next } };
      }
      return { selection: { ...state.selection, selectedEntityIds: [id], focusedEntityId: id } };
    }),

    deselectAll: () => set((state) => ({
      selection: { ...state.selection, selectedEntityIds: [], focusedEntityId: undefined }
    })),

    hoverEntity: (id) => set((state) => ({
      selection: { ...state.selection, hoveredEntityId: id }
    })),

    toggleEntityVisibility: (id) => set((state) => {
      const entities = new Map(state.entities);
      const entity = entities.get(id);
      if (!entity) return {};
      entities.set(id, {
        ...entity,
        visibility: entity.visibility === "VISIBLE" ? "HIDDEN" : "VISIBLE",
      });
      return { entities };
    }),

    toggleEntityLock: (id) => set((state) => {
      const entities = new Map(state.entities);
      const entity = entities.get(id);
      if (!entity) return {};
      entities.set(id, {
        ...entity,
        lock: entity.lock === "LOCKED" ? "UNLOCKED" : "LOCKED",
      });
      return { entities };
    }),

    isolateEntity: (id) => set((state) => {
      const entities = new Map(state.entities);
      // Collect target and all its descendant IDs
      const toKeep = new Set<string>();
      const addDescendants = (currentId: string) => {
        toKeep.add(currentId);
        const node = entities.get(currentId);
        if (node && node.childIds) {
          node.childIds.forEach(addDescendants);
        }
      };
      addDescendants(id);

      entities.forEach((ent, k) => {
        entities.set(k, {
          ...ent,
          visibility: toKeep.has(k) ? "VISIBLE" : "HIDDEN",
        });
      });
      return { entities, selection: { ...state.selection, selectedEntityIds: [id], focusedEntityId: id } };
    }),

    showAllEntities: () => set((state) => {
      const entities = new Map(state.entities);
      entities.forEach((ent, k) => {
        entities.set(k, { ...ent, visibility: "VISIBLE" });
      });
      return { entities };
    }),

    addSession: (session) => set((state) => ({
      sessions: [session, ...state.sessions],
      selectedSessionId: session.id,
    })),

    setViewportMode: (viewportMode) => set({ viewportMode }),
    setShadingMode: (shadingMode) => set({ shadingMode }),

    toggleViewportOption: (option) => set((state) => ({
      [option]: !state[option],
    })),

    setCommandPaletteOpen: (commandPaletteOpen) => set({ commandPaletteOpen }),
    setActiveMeasurementTool: (activeMeasurementTool) => set({ activeMeasurementTool }),
    addMeasurement: (measurement) =>
      set((s) => ({ measurements: [measurement, ...s.measurements] })),
    clearMeasurements: () => set({ measurements: [] }),
    setStudioBottomTab: (studioBottomTab) => set({ studioBottomTab }),

    toggleOutliner: () => set((s) => ({ outlinerCollapsed: !s.outlinerCollapsed })),
    setOutlinerCollapsed: (outlinerCollapsed) => set({ outlinerCollapsed }),
    toggleInspector: () => set((s) => ({ inspectorCollapsed: !s.inspectorCollapsed })),
    setInspectorCollapsed: (inspectorCollapsed) => set({ inspectorCollapsed }),
    toggleBottomDrawer: (tab) => set((s) => {
      if (tab && s.bottomDrawerTab !== tab) {
        return { bottomDrawerOpen: true, bottomDrawerTab: tab, studioBottomTab: tab };
      }
      return { bottomDrawerOpen: !s.bottomDrawerOpen, ...(tab ? { bottomDrawerTab: tab, studioBottomTab: tab } : {}) };
    }),
    setBottomDrawerTab: (tab) => set({ bottomDrawerTab: tab, studioBottomTab: tab, bottomDrawerOpen: true }),

    toggleComparisonBenchmark: (id) => set((s) => {
      const exists = s.comparisonBenchmarkIds.includes(id);
      if (exists) {
        return { comparisonBenchmarkIds: s.comparisonBenchmarkIds.filter((x) => x !== id) };
      }
      if (s.comparisonBenchmarkIds.length >= 3) {
        return { comparisonBenchmarkIds: [...s.comparisonBenchmarkIds.slice(1), id] };
      }
      return { comparisonBenchmarkIds: [...s.comparisonBenchmarkIds, id] };
    }),
    clearComparisonBenchmarks: () => set({ comparisonBenchmarkIds: [] }),
    loadBenchmarkIntoStudio: (benchmarkId) => {
      const target = get().benchmarks.find((b) => b.id === benchmarkId);
      if (!target) return;
      set({
        selectedBenchmarkId: benchmarkId,
        activeWorkspace: "studio",
        world: {
          id: `world-${target.id}`,
          projectId: `proj-${target.id}`,
          name: `${target.name} Complex`,
          status: target.reconstructionStatus === "VALIDATED" || target.reconstructionStatus === "WORLDIR_COMPILED" ? "READY" : "PROCESSING",
          entityCount: target.reconstructionStatus === "VALIDATED" ? 24 : 6,
          sessionIds: ["sess-001"],
          versions: ["v1"],
          currentVersion: "v1",
          createdAt: "2026-09-15T08:00:00Z",
          updatedAt: "2026-09-16T10:00:00Z",
        },
      });
    },

    setCaptureScreen: (captureScreen) => set({ captureScreen }),
    setCameraLens: (cameraLens) => set({ cameraLens }),
    toggleRecording: () => set((s) => ({ isRecording: !s.isRecording })),
    triggerPhotoCapture: () => set((s) => ({
      liveCaptureQuality: {
        ...s.liveCaptureQuality,
        capturedFrames: s.liveCaptureQuality.capturedFrames + 1,
      },
    })),
    toggleSimulateMobileDevice: () => set((s) => ({ simulateMobileDevice: !s.simulateMobileDevice })),
    toggleCoverageHud: () => set((s) => ({ showCoverageHud: !s.showCoverageHud })),
    toggleQualityDetails: () => set((s) => ({ showQualityDetails: !s.showQualityDetails })),

    setSelectedSourceTypeFilter: (selectedSourceTypeFilter) => set({ selectedSourceTypeFilter }),
    setSelectedSessionId: (selectedSessionId) => set({ selectedSessionId }),
    setTimelineScrubSec: (timelineScrubSec) => set({ timelineScrubSec }),

    setSelectedBenchmarkId: (selectedBenchmarkId) => set({ selectedBenchmarkId }),
    setBenchmarkCategoryFilter: (benchmarkCategoryFilter) => set({ benchmarkCategoryFilter }),

    loadMockData: () => {
      const entityMap = new Map<EntityId, Entity>();
      MOCK_ENTITIES.forEach((e) => entityMap.set(e.id, e));
      const roots = MOCK_ENTITIES.filter((e) => !e.parentId).map((e) => e.id);
      set({
        project: MOCK_PROJECT,
        sessions: MOCK_SESSIONS,
        builds: [MOCK_BUILD],
        entities: entityMap,
        entityTree: roots,
        measurements: MOCK_MEASUREMENTS,
        validationIssues: MOCK_VALIDATION_ISSUES,
        world: {
          id: "world-001",
          projectId: "proj-kolkata-001",
          name: "Victoria Memorial Monument Complex [DEMO]",
          status: "READY",
          entityCount: MOCK_ENTITIES.length,
          sessionIds: ["sess-001", "sess-002"],
          versions: ["v1", "v2", "v3"],
          currentVersion: "v3",
          createdAt: "2026-09-14T09:00:00Z",
          updatedAt: "2026-09-15T14:30:00Z",
        },
      });
    },

    refreshWorldVersions: async () => {
      const worlds = await fetchWorldList();
      set({ worldVersions: worlds });
    },

    loadWorldFromBackend: async (versionId?: string) => {
      const status = await checkBackendStatus();
      if (!status.backend) {
        set({
          backendConnected: false,
          backendError: status.error ?? "Backend offline",
          loadedFromBackend: false,
          pointCloudPositions: null,
          pointCloudColors: null,
          pointCloudCount: 0,
          pointCloudStatus: "UNAVAILABLE",
          pointCloudError: status.error ?? "Backend offline",
        });
        get().loadMockData();
        return false;
      }

      set({
        backendConnected: true,
        backendError: null,
        pointCloudStatus: "LOADING",
        pointCloudError: null,
      });
      const worlds = await fetchWorldList();
      set({ worldVersions: worlds });

      const targetVid = versionId ?? status.latest_version ?? worlds[0]?.version_id;
      if (!targetVid) {
        set({
          loadedFromBackend: false,
          pointCloudStatus: "UNAVAILABLE",
          pointCloudError: "No version available",
        });
        get().loadMockData();
        return false;
      }

      const detail = await fetchWorldDetail(targetVid);
      if (!detail || detail.error || !detail.entities) {
        set({
          backendError: detail?.error ?? "Failed to load WorldIR version",
          loadedFromBackend: false,
          pointCloudStatus: "UNAVAILABLE",
          pointCloudError: detail?.error ?? "Failed to load WorldIR version",
        });
        get().loadMockData();
        return false;
      }

      // Real point cloud streaming (NO FAKE COMPLETION)
      const pcResult = await fetchWorldPointCloud(targetVid);

      const entityMap = new Map<EntityId, Entity>();
      const converted = detail.entities.map((e) =>
        convertBackendEntityToEntity(e, detail.geometries)
      );
      converted.forEach((e) => entityMap.set(e.id, e));
      const roots = converted.filter((e) => !e.parentId).map((e) => e.id);

      set({
        entities: entityMap,
        entityTree: roots.length > 0 ? roots : converted.map((e) => e.id),
        activeWorldVersion: targetVid,
        loadedFromBackend: true,
        selectedWorldId: detail.world_id,
        geometries: detail.geometries ?? {},
        pointCloudPositions: pcResult.available && pcResult.data ? pcResult.data.positions : null,
        pointCloudColors: pcResult.available && pcResult.data ? (pcResult.data.colors ?? null) : null,
        pointCloudCount: pcResult.available && pcResult.data ? pcResult.data.pointCount : 0,
        pointCloudStatus: pcResult.available ? "AVAILABLE" : "UNAVAILABLE",
        pointCloudError: pcResult.available ? null : (pcResult.error ?? "No point cloud artifact found"),
        world: {
          id: detail.world_id,
          projectId: "proj-backend-active",
          name: detail.name || `WorldIR (${targetVid})`,
          status: "READY",
          entityCount: detail.entity_count,
          sessionIds: ["sess-001"],
          versions: worlds.map((w) => w.version_id),
          currentVersion: targetVid,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      });

      get().addNotification({
        type: "success",
        title: "Real WorldIR Loaded",
        message: `Mounted ${detail.entity_count} entities from version ${targetVid}. Point cloud: ${
          pcResult.available && pcResult.data
            ? `${pcResult.data.pointCount.toLocaleString()} points`
            : "Unavailable"
        }.`,
      });
      return true;
    },
  }))
);
