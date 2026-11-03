/**
 * Types and utilities for the 50 Open-Source Technologies Catalog and Benchmarking Harness.
 */

export type SubsystemType =
  | "visual_perception"
  | "geometric_estimation"
  | "reconstruction_mapping"
  | "pointcloud_3d_perception"
  | "geometry_cad_mesh"
  | "geospatial_environment"
  | "inference_acceleration";

export interface TechnologyItem {
  id: string;
  name: string;
  subsystem: SubsystemType;
  license: string;
  repository_url: string;
  role: string;
  primary_contribution: string;
  reality_engine_integration_target: string;
  top_priority: boolean;
  status: "available" | "reference_simulated" | "not_installed";
  install_hint?: string;
}

export interface BenchmarkMetric {
  tech_id: string;
  subsystem: SubsystemType;
  latency_ms: number;
  memory_peak_mb: number;
  vram_peak_mb: number;
  success: boolean;
  accuracy_metric_name: string;
  accuracy_metric_value: number;
  within_rtx4050_bound: boolean;
}

export const SUBSYSTEM_METADATA: Record<
  SubsystemType,
  { label: string; icon: string; description: string }
> = {
  visual_perception: {
    label: "Visual Perception",
    icon: "Eye",
    description: "2D/3D object detection, semantic & panoptic segmentation, feature matching",
  },
  geometric_estimation: {
    label: "Geometric Estimation",
    icon: "Compass",
    description: "Calibration, factor-graph optimization, robust TLS registration, Lie algebra",
  },
  reconstruction_mapping: {
    label: "Reconstruction & Mapping",
    icon: "Box",
    description: "Photogrammetry, multi-view geometry, SLAM, LiDAR odometry",
  },
  pointcloud_3d_perception: {
    label: "3D Point-Cloud Perception",
    icon: "Cloud",
    description: "Point-cloud filtering, normal estimation, sparse convolution, 3D segmentation",
  },
  geometry_cad_mesh: {
    label: "Geometry, CAD & Mesh",
    icon: "Layers",
    description: "2-manifold topological repair, CSG booleans, QEM decimation, cache optimization",
  },
  geospatial_environment: {
    label: "Geospatial & Large-Scale",
    icon: "Globe",
    description: "GIS DEM raster grids, D8 hydrological routing, point streaming, coordinate transforms",
  },
  inference_acceleration: {
    label: "Inference Acceleration",
    icon: "Zap",
    description: "INT8 dynamic quantization, hardware execution providers, VRAM enforcement",
  },
};

/**
 * Filter technologies by search query and subsystem category.
 */
export function filterTechnologies(
  items: TechnologyItem[],
  query: string,
  subsystemFilter: SubsystemType | "all" = "all"
): TechnologyItem[] {
  const q = query.trim().toLowerCase();
  return items.filter((tech) => {
    const matchesSubsystem =
      subsystemFilter === "all" || tech.subsystem === subsystemFilter;
    if (!matchesSubsystem) return false;

    if (!q) return true;
    return (
      tech.name.toLowerCase().includes(q) ||
      tech.role.toLowerCase().includes(q) ||
      tech.id.toLowerCase().includes(q) ||
      tech.primary_contribution.toLowerCase().includes(q)
    );
  });
}

/**
 * Group technologies by their subsystem category.
 */
export function groupTechnologiesBySubsystem(
  items: TechnologyItem[]
): Record<SubsystemType, TechnologyItem[]> {
  const groups: Record<SubsystemType, TechnologyItem[]> = {
    visual_perception: [],
    geometric_estimation: [],
    reconstruction_mapping: [],
    pointcloud_3d_perception: [],
    geometry_cad_mesh: [],
    geospatial_environment: [],
    inference_acceleration: [],
  };

  for (const item of items) {
    if (groups[item.subsystem]) {
      groups[item.subsystem].push(item);
    }
  }

  return groups;
}

/**
 * Check if license requires distribution/copyleft notice.
 */
export function isCopyleftLicense(license: string): boolean {
  const upper = license.toUpperCase();
  return upper.includes("GPL") || upper.includes("AGPL");
}

/**
 * Check if the benchmark results satisfy the RTX 4050 6GB (6,144 MB) constraint.
 */
export function isHardwareCompliant(metric: BenchmarkMetric, ceilingMb: number = 6144.0): boolean {
  return metric.vram_peak_mb <= ceilingMb;
}
