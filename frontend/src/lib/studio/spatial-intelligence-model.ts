/**
 * Studio Spatial Intelligence Client Model.
 *
 * Provides data contracts and formatting utilities for:
 * - WGS84 Georeferencing and ENU local tangent plane coordinates
 * - Architectural design style & ornamentation level
 * - Whole-building multi-storey envelope metrics
 * - Physical plausibility & gravity validation reports
 * - Reconstruction completeness progress
 * - Environmental lighting & dominant irradiance vectors
 */

export interface GeodeticOrigin {
  latitude_deg: number;
  longitude_deg: number;
  altitude_m: number;
}

export interface GeoreferenceDiagnostics {
  rmse_horizontal_m: number;
  rmse_vertical_m: number;
  rmse_total_3d_m: number;
  scale_factor: number;
  inlier_count?: number;
  total_correspondences?: number;
}

export interface GeoreferenceData {
  status: "georeferenced" | "unreferenced";
  frame: "ENU" | "WORLD" | "WGS84";
  origin: GeodeticOrigin;
  diagnostics?: GeoreferenceDiagnostics;
  notes?: string[];
}

export interface DesignStyleSummary {
  primary_style: string;
  confidence: number;
  ornamentation: string;
}

export interface BuildingEnvelopeSummary {
  total_storeys: number;
  gross_floor_area_m2: number;
  building_height_m: number;
  footprint_area_m2?: number;
}

export interface PhysicalPlausibilitySummary {
  is_plausible: boolean;
  collisions_count?: number;
  floating_count?: number;
  auto_fix_proposals?: number;
}

export interface CompletenessSummary {
  overall_completeness_pct: number;
  rooms_evaluated?: number;
}

export interface LightingSummary {
  dominant_light_direction?: [number, number, number];
  ambient_irradiance_rgb?: [number, number, number];
}

export interface SharedWallsSummary {
  consolidated_wall_count: number;
}

export interface SpatialIntelligenceSummary {
  world_id: string;
  version_id?: string;
  coordinate_frame: "ENU" | "WORLD" | string;
  transforms_count: number;
  georeference?: GeoreferenceData | null;
  design_style?: DesignStyleSummary | null;
  building_envelope?: BuildingEnvelopeSummary | null;
  physical_plausibility?: PhysicalPlausibilitySummary | null;
  scene_completeness?: CompletenessSummary | null;
  lighting?: LightingSummary | null;
  shared_walls?: SharedWallsSummary | null;
}

/**
 * Formats WGS84 coordinates into clean readable string: e.g. "37.7749° N, 122.4194° W (20m alt)"
 */
export function formatCoordinates(origin: GeodeticOrigin | null | undefined): string {
  if (!origin) return "Local / Arbitrary Origin";
  const latDir = origin.latitude_deg >= 0 ? "N" : "S";
  const lonDir = origin.longitude_deg >= 0 ? "E" : "W";
  const latStr = Math.abs(origin.latitude_deg).toFixed(5);
  const lonStr = Math.abs(origin.longitude_deg).toFixed(5);
  const altStr = Math.round(origin.altitude_m);
  return `${latStr}° ${latDir}, ${lonStr}° ${lonDir} (${altStr}m alt)`;
}

/**
 * Returns human-readable design style label from enum string: "MODERN_MINIMALIST" -> "Modern Minimalist"
 */
export function formatStyleDisplay(styleStr: string | null | undefined): string {
  if (!styleStr) return "Vernacular / Contemporary";
  return styleStr
    .toLowerCase()
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Computes status pill color and text for physical plausibility.
 */
export function formatPlausibilityStatus(
  report: PhysicalPlausibilitySummary | null | undefined
): { label: string; variant: "success" | "warning" | "error" } {
  if (!report) {
    return { label: "Unchecked", variant: "warning" };
  }
  if (report.is_plausible) {
    return { label: "Physics Verified", variant: "success" };
  }
  const issues = (report.collisions_count || 0) + (report.floating_count || 0);
  return { label: `${issues} ${issues === 1 ? "Anomaly" : "Anomalies"}`, variant: "error" };
}

/**
 * Evaluates whether the world has geodetic anchoring.
 */
export function isGeoreferenced(data: SpatialIntelligenceSummary | null | undefined): boolean {
  if (!data) return false;
  return data.coordinate_frame === "ENU" && data.georeference?.status === "georeferenced";
}
