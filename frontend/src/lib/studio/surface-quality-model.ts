/**
 * Data models and studio token helpers for Surface Quality Analysis:
 * ASTM E1155 Floor Flatness (FF/FL) & Structural Deflection.
 */

export type ASTMClassification =
  | "super_flat"
  | "very_flat"
  | "flat"
  | "moderately_flat"
  | "conventional"
  | "non_compliant";

export type DeflectionCode = "L/480" | "L/360" | "L/240" | "excessive";

export interface SurfaceRunData {
  run_id: string;
  direction: "X" | "Z";
  start_pos: [number, number, number];
  end_pos: [number, number, number];
  length_m: number;
  sample_count: number;
  line_ff: number;
  line_fl: number;
  mean_curvature_mm: number;
  std_curvature_mm: number;
}

export interface FloorFlatnessReport {
  entity_id: string;
  surface_name: string;
  sample_count: number;
  spacing_m: number;
  overall_ff: number;
  overall_fl: number;
  minimum_local_ff: number;
  minimum_local_fl: number;
  target_ff: number;
  target_fl: number;
  classification: ASTMClassification;
  compliance_passed: boolean;
  mean_elevation_m: number;
  elevation_std_dev_mm: number;
  max_high_spot_mm: number;
  max_low_spot_mm: number;
  area_sqm: number;
  runs: SurfaceRunData[];
}

export interface DeflectionSample {
  distance_m: number;
  deflection_mm: number;
  baseline_m: number;
  measured_m: number;
}

export interface BeamDeflectionReport {
  entity_id: string;
  beam_name: string;
  span_length_m: number;
  start_pos: [number, number, number];
  end_pos: [number, number, number];
  max_deflection_mm: number;
  midspan_deflection_mm: number;
  deflection_ratio: number;
  deflection_ratio_str: string;
  target_code: DeflectionCode;
  achieved_code: DeflectionCode;
  is_compliant: boolean;
  deflection_profile: DeflectionSample[];
}

export interface HeatmapPoint {
  x: number;
  y: number;
  z: number;
  deviation_mm: number;
  color_rgb: [number, number, number];
  color_hex: string;
}

export interface HeatmapData {
  entity_id: string;
  surface_name: string;
  reference_elevation_m: number;
  bounds: {
    min_x: number;
    max_x: number;
    min_z: number;
    max_z: number;
  };
  point_count: number;
  points: HeatmapPoint[];
  min_deviation_mm: number;
  max_deviation_mm: number;
  mean_deviation_mm: number;
  std_deviation_mm: number;
  rms_deviation_mm: number;
  palette: string;
}

export interface SurfaceQualityReport {
  world_id: string;
  timestamp: string;
  floors: FloorFlatnessReport[];
  beams: BeamDeflectionReport[];
  overall_compliance: boolean;
  total_surfaces_analyzed: number;
}

export interface ClassificationMeta {
  label: string;
  description: string;
  badgeClass: string;
}

export function getASTMClassificationMeta(classification: ASTMClassification): ClassificationMeta {
  switch (classification) {
    case "super_flat":
      return {
        label: "Super Flat",
        description: "High-bay automated warehouses & studios (FF ≥ 60, FL ≥ 40)",
        badgeClass: "text-success bg-success/10 border-success/30",
      };
    case "very_flat":
      return {
        label: "Very Flat",
        description: "Narrow-aisle high-reach forklift tracks (FF ≥ 50, FL ≥ 30)",
        badgeClass: "text-accent bg-accent/10 border-accent/30",
      };
    case "flat":
      return {
        label: "Flat",
        description: "Gymnasiums, industrial, polished concrete (FF ≥ 35, FL ≥ 25)",
        badgeClass: "text-accent bg-accent/10 border-accent/30",
      };
    case "moderately_flat":
      return {
        label: "Moderately Flat",
        description: "Commercial retail, offices & thin-set tile (FF ≥ 25, FL ≥ 20)",
        badgeClass: "text-warning bg-warning/10 border-warning/30",
      };
    case "conventional":
      return {
        label: "Conventional",
        description: "Carpeted areas and general storage (FF ≥ 20, FL ≥ 15)",
        badgeClass: "text-muted-fg bg-surface border-border",
      };
    case "non_compliant":
    default:
      return {
        label: "Non-Compliant",
        description: "Below minimum standard tolerances (< FF 20 / FL 15)",
        badgeClass: "text-danger bg-danger/10 border-danger/30",
      };
  }
}

export interface DeflectionCodeMeta {
  label: string;
  description: string;
  badgeClass: string;
}

export function getDeflectionCodeMeta(code: DeflectionCode): DeflectionCodeMeta {
  switch (code) {
    case "L/480":
      return {
        label: "Strict (L/480)",
        description: "Brittle plaster and delicate finishes",
        badgeClass: "text-success bg-success/10 border-success/30",
      };
    case "L/360":
      return {
        label: "Standard (L/360)",
        description: "Standard architectural floor supports",
        badgeClass: "text-accent bg-accent/10 border-accent/30",
      };
    case "L/240":
      return {
        label: "Permissible (L/240)",
        description: "Roofs & industrial mezzanines",
        badgeClass: "text-warning bg-warning/10 border-warning/30",
      };
    case "excessive":
    default:
      return {
        label: "Excessive Sag (< L/240)",
        description: "Exceeds structural sagging threshold",
        badgeClass: "text-danger bg-danger/10 border-danger/30",
      };
  }
}

export function formatFFNumber(val: number): string {
  return val.toFixed(1);
}

export function formatDeviationMm(val: number): string {
  const sign = val > 0 ? "+" : "";
  return `${sign}${val.toFixed(1)} mm`;
}
