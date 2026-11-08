/**
 * Data models and token helpers for Architectural Section & Elevation Drawings.
 */

export type DrawingType =
  | "section_longitudinal"
  | "section_transverse"
  | "elevation_north"
  | "elevation_south"
  | "elevation_east"
  | "elevation_west"
  | "custom_section";

export type SheetSize = "A1" | "A2" | "A3" | "ARCH_D";

export interface DrawingLine {
  layer: string;
  start: [number, number];
  end: [number, number];
  stroke_width_mm: number;
  is_dashed: boolean;
}

export interface DrawingPolygon {
  layer: string;
  vertices: [number, number][];
  fill_hatch: boolean;
}

export interface LevelMarker {
  label: string;
  elevation_m: number;
  datum_u: number;
  formatted_elevation: string;
}

export interface DimensionString {
  start: [number, number];
  end: [number, number];
  measured_length_m: number;
  label: string;
}

export interface ArchitecturalDrawing {
  drawing_id: string;
  world_id: string;
  title: string;
  drawing_type: DrawingType;
  scale_label: string;
  scale_factor: number;
  sheet_size: SheetSize;
  cut_plane_offset_m: number;
  extent_u: [number, number];
  extent_v: [number, number];
  line_count: number;
  polygon_count: number;
  level_count: number;
  dimension_count: number;
  lines: DrawingLine[];
  cut_polygons: DrawingPolygon[];
  levels: LevelMarker[];
  dimensions: DimensionString[];
  created_at_iso: string;
}

export interface DrawingPreset {
  id: string;
  title: string;
  type: DrawingType;
  axis: string;
  description: string;
}

export interface DrawingPresetsResponse {
  world_id: string;
  available_scales: string[];
  available_sheets: SheetSize[];
  presets: DrawingPreset[];
}

export interface DrawingTypeMeta {
  label: string;
  category: "section" | "elevation";
  badgeClass: string;
}

export function getDrawingTypeMeta(type: DrawingType): DrawingTypeMeta {
  switch (type) {
    case "section_longitudinal":
      return {
        label: "Section A-A (Longitudinal)",
        category: "section",
        badgeClass: "text-accent bg-accent/10 border-accent/30",
      };
    case "section_transverse":
      return {
        label: "Section B-B (Transverse)",
        category: "section",
        badgeClass: "text-accent bg-accent/10 border-accent/30",
      };
    case "elevation_north":
      return {
        label: "North Elevation",
        category: "elevation",
        badgeClass: "text-success bg-success/10 border-success/30",
      };
    case "elevation_south":
      return {
        label: "South Elevation",
        category: "elevation",
        badgeClass: "text-success bg-success/10 border-success/30",
      };
    case "elevation_east":
      return {
        label: "East Elevation",
        category: "elevation",
        badgeClass: "text-warning bg-warning/10 border-warning/30",
      };
    case "elevation_west":
      return {
        label: "West Elevation",
        category: "elevation",
        badgeClass: "text-warning bg-warning/10 border-warning/30",
      };
    case "custom_section":
    default:
      return {
        label: "Custom Section Cut",
        category: "section",
        badgeClass: "text-muted-fg bg-surface border-border",
      };
  }
}

export function formatDrawingScale(scale: string): string {
  return scale.startsWith("1:") ? scale : `1:${scale}`;
}
