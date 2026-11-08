/**
 * Data models and studio token helpers for Automated MEP & Conduit Routing Perception
 * with IFC4x3 Facility Topology.
 */

export type MepSystemType =
  | "electrical"
  | "hvac"
  | "plumbing"
  | "fire_protection"
  | "data_comm";

export type MepElementType =
  | "conduit"
  | "cable_tray"
  | "pipe"
  | "duct"
  | "elbow"
  | "tee"
  | "cross"
  | "junction_box"
  | "panelboard"
  | "outlet"
  | "switch"
  | "light_fixture"
  | "air_terminal"
  | "plumbing_fixture";

export type CrossSectionKind = "circular" | "rectangular";
export type FlowDirection = "source" | "sink" | "bidirectional";

export interface MepPort {
  port_id: string;
  element_id: string;
  position: [number, number, number];
  direction: [number, number, number];
  flow_direction: FlowDirection;
  connected_port_id?: string | null;
  nominal_size_mm: number;
}

export interface MepSegment {
  segment_id: string;
  name: string;
  system_type: MepSystemType;
  element_type: MepElementType;
  start_point: [number, number, number];
  end_point: [number, number, number];
  waypoints: [number, number, number][];
  cross_section: CrossSectionKind;
  diameter_mm: number;
  width_mm: number;
  height_mm: number;
  length_m: number;
  ports: MepPort[];
  source_node_id?: string | null;
  target_node_id?: string | null;
  storey_id?: string | null;
  room_id?: string | null;
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
  provenance: string;
}

export interface MepFitting {
  fitting_id: string;
  name: string;
  system_type: MepSystemType;
  element_type: MepElementType;
  position: [number, number, number];
  angle_deg: number;
  connected_segment_ids: string[];
  ports: MepPort[];
}

export interface MepTerminal {
  terminal_id: string;
  name: string;
  system_type: MepSystemType;
  element_type: MepElementType;
  position: [number, number, number];
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
  host_wall_id?: string | null;
  storey_id?: string | null;
  room_id?: string | null;
  ports: MepPort[];
}

export interface MepClearanceViolation {
  violation_id: string;
  element_a_id: string;
  element_a_system: MepSystemType;
  element_b_id: string;
  element_b_system: MepSystemType;
  actual_clearance_m: number;
  required_clearance_m: number;
  location: [number, number, number];
  severity: "CRITICAL" | "MAJOR" | "MINOR";
  description: string;
}

export interface MepDistributionNetwork {
  system_id: string;
  name: string;
  system_type: MepSystemType;
  root_panel_id?: string | null;
  segment_count: number;
  fitting_count: number;
  terminal_count: number;
  total_length_m: number;
  total_volume_m3: number;
  is_connected_dag: boolean;
  segments: MepSegment[];
  fittings: MepFitting[];
  terminals: MepTerminal[];
}

export interface MepFacilityTopology {
  world_id: string;
  total_conduit_length_m: number;
  total_pipe_length_m: number;
  total_duct_length_m: number;
  summary: {
    system_count: number;
    total_elements: number;
    total_run_length_m: number;
    clearance_violations_count: number;
    code_compliant: boolean;
  };
  clearance_violations: MepClearanceViolation[];
  systems: MepDistributionNetwork[];
}

/**
 * Hex color tokens for Three.js rendering and custom SVG canvas.
 */
export function getMepSystemHex(systemType: MepSystemType): string {
  switch (systemType) {
    case "electrical":
      return "#f59e0b"; // Amber/Gold
    case "plumbing":
      return "#06b6d4"; // Cyan
    case "hvac":
      return "#10b981"; // Emerald
    case "fire_protection":
      return "#f43f5e"; // Rose
    case "data_comm":
      return "#8b5cf6"; // Purple
    default:
      return "#94a3b8";
  }
}

/**
 * Format linear length in metric meters.
 */
export function formatMepLength(meters: number): string {
  if (meters >= 1000) {
    return `${(meters / 1000).toFixed(2)} km`;
  }
  return `${meters.toFixed(2)} m`;
}

/**
 * Format cross-sectional dimension string.
 */
export function formatMepSize(segment: MepSegment): string {
  if (segment.cross_section === "circular") {
    return `Ø ${Math.round(segment.diameter_mm)} mm`;
  }
  return `${Math.round(segment.width_mm)}×${Math.round(segment.height_mm)} mm`;
}
