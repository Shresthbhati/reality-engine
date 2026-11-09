/**
 * Data models and token helpers for Life Safety & ADA Accessibility Compliance.
 * Conforms to ADA 2010 Standards and IBC Chapter 10 Means of Egress.
 */

export type ComplianceStatus = "pass" | "fail" | "warning" | "review_required";

export type CodeCheckCategory =
  | "door_clearance"
  | "corridor_width"
  | "turning_space"
  | "egress_distance"
  | "door_swing"
  | "occupant_load";

export interface ComplianceCheckItem {
  item_id: string;
  category: CodeCheckCategory;
  standard_ref: string;
  title: string;
  status: ComplianceStatus;
  measured_value: number;
  required_value: number;
  unit: string;
  message: string;
  remediation_notes: string;
  room_id: string | null;
  room_name: string | null;
  entity_id: string | null;
  coordinates_xyz: [number, number, number] | null;
}

export interface EgressRoute {
  route_id: string;
  origin_room_id: string;
  origin_room_name: string;
  destination_door_id: string;
  total_distance_m: number;
  max_permitted_distance_m: number;
  status: ComplianceStatus;
  polyline: [number, number][];
  bottlenecks: string[];
}

export interface TurningSpaceCheck {
  room_id: string;
  room_name: string;
  has_clearance: boolean;
  max_inscribed_diameter_m: number;
  required_diameter_m: number;
  center_xyz: [number, number, number];
  status: ComplianceStatus;
}

export interface LifeSafetySummary {
  total_checks: number;
  passed_checks: number;
  failed_checks: number;
  warning_checks: number;
  compliance_score_pct: number;
  overall_status: "pass" | "fail";
  max_egress_distance_m: number;
  total_doors_audited: number;
  total_rooms_audited: number;
  total_occupant_load: number;
}

export interface LifeSafetyReport {
  world_id: string;
  project_name: string;
  timestamp: string;
  occupancy_group: string;
  is_sprinklered: boolean;
  total_occupant_load: number;
  summary: LifeSafetySummary;
  checks: ComplianceCheckItem[];
  egress_routes: EgressRoute[];
  turning_spaces: TurningSpaceCheck[];
}

export interface CategoryMeta {
  label: string;
  standard: string;
  badgeClass: string;
}

export function getCategoryMeta(category: CodeCheckCategory): CategoryMeta {
  switch (category) {
    case "door_clearance":
      return {
        label: "Door Clear Width",
        standard: "ADA §404 / IBC §1010",
        badgeClass: "text-accent bg-accent/10 border-accent/20",
      };
    case "corridor_width":
      return {
        label: "Corridor Clear Width",
        standard: "ADA §403 / IBC §1020",
        badgeClass: "text-accent bg-accent/10 border-accent/20",
      };
    case "turning_space":
      return {
        label: "Wheelchair Turning Space",
        standard: "ADA 2010 §304",
        badgeClass: "text-accent bg-accent/10 border-accent/20",
      };
    case "egress_distance":
      return {
        label: "Egress Travel Distance",
        standard: "IBC 2024 §1017",
        badgeClass: "text-accent bg-accent/10 border-accent/20",
      };
    case "door_swing":
      return {
        label: "Door Swing Direction",
        standard: "IBC §1010.1.2",
        badgeClass: "text-accent bg-accent/10 border-accent/20",
      };
    case "occupant_load":
      return {
        label: "Occupant Load Density",
        standard: "IBC Table 1004.5",
        badgeClass: "text-accent bg-accent/10 border-accent/20",
      };
  }
}

export function formatComplianceStatus(status: ComplianceStatus): {
  label: string;
  colorClass: string;
  bgClass: string;
} {
  switch (status) {
    case "pass":
      return { label: "PASS", colorClass: "text-success", bgClass: "bg-success/10 border-success/20" };
    case "fail":
      return { label: "FAIL", colorClass: "text-danger", bgClass: "bg-danger/10 border-danger/20" };
    case "warning":
      return { label: "WARNING", colorClass: "text-warn", bgClass: "bg-warn/10 border-warn/20" };
    case "review_required":
      return { label: "REVIEW", colorClass: "text-warn", bgClass: "bg-warn/10 border-warn/20" };
  }
}

export function formatDistance(meters: number): string {
  const feet = meters * 3.28084;
  return `${meters.toFixed(2)} m (${feet.toFixed(1)} ft)`;
}
