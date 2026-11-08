/**
 * Studio Automated Issue & Clash Detection Client Model (BCF 2.1 / 3.0).
 *
 * Provides models and helper algorithms for multi-trade geometric clash detection:
 * - Clash types: Hard, Clearance, Duplicate
 * - Clash severities: Critical, Major, Minor, Info
 * - Color-coded semantic styles for 3D viewport markers and UI badges
 * - BCF camera viewpoint representation
 */

export type ClashType = "hard" | "clearance" | "duplicate";

export type ClashSeverity = "critical" | "major" | "minor" | "info";

export type ClashStatus = "active" | "resolved" | "review" | "approved";

export interface ClashViewpoint {
  camera_position: [number, number, number];
  camera_direction: [number, number, number];
  camera_up: [number, number, number];
  field_of_view: number;
}

export interface ClashItem {
  id: string;
  clash_type: ClashType;
  severity: ClashSeverity;
  status: ClashStatus;
  entity_a_id: string;
  entity_b_id: string;
  entity_a_name: string;
  entity_b_name: string;
  trade_a: string;
  trade_b: string;
  entity_a_type?: string;
  entity_b_type?: string;
  penetration_depth_m: number;
  clearance_distance_m: number;
  intersection_volume_m3: number;
  clash_point: [number, number, number];
  bounds_min: [number, number, number];
  bounds_max: [number, number, number];
  viewpoint?: ClashViewpoint | null;
  title: string;
  description: string;
  assigned_to?: string;
  created_at_iso: string;
}

export interface ClashReportSummary {
  world_id: string;
  timestamp_iso: string;
  total_clashes: number;
  hard_clashes: number;
  clearance_clashes: number;
  duplicate_clashes: number;
  by_severity: Record<string, number>;
  by_trade_pair: Record<string, number>;
  items: ClashItem[];
}

export interface ClashColorToken {
  border: string;
  bg: string;
  text: string;
  hex: number;
}

/**
 * Returns standardized color tokens for clash severity levels.
 */
export function getClashSeverityColor(severity: ClashSeverity): ClashColorToken {
  switch (severity) {
    case "critical":
      return { border: "border-error", bg: "bg-error-subtle", text: "text-error", hex: 0xf43f5e };
    case "major":
      return { border: "border-warning", bg: "bg-warning-subtle", text: "text-warning", hex: 0xf59e0b };
    case "minor":
      return { border: "border-info", bg: "bg-info-subtle", text: "text-info", hex: 0x0ea5e9 };
    case "info":
    default:
      return { border: "border-line-strong", bg: "bg-elevated/40", text: "text-fg-muted", hex: 0x71717a };
  }
}

/**
 * Returns standardized color tokens for clash types.
 */
export function getClashTypeColor(type: ClashType): ClashColorToken {
  switch (type) {
    case "hard":
      return { border: "border-error", bg: "bg-error-subtle", text: "text-error", hex: 0xf43f5e };
    case "clearance":
      return { border: "border-warning", bg: "bg-warning-subtle", text: "text-warning", hex: 0xf59e0b };
    case "duplicate":
      return { border: "border-purple", bg: "bg-purple-subtle", text: "text-purple", hex: 0xa855f7 };
    default:
      return { border: "border-line", bg: "bg-elevated", text: "text-fg-muted", hex: 0x71717a };
  }
}

/**
 * Summarizes counts by severity.
 */
export function countClashesBySeverity(clashes: ClashItem[]): Record<ClashSeverity, number> {
  const counts: Record<ClashSeverity, number> = {
    critical: 0,
    major: 0,
    minor: 0,
    info: 0,
  };
  for (const c of clashes) {
    if (c.severity in counts) {
      counts[c.severity]++;
    }
  }
  return counts;
}

/**
 * Filter clashes by severity, type, or search term.
 */
export function filterClashes(
  clashes: ClashItem[],
  severityFilter: ClashSeverity | "all" = "all",
  typeFilter: ClashType | "all" = "all",
  searchQuery = "",
): ClashItem[] {
  const q = searchQuery.toLowerCase().trim();
  return clashes.filter((c) => {
    if (severityFilter !== "all" && c.severity !== severityFilter) return false;
    if (typeFilter !== "all" && c.clash_type !== typeFilter) return false;
    if (q) {
      const matchTitle = c.title.toLowerCase().includes(q);
      const matchDesc = c.description.toLowerCase().includes(q);
      const matchEntA = c.entity_a_name.toLowerCase().includes(q);
      const matchEntB = c.entity_b_name.toLowerCase().includes(q);
      if (!matchTitle && !matchDesc && !matchEntA && !matchEntB) return false;
    }
    return true;
  });
}
