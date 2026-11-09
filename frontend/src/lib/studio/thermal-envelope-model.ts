/**
 * Data models and token helpers for Building Envelope Thermal Performance
 * & ASHRAE 90.1 / IECC Energy Code Verification.
 */

export type ClimateZone =
  | "1A_very_hot_humid"
  | "2A_hot_humid"
  | "3A_warm_humid"
  | "4A_mixed_humid"
  | "5A_cold"
  | "6A_cold"
  | "7_very_cold"
  | "8_subarctic";

export const CLIMATE_ZONE_LABELS: Record<ClimateZone, string> = {
  "1A_very_hot_humid": "Zone 1A (Very Hot - Humid e.g. Miami, HI)",
  "2A_hot_humid": "Zone 2A (Hot - Humid e.g. Houston, New Orleans)",
  "3A_warm_humid": "Zone 3A (Warm - Humid e.g. Atlanta, Dallas)",
  "4A_mixed_humid": "Zone 4A (Mixed - Humid e.g. New York, DC)",
  "5A_cold": "Zone 5A (Cold e.g. Chicago, Boston, Denver)",
  "6A_cold": "Zone 6A (Cold e.g. Minneapolis, Burlington)",
  "7_very_cold": "Zone 7 (Very Cold e.g. Duluth, Anchorage)",
  "8_subarctic": "Zone 8 (Subarctic e.g. Fairbanks, Nome)",
};

export type EnvelopeComponentType =
  | "exterior_wall"
  | "roof"
  | "fenestration"
  | "exterior_door"
  | "slab_on_grade";

export type ThermalStatus =
  | "high_efficiency"
  | "compliant"
  | "non_compliant"
  | "thermal_bridge";

export interface EnvelopeSurfaceItem {
  surface_id: string;
  component_type: EnvelopeComponentType;
  name: string;
  orientation: "north" | "south" | "east" | "west" | "horizontal" | string;
  area_m2: number;
  u_factor_w_m2k: number;
  r_value_si: number;
  r_value_ip: number;
  target_u_w_m2k: number;
  ua_w_k: number;
  status: ThermalStatus;
  heat_loss_watts_at_design_dt: number;
  shgc: number | null;
  max_shgc: number | null;
}

export interface ThermalEnvelopeSummary {
  gross_exterior_wall_area_m2: number;
  total_glazing_area_m2: number;
  window_to_wall_ratio_pct: number;
  wwr_prescriptive_compliant: boolean;
  max_prescriptive_wwr_pct: number;
  total_roof_area_m2: number;
  total_ua_actual_w_k: number;
  total_ua_target_w_k: number;
  ua_tradeoff_margin_pct: number;
  total_design_heat_loss_kw: number;
  surface_count: number;
}

export interface ThermalEnvelopeReport {
  world_id: string;
  project_name: string;
  timestamp: string;
  climate_zone: string;
  design_delta_t_k: number;
  summary: ThermalEnvelopeSummary;
  surfaces: EnvelopeSurfaceItem[];
  compliance_verdict: "COMPLIANT" | "NON_COMPLIANT";
}

export function formatThermalStatus(status: ThermalStatus): {
  label: string;
  colorClass: string;
  bgClass: string;
} {
  switch (status) {
    case "high_efficiency":
      return { label: "HIGH EFFICIENCY", colorClass: "text-success", bgClass: "bg-success/10 border-success/20" };
    case "compliant":
      return { label: "COMPLIANT", colorClass: "text-success", bgClass: "bg-success/10 border-success/20" };
    case "non_compliant":
      return { label: "NON-COMPLIANT", colorClass: "text-danger", bgClass: "bg-danger/10 border-danger/20" };
    case "thermal_bridge":
      return { label: "THERMAL BRIDGE", colorClass: "text-warn", bgClass: "bg-warn/10 border-warn/20" };
  }
}

export function formatUValue(uVal: number): string {
  const ipU = uVal * 0.17611;
  return `${uVal.toFixed(3)} W/m²K (U-${ipU.toFixed(2)})`;
}

export function formatRValue(rValSI: number): string {
  const ipR = rValSI * 5.67826;
  return `R-${ipR.toFixed(1)} (${rValSI.toFixed(2)} m²K/W)`;
}

export function formatClimateZone(zone: string): string {
  return CLIMATE_ZONE_LABELS[zone as ClimateZone] ?? zone;
}
