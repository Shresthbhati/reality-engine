/**
 * CSI MasterFormat 2020 Data Models and Formatter Helpers for
 * Architectural Bill of Quantities (BoQ) & Material Takeoff (MTO).
 */

export type CsiDivision =
  | "03_concrete"
  | "04_masonry"
  | "06_wood_plastics"
  | "07_thermal_moisture"
  | "08_openings"
  | "09_finishes"
  | "22_plumbing"
  | "26_electrical";

export const CSI_DIVISION_LABELS: Record<CsiDivision, string> = {
  "03_concrete": "Division 03 - Concrete",
  "04_masonry": "Division 04 - Masonry",
  "06_wood_plastics": "Division 06 - Wood, Plastics & Composites",
  "07_thermal_moisture": "Division 07 - Thermal & Moisture Protection",
  "08_openings": "Division 08 - Openings (Doors & Glazing)",
  "09_finishes": "Division 09 - Finishes (Drywall, Flooring, Paint)",
  "22_plumbing": "Division 22 - Plumbing Fixtures",
  "26_electrical": "Division 26 - Electrical & Lighting",
};

export interface TakeoffItem {
  item_id: string;
  csi_code: string;
  csi_division: CsiDivision;
  csi_division_name: string;
  description: string;
  category: "wall" | "floor" | "ceiling" | "trim" | "opening" | "fixture" | string;
  room_id: string | null;
  room_name: string | null;
  quantity_net: number;
  waste_percent: number;
  quantity_gross: number;
  unit: string;
  unit_cost_material: number;
  unit_cost_labor: number;
  total_cost_material: number;
  total_cost_labor: number;
  total_cost: number;
  notes: string;
}

export interface DivisionSummary {
  division: CsiDivision;
  division_name: string;
  total_material_cost: number;
  total_labor_cost: number;
  total_cost: number;
  item_count: number;
  percentage_of_total: number;
}

export interface RoomTakeoff {
  room_id: string;
  room_name: string;
  floor_area_m2: number;
  perimeter_m: number;
  ceiling_height_m: number;
  wall_gross_area_m2: number;
  wall_opening_area_m2: number;
  wall_net_area_m2: number;
  ceiling_area_m2: number;
  baseboard_length_m: number;
  doors_count: number;
  windows_count: number;
  fixtures_count: number;
  floor_finish: string;
  wall_finish: string;
  ceiling_finish: string;
  total_cost: number;
  items: TakeoffItem[];
}

export interface TakeoffSummary {
  total_floor_area_m2: number;
  total_net_wall_area_m2: number;
  total_openings_count: number;
  total_fixtures_count: number;
  grand_total_material: number;
  grand_total_labor: number;
  grand_total_cost: number;
  cost_per_m2: number;
}

export interface TakeoffReport {
  world_id: string;
  project_name: string;
  timestamp: string;
  currency: string;
  summary: TakeoffSummary;
  divisions: DivisionSummary[];
  rooms: RoomTakeoff[];
  items_count: number;
}

export interface UnitRateOverride {
  material_cost?: number;
  labor_cost?: number;
  waste_percent?: number;
}

export function formatCurrency(amount: number, currency: string = "USD"): string {
  const sym = currency === "USD" ? "$" : `${currency} `;
  return `${sym}${amount.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function formatCsiDivision(division: CsiDivision): string {
  return CSI_DIVISION_LABELS[division] ?? division;
}

export function formatQuantity(qty: number, unit: string): string {
  return `${qty.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${unit}`;
}
