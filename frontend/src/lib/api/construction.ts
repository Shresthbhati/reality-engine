import { apiPost } from "./client";

export type Trade = "structure" | "architecture" | "mep" | "finishes" | "other";

export type ComplianceStatus =
  | "completed"
  | "misaligned"
  | "in_progress"
  | "missing"
  | "unplanned";

export interface ElementCompliance {
  element_id: string;
  planned_id: string | null;
  name: string;
  trade: Trade;
  entity_type: string;
  status: ComplianceStatus;
  deviation_m: number;
  deviation_vector: [number, number, number];
  tolerance_m: number;
  planned_pos: [number, number, number] | null;
  observed_pos: [number, number, number] | null;
  confidence: number;
  notes: string;
}

export interface TradeProgressSummary {
  trade: Trade;
  planned_count: number;
  completed_count: number;
  misaligned_count: number;
  in_progress_count: number;
  missing_count: number;
  unplanned_count: number;
  completion_pct: number;
}

export interface SiteComplianceReportSummary {
  design_world_id: string;
  site_world_id: string;
  inspection_date: string;
  default_tolerance_m: number;
  overall_completion_pct: number;
  total_planned: number;
  completed_count: number;
  misaligned_count: number;
  in_progress_count: number;
  missing_count: number;
  unplanned_count: number;
  alignment_residual_m: number;
  alignment_converged: boolean;
  trades: Record<string, TradeProgressSummary>;
}

export interface SiteComplianceReport {
  summary: SiteComplianceReportSummary;
  elements: ElementCompliance[];
}

export interface TimelineMilestone {
  epoch_id: string;
  timestamp_iso: string;
  overall_completion_pct: number;
  elements_completed: number;
  elements_in_progress: number;
  elements_missing: number;
  elements_unplanned: number;
  net_velocity_elements_per_day: number;
}

export interface ConstructionTimelineReport {
  project_id: string;
  milestones: TimelineMilestone[];
  total_epochs: number;
  rework_count: number;
  rework_details: string[];
}

/** Verify as-built site world against design plan model */
export function verifyPlan(
  designWorldId: string,
  siteWorldId: string,
  toleranceM: number = 0.05,
  autoAlign: boolean = true
): Promise<SiteComplianceReport> {
  return apiPost<SiteComplianceReport>("/api/construction/verify", {
    design_world_id: designWorldId,
    site_world_id: siteWorldId,
    tolerance_m: toleranceM,
    auto_align: autoAlign,
  });
}

/** Track multi-epoch site construction progression over time */
export function fetchConstructionTimeline(
  designWorldId: string,
  epochWorldIds: string[],
  toleranceM: number = 0.05
): Promise<ConstructionTimelineReport> {
  return apiPost<ConstructionTimelineReport>("/api/construction/timeline", {
    design_world_id: designWorldId,
    epoch_world_ids: epochWorldIds,
    tolerance_m: toleranceM,
  });
}

/** Download the formal Construction Progress PDF inspection report */
export async function downloadConstructionPdf(
  designWorldId: string,
  siteWorldId: string,
  projectName: string = "Construction Site As-Built Verification",
  toleranceM: number = 0.05
): Promise<void> {
  const res = await fetch("/api/construction/report/pdf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      design_world_id: designWorldId,
      site_world_id: siteWorldId,
      project_name: projectName,
      tolerance_m: toleranceM,
    }),
  });

  if (!res.ok) {
    throw new Error(`Failed to download report: HTTP ${res.status}`);
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `construction_progress_${siteWorldId}.pdf`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}
