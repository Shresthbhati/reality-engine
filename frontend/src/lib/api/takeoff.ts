import { apiBaseUrl, apiGet, apiPost } from "./client";
import type {
  TakeoffReport,
  UnitRateOverride,
} from "@/lib/studio/takeoff-model";

export interface CustomEstimateOptions {
  projectName?: string;
  rateOverrides?: Record<string, UnitRateOverride>;
  wasteOverrides?: Record<string, number>;
}

/**
 * Fetch the automated Bill of Quantities (BoQ) and Material Takeoff (MTO) for a world.
 */
export async function fetchMaterialTakeoff(worldId: string): Promise<TakeoffReport> {
  return apiGet<TakeoffReport>(`/api/worlds/${encodeURIComponent(worldId)}/takeoff`);
}

/**
 * Calculate a custom cost scenario with user-specified unit rates and waste overrides.
 */
export async function calculateCustomEstimate(
  worldId: string,
  options: CustomEstimateOptions = {},
): Promise<TakeoffReport> {
  return apiPost<TakeoffReport>(`/api/worlds/${encodeURIComponent(worldId)}/takeoff/estimate`, {
    project_name: options.projectName ?? null,
    rate_overrides: options.rateOverrides ?? null,
    waste_overrides: options.wasteOverrides ?? null,
  });
}

/**
 * Get direct URL to download Takeoff CSV spreadsheet.
 */
export function getTakeoffCsvExportUrl(worldId: string): string {
  const base = apiBaseUrl();
  return `${base}/api/worlds/${encodeURIComponent(worldId)}/takeoff/export/csv`;
}

/**
 * Get direct URL to download PDF-1.4 Takeoff Certificate report.
 */
export function getTakeoffPdfExportUrl(worldId: string, projectName?: string): string {
  const base = apiBaseUrl();
  const params = new URLSearchParams();
  if (projectName) params.set("project_name", projectName);
  const qs = params.toString() ? `?${params.toString()}` : "";
  return `${base}/api/worlds/${encodeURIComponent(worldId)}/takeoff/export/pdf${qs}`;
}

/**
 * Download Takeoff CSV file via browser download trigger.
 */
export async function downloadTakeoffCsv(worldId: string): Promise<void> {
  const url = getTakeoffCsvExportUrl(worldId);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download takeoff CSV: HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = downloadUrl;
  a.download = `takeoff_${worldId.slice(0, 8)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(downloadUrl);
}

/**
 * Download Takeoff PDF report via browser download trigger.
 */
export async function downloadTakeoffPdf(worldId: string, projectName?: string): Promise<void> {
  const url = getTakeoffPdfExportUrl(worldId, projectName);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download takeoff PDF: HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = downloadUrl;
  a.download = `boq_takeoff_${worldId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(downloadUrl);
}
