import { apiBaseUrl, apiGet, apiPost } from "./client";
import type {
  HeatmapData,
  SurfaceQualityReport,
} from "@/lib/studio/surface-quality-model";

export interface SurfaceQualityOptions {
  versionId?: string;
  targetFf?: number;
  targetFl?: number;
  targetDeflectionCode?: string;
}

export interface HeatmapOptions {
  entityId?: string;
  versionId?: string;
  gridResolutionM?: number;
  maxDeviationThresholdMm?: number;
}

/**
 * Retrieve surface quality audit report (ASTM E1155 FF/FL and beam deflection).
 */
export async function fetchSurfaceQuality(
  worldId: string,
  options: SurfaceQualityOptions = {},
): Promise<SurfaceQualityReport> {
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.targetFf !== undefined) params.set("target_ff", String(options.targetFf));
  if (options.targetFl !== undefined) params.set("target_fl", String(options.targetFl));
  if (options.targetDeflectionCode) params.set("target_deflection_code", options.targetDeflectionCode);

  const qs = params.toString() ? `?${params.toString()}` : "";
  return apiGet<SurfaceQualityReport>(`/api/worlds/${encodeURIComponent(worldId)}/surface-quality${qs}`);
}

/**
 * Trigger parameterized ASTM E1155 floor flatness and structural deflection audit.
 */
export async function analyzeSurfaceQuality(
  worldId: string,
  options: SurfaceQualityOptions = {},
): Promise<SurfaceQualityReport> {
  return apiPost<SurfaceQualityReport>(`/api/worlds/${encodeURIComponent(worldId)}/surface-quality/analyze`, {
    version_id: options.versionId ?? null,
    target_ff: options.targetFf ?? 25.0,
    target_fl: options.targetFl ?? 20.0,
    target_deflection_code: options.targetDeflectionCode ?? "L/360",
  });
}

/**
 * Retrieve high-resolution surface deviation heatmap for 3D viewport visualization.
 */
export async function fetchSurfaceHeatmap(
  worldId: string,
  options: HeatmapOptions = {},
): Promise<HeatmapData> {
  const params = new URLSearchParams();
  if (options.entityId) params.set("entity_id", options.entityId);
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.gridResolutionM !== undefined) params.set("grid_resolution_m", String(options.gridResolutionM));
  if (options.maxDeviationThresholdMm !== undefined) {
    params.set("max_deviation_threshold_mm", String(options.maxDeviationThresholdMm));
  }

  const qs = params.toString() ? `?${params.toString()}` : "";
  return apiGet<HeatmapData>(`/api/worlds/${encodeURIComponent(worldId)}/surface-quality/heatmap${qs}`);
}

/**
 * Download ASTM E1155 / ACI 117 surface quality compliance certificate as PDF.
 */
export async function downloadSurfaceCertificatePdf(
  worldId: string,
  projectName = "Floor Flatness & Structural Deflection Audit",
): Promise<void> {
  const base = apiBaseUrl();
  const url = `${base}/api/worlds/${encodeURIComponent(worldId)}/surface-quality/certificate?project_name=${encodeURIComponent(projectName)}`;

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download certificate: HTTP ${res.status}`);
  }

  const blob = await res.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = downloadUrl;
  a.download = `surface_compliance_${worldId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(downloadUrl);
}
