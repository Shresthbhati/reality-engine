import { apiBaseUrl, apiGet, apiPost } from "./client";
import type { ThermalEnvelopeReport } from "@/lib/studio/thermal-envelope-model";

export interface EvaluateThermalOptions {
  versionId?: string;
  climateZone?: string;
  designDeltaTK?: number;
  customUOverrides?: Record<string, number>;
  projectName?: string;
}

/**
 * Fetch building envelope thermal report & ASHRAE 90.1 compliance.
 */
export async function fetchThermalEnvelopeReport(
  worldId: string,
  options: EvaluateThermalOptions = {},
): Promise<ThermalEnvelopeReport> {
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.climateZone) params.set("climate_zone", options.climateZone);
  if (options.designDeltaTK !== undefined) params.set("design_delta_t_k", String(options.designDeltaTK));

  const qs = params.toString() ? `?${params.toString()}` : "";
  return apiGet<ThermalEnvelopeReport>(`/api/worlds/${encodeURIComponent(worldId)}/thermal-envelope${qs}`);
}

/**
 * Evaluate envelope thermal performance with custom U-factors and climate parameters.
 */
export async function evaluateCustomThermalScenario(
  worldId: string,
  options: EvaluateThermalOptions = {},
): Promise<ThermalEnvelopeReport> {
  return apiPost<ThermalEnvelopeReport>(`/api/worlds/${encodeURIComponent(worldId)}/thermal-envelope/evaluate`, {
    version_id: options.versionId ?? null,
    climate_zone: options.climateZone ?? "5A_cold",
    design_delta_t_k: options.designDeltaTK ?? 25.0,
    custom_u_overrides: options.customUOverrides ?? null,
    project_name: options.projectName ?? null,
  });
}

/**
 * Get direct URL to download formal PDF-1.4 ASHRAE 90.1 Thermal Certificate.
 */
export function getThermalCertificateUrl(
  worldId: string,
  options: EvaluateThermalOptions = {},
): string {
  const base = apiBaseUrl();
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.climateZone) params.set("climate_zone", options.climateZone);

  const qs = params.toString() ? `?${params.toString()}` : "";
  return `${base}/api/worlds/${encodeURIComponent(worldId)}/thermal-envelope/certificate${qs}`;
}

/**
 * Download formal PDF-1.4 ASHRAE 90.1 / IECC Thermal Certificate.
 */
export async function downloadThermalCertificate(
  worldId: string,
  options: EvaluateThermalOptions = {},
): Promise<void> {
  const url = getThermalCertificateUrl(worldId, options);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download thermal envelope certificate: HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = downloadUrl;
  a.download = `thermal_envelope_certificate_${worldId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(downloadUrl);
}
