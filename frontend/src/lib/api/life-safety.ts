import { apiBaseUrl, apiGet, apiPost } from "./client";
import type {
  EgressRoute,
  LifeSafetyReport,
} from "@/lib/studio/life-safety-model";

export interface AuditLifeSafetyOptions {
  versionId?: string;
  occupancyGroup?: string;
  isSprinklered?: boolean;
  projectName?: string;
}

/**
 * Fetch the Life Safety & ADA Accessibility Compliance Audit for a world.
 */
export async function fetchLifeSafetyReport(
  worldId: string,
  options: AuditLifeSafetyOptions = {},
): Promise<LifeSafetyReport> {
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.occupancyGroup) params.set("occupancy_group", options.occupancyGroup);
  if (options.isSprinklered !== undefined) params.set("is_sprinklered", String(options.isSprinklered));

  const qs = params.toString() ? `?${params.toString()}` : "";
  return apiGet<LifeSafetyReport>(`/api/worlds/${encodeURIComponent(worldId)}/life-safety${qs}`);
}

/**
 * Run a custom Life Safety & ADA Compliance audit scenario.
 */
export async function runCustomLifeSafetyAudit(
  worldId: string,
  options: AuditLifeSafetyOptions = {},
): Promise<LifeSafetyReport> {
  return apiPost<LifeSafetyReport>(`/api/worlds/${encodeURIComponent(worldId)}/life-safety/audit`, {
    version_id: options.versionId ?? null,
    occupancy_group: options.occupancyGroup ?? "Business (Group B)",
    is_sprinklered: options.isSprinklered ?? true,
    project_name: options.projectName ?? null,
  });
}

/**
 * Fetch egress route polylines and travel metrics.
 */
export async function fetchEgressRoutes(
  worldId: string,
  options: AuditLifeSafetyOptions = {},
): Promise<EgressRoute[]> {
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.isSprinklered !== undefined) params.set("is_sprinklered", String(options.isSprinklered));

  const qs = params.toString() ? `?${params.toString()}` : "";
  return apiGet<EgressRoute[]>(`/api/worlds/${encodeURIComponent(worldId)}/life-safety/egress-path${qs}`);
}

/**
 * Get direct URL to download formal PDF-1.4 Life Safety Certificate.
 */
export function getLifeSafetyCertificateUrl(
  worldId: string,
  options: AuditLifeSafetyOptions = {},
): string {
  const base = apiBaseUrl();
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.occupancyGroup) params.set("occupancy_group", options.occupancyGroup);
  if (options.isSprinklered !== undefined) params.set("is_sprinklered", String(options.isSprinklered));

  const qs = params.toString() ? `?${params.toString()}` : "";
  return `${base}/api/worlds/${encodeURIComponent(worldId)}/life-safety/certificate${qs}`;
}

/**
 * Download formal Life Safety & ADA Accessibility Compliance Certificate PDF.
 */
export async function downloadLifeSafetyCertificate(
  worldId: string,
  options: AuditLifeSafetyOptions = {},
): Promise<void> {
  const url = getLifeSafetyCertificateUrl(worldId, options);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download life safety certificate: HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = downloadUrl;
  a.download = `life_safety_certificate_${worldId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(downloadUrl);
}
