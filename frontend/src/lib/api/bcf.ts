import { ApiError, apiBaseUrl, apiGet, apiPost } from "./client";
import type { ClashReportSummary } from "@/lib/studio/clash-bcf-model";

export interface ClashDetectOptions {
  versionId?: string;
  hardToleranceM?: number;
  clearanceBufferM?: number;
  checkDuplicates?: boolean;
  checkClearance?: boolean;
}

export interface BcfImportResponse {
  world_id: string;
  filename: string;
  imported_topic_count: number;
  topics: Array<{
    id: string;
    title: string;
    topic_type: string;
    topic_status: string;
    priority: string;
    description: string;
    author: string;
    created_at_iso: string;
    components: string[];
  }>;
}

/**
 * Retrieve detected clashes and coordination issues for a world.
 */
export async function fetchWorldClashes(
  worldId: string,
  options: ClashDetectOptions = {},
): Promise<ClashReportSummary> {
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.hardToleranceM !== undefined) params.set("hard_tolerance_m", String(options.hardToleranceM));
  if (options.clearanceBufferM !== undefined) params.set("clearance_buffer_m", String(options.clearanceBufferM));
  if (options.checkDuplicates !== undefined) params.set("check_duplicates", String(options.checkDuplicates));
  if (options.checkClearance !== undefined) params.set("check_clearance", String(options.checkClearance));

  const qs = params.toString() ? `?${params.toString()}` : "";
  return apiGet<ClashReportSummary>(`/api/worlds/${encodeURIComponent(worldId)}/clashes${qs}`);
}

/**
 * Run parameterized multi-trade clash detection on a world.
 */
export async function detectWorldClashes(
  worldId: string,
  options: ClashDetectOptions = {},
): Promise<ClashReportSummary> {
  return apiPost<ClashReportSummary>(`/api/worlds/${encodeURIComponent(worldId)}/clashes/detect`, {
    version_id: options.versionId ?? null,
    hard_tolerance_m: options.hardToleranceM ?? 0.01,
    clearance_buffer_m: options.clearanceBufferM ?? 0.05,
    check_duplicates: options.checkDuplicates ?? true,
    check_clearance: options.checkClearance ?? true,
  });
}

/**
 * Download a standard buildingSMART BCF 2.1 zip package (.bcfzip).
 */
export async function downloadBcfZip(
  worldId: string,
  options: { versionId?: string; hardToleranceM?: number; clearanceBufferM?: number } = {},
): Promise<Blob> {
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.hardToleranceM !== undefined) params.set("hard_tolerance_m", String(options.hardToleranceM));
  if (options.clearanceBufferM !== undefined) params.set("clearance_buffer_m", String(options.clearanceBufferM));

  const qs = params.toString() ? `?${params.toString()}` : "";
  const target = `${apiBaseUrl()}/api/worlds/${encodeURIComponent(worldId)}/bcf/export${qs}`;

  const res = await fetch(target, { cache: "no-store" });
  if (!res.ok) {
    throw new ApiError({
      message: `Failed to export BCF 2.1 zip package (HTTP ${res.status})`,
      status: res.status,
      code: res.status === 404 ? "not_found" : "server_error",
      retryable: res.status >= 500,
    });
  }
  return res.blob();
}

/**
 * Import an external BCF 2.1 / 3.0 zip package and parse coordination topics.
 */
export async function importBcfPackage(worldId: string, file: File): Promise<BcfImportResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const target = `${apiBaseUrl()}/api/worlds/${encodeURIComponent(worldId)}/bcf/import`;
  const res = await fetch(target, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new ApiError({
      message: `Failed to import BCF package (HTTP ${res.status})`,
      status: res.status,
      code: res.status === 400 ? "validation_failed" : "server_error",
      retryable: res.status >= 500,
    });
  }
  return res.json() as Promise<BcfImportResponse>;
}
