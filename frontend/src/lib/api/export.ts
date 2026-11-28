import { apiPost, apiGet, isApiError } from "./client";

export type ExportFormat = "gltf" | "usda" | "ifc" | "blender" | "cityjson" | "citygml" | "geojson" | "godot";

export interface ExportResult {
  version_id: string;
  format: ExportFormat;
  artifact_uri: string;
  download_url: string;
  content_hash: string;
  entities_exported: string[];
  entities_skipped: string[];
  skip_reasons: string[];
}

/** Exports the world's current (or ?version=) WorldIR to a real format
 * via sdk.reality.export() -- POST /api/worlds/{id}/export. */
export function exportWorld(
  worldId: string,
  format: ExportFormat,
  version?: string,
): Promise<ExportResult> {
  return apiPost<ExportResult>(`/api/worlds/${encodeURIComponent(worldId)}/export`, {
    format,
    version: version ?? null,
  });
}

/** Download an exported artifact, showing error toasts on failure.
 * Handles both direct API access and Next.js proxy routes.
 * Returns the blob for further processing if needed. */
export async function downloadArtifact(
  result: ExportResult,
  onError?: (message: string) => void,
): Promise<Blob | null> {
  try {
    const response = await apiGet<Blob>(result.download_url, {
      // Use no-cache to ensure we get the latest artifact
      cache: "no-store",
    });
    return response;
  } catch (err) {
    const message = isApiError(err)
      ? err.describe()
      : "Failed to download artifact";
    console.error("Artifact download error:", err);
    onError?.(message);
    return null;
  }
}
