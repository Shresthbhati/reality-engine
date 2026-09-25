import { apiPost } from "./client";

export type ExportFormat = "gltf" | "usda" | "blender" | "cityjson" | "citygml";

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
