import { apiGet, apiPost } from "./client";

export type ExportFormat =
  | "gltf"
  | "glb"
  | "usda"
  | "obj"
  | "ply"
  | "stl"
  | "ifc"
  | "blender"
  | "cityjson"
  | "citygml"
  | "geojson"
  | "godot"
  | "floorplan_svg"
  | "floorplan_dxf"
  | "room_report_pdf";

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
  // 60s, not the client's 15s default: building a gltf/godot bundle for a big world can exceed it
  return apiPost<ExportResult>(
    `/api/worlds/${encodeURIComponent(worldId)}/export`,
    { format, version: version ?? null },
    { timeoutMs: 60_000 },
  );
}

export interface ExportFormatStatus {
  format: string;
  available: boolean;
  reason: string | null;
}

/** Which formats this world can export right now, with a one-line reason when not. */
export async function listExportFormats(worldId: string): Promise<ExportFormatStatus[]> {
  const res = await apiGet<{ formats: ExportFormatStatus[] }>(
    `/api/worlds/${encodeURIComponent(worldId)}/export/formats`,
  );
  return res.formats;
}
