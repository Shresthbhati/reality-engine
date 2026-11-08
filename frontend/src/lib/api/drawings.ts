import { apiBaseUrl, apiGet, apiPost } from "./client";
import type {
  ArchitecturalDrawing,
  DrawingPresetsResponse,
  DrawingType,
  SheetSize,
} from "@/lib/studio/drawing-sheet-model";

export interface GenerateDrawingOptions {
  versionId?: string;
  drawingType?: DrawingType;
  cutOffsetM?: number;
  projectionDepthM?: number;
  scaleLabel?: string;
  sheetSize?: SheetSize;
  drawingTitle?: string;
}

export interface RenderDrawingOptions extends GenerateDrawingOptions {
  theme?: "dark" | "light";
}

/**
 * Retrieve standard architectural section and elevation presets for a world.
 */
export async function fetchDrawingPresets(worldId: string): Promise<DrawingPresetsResponse> {
  return apiGet<DrawingPresetsResponse>(`/api/worlds/${encodeURIComponent(worldId)}/drawings/presets`);
}

/**
 * Generate 2D architectural building section or elevation drawing dataset.
 */
export async function generateDrawing(
  worldId: string,
  options: GenerateDrawingOptions = {},
): Promise<ArchitecturalDrawing> {
  return apiPost<ArchitecturalDrawing>(`/api/worlds/${encodeURIComponent(worldId)}/drawings/generate`, {
    version_id: options.versionId ?? null,
    drawing_type: options.drawingType ?? "section_longitudinal",
    cut_offset_m: options.cutOffsetM ?? null,
    projection_depth_m: options.projectionDepthM ?? 12.0,
    scale_label: options.scaleLabel ?? "1:100",
    sheet_size: options.sheetSize ?? "A2",
    drawing_title: options.drawingTitle ?? null,
  });
}

/**
 * Build URL to render vector SVG drawing sheet directly.
 */
export function getDrawingSvgUrl(worldId: string, options: RenderDrawingOptions = {}): string {
  const base = apiBaseUrl();
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.drawingType) params.set("drawing_type", options.drawingType);
  if (options.cutOffsetM !== undefined && options.cutOffsetM !== null) {
    params.set("cut_offset_m", String(options.cutOffsetM));
  }
  if (options.projectionDepthM !== undefined) {
    params.set("projection_depth_m", String(options.projectionDepthM));
  }
  if (options.scaleLabel) params.set("scale_label", options.scaleLabel);
  if (options.sheetSize) params.set("sheet_size", options.sheetSize);
  if (options.theme) params.set("theme", options.theme);

  const qs = params.toString() ? `?${params.toString()}` : "";
  return `${base}/api/worlds/${encodeURIComponent(worldId)}/drawings/render/svg${qs}`;
}

/**
 * Download AutoCAD-compatible DXF file.
 */
export async function downloadDrawingDxf(
  worldId: string,
  options: GenerateDrawingOptions = {},
): Promise<void> {
  const base = apiBaseUrl();
  const params = new URLSearchParams();
  if (options.versionId) params.set("version_id", options.versionId);
  if (options.drawingType) params.set("drawing_type", options.drawingType);
  if (options.cutOffsetM !== undefined && options.cutOffsetM !== null) {
    params.set("cut_offset_m", String(options.cutOffsetM));
  }
  if (options.scaleLabel) params.set("scale_label", options.scaleLabel);
  if (options.sheetSize) params.set("sheet_size", options.sheetSize);

  const qs = params.toString() ? `?${params.toString()}` : "";
  const url = `${base}/api/worlds/${encodeURIComponent(worldId)}/drawings/export/dxf${qs}`;

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download DXF drawing: HTTP ${res.status}`);
  }

  const blob = await res.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = downloadUrl;
  a.download = `${options.drawingType || "section"}_${worldId.slice(0, 8)}.dxf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(downloadUrl);
}

/**
 * Download standard vector SVG drawing file.
 */
export async function downloadDrawingSvg(
  worldId: string,
  options: RenderDrawingOptions = {},
): Promise<void> {
  const url = getDrawingSvgUrl(worldId, options);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download SVG drawing: HTTP ${res.status}`);
  }

  const blob = await res.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = downloadUrl;
  a.download = `${options.drawingType || "section"}_${worldId.slice(0, 8)}.svg`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(downloadUrl);
}
