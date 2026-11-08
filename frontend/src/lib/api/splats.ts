import { ApiError, apiBaseUrl, apiGet, apiPost } from "./client";

export interface SplatMetadata {
  available: boolean;
  world_id: string;
  version_id: string;
  splat_count: number;
  bounds: { min: [number, number, number]; max: [number, number, number] } | null;
  byte_size: number;
  formats: string[];
  artifact_uri: string | null;
  generated_from_points?: boolean;
  can_generate: boolean;
  updated_at?: string;
}

export interface SplatGenerateResponse {
  success: boolean;
  world_id: string;
  version_id: string;
  splat_count: number;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  byte_size: number;
  artifact_uri: string;
  formats: string[];
}

export interface SplatUploadResponse {
  success: boolean;
  world_id: string;
  version_id: string;
  splat_count: number;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  byte_size: number;
  artifact_uri: string;
  formats: string[];
}

/**
 * Fetch Gaussian Splatting metadata for the specified or current world version.
 */
export async function fetchSplatMetadata(worldId: string, versionId?: string): Promise<SplatMetadata> {
  const query = versionId ? `?version=${encodeURIComponent(versionId)}` : "";
  return apiGet<SplatMetadata>(`/api/worlds/${encodeURIComponent(worldId)}/splats${query}`);
}

/**
 * Fetch raw binary .splat or .ply stream for client-side viewport rasterization.
 */
export async function fetchSplatBuffer(
  worldId: string,
  options?: {
    versionId?: string;
    format?: "splat" | "ply";
    maxPoints?: number;
  },
): Promise<ArrayBuffer> {
  const params = new URLSearchParams();
  if (options?.versionId) params.set("version", options.versionId);
  if (options?.format) params.set("format", options.format);
  if (options?.maxPoints) params.set("max_points", String(options.maxPoints));

  const qs = params.toString() ? `?${params.toString()}` : "";
  const target = `${apiBaseUrl()}/api/worlds/${encodeURIComponent(worldId)}/splats/stream${qs}`;

  const res = await fetch(target, { cache: "no-store" });
  if (!res.ok) {
    throw new ApiError({
      message: `Failed to stream Gaussian splats (HTTP ${res.status})`,
      status: res.status,
      code: res.status === 404 ? "not_found" : "server_error",
      retryable: res.status >= 500,
    });
  }

  return res.arrayBuffer();
}

/**
 * Generate 3D Gaussian Splats from the world's existing reconstruction point geometry.
 */
export async function generateWorldSplats(
  worldId: string,
  options?: {
    defaultRadius?: number;
    maxPoints?: number;
    versionId?: string;
  },
): Promise<SplatGenerateResponse> {
  const query = options?.versionId ? `?version=${encodeURIComponent(options.versionId)}` : "";
  return apiPost<SplatGenerateResponse>(
    `/api/worlds/${encodeURIComponent(worldId)}/splats/generate${query}`,
    {
      default_radius: options?.defaultRadius ?? 0.03,
      max_points: options?.maxPoints,
    },
  );
}

/**
 * Upload custom .splat or .ply Gaussian Splat file for a world.
 */
export async function uploadWorldSplats(
  worldId: string,
  file: File,
  versionId?: string,
): Promise<SplatUploadResponse> {
  const query = versionId ? `?version=${encodeURIComponent(versionId)}` : "";
  const target = `${apiBaseUrl()}/api/worlds/${encodeURIComponent(worldId)}/splats/upload${query}`;

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(target, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { detail?: string } | null;
    throw new ApiError({
      message: body?.detail ?? `Failed to upload Gaussian splats (HTTP ${res.status})`,
      status: res.status,
      code: res.status === 422 ? "validation_failed" : "server_error",
      retryable: res.status >= 500,
    });
  }

  return res.json();
}
