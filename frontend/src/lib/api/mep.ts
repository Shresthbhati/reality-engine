import { apiBaseUrl, apiGet, apiPost } from "./client";
import type { MepFacilityTopology } from "@/lib/studio/mep-topology-model";

export interface MepDetectOptions {
  versionId?: string;
  systemFilter?: string;
  minClearanceM?: number;
}

export interface MepRouteOptions {
  versionId?: string;
  startPoint: [number, number, number];
  endPoint: [number, number, number];
  ceilingZ?: number;
  systemType?: string;
  diameterMm?: number;
}

export interface MepRouteResponse {
  world_id: string;
  system_type: string;
  diameter_mm: number;
  start_point: [number, number, number];
  end_point: [number, number, number];
  ceiling_z: number;
  total_length_m: number;
  bend_count: number;
  waypoints: [number, number, number][];
}

/**
 * Retrieve perceived MEP facility topology and distribution systems.
 */
export async function fetchMepTopology(
  worldId: string,
  versionId?: string,
): Promise<MepFacilityTopology> {
  const qs = versionId ? `?version_id=${encodeURIComponent(versionId)}` : "";
  return apiGet<MepFacilityTopology>(`/api/worlds/${encodeURIComponent(worldId)}/mep/topology${qs}`);
}

/**
 * Trigger MEP perception, auto-routing, and code clearance verification.
 */
export async function detectMepTopology(
  worldId: string,
  options: MepDetectOptions = {},
): Promise<MepFacilityTopology> {
  return apiPost<MepFacilityTopology>(`/api/worlds/${encodeURIComponent(worldId)}/mep/detect`, {
    version_id: options.versionId ?? null,
    system_filter: options.systemFilter ?? null,
    min_clearance_m: options.minClearanceM ?? 0.15,
  });
}

/**
 * Compute optimal orthogonal route between two 3D coordinates.
 */
export async function computeMepRoute(
  worldId: string,
  options: MepRouteOptions,
): Promise<MepRouteResponse> {
  return apiPost<MepRouteResponse>(`/api/worlds/${encodeURIComponent(worldId)}/mep/route`, {
    version_id: options.versionId ?? null,
    start_point: options.startPoint,
    end_point: options.endPoint,
    ceiling_z: options.ceilingZ ?? 2.4,
    system_type: options.systemType ?? "electrical",
    diameter_mm: options.diameterMm ?? 25.0,
  });
}

/**
 * URL to download buildingSMART IFC4X3 Building Services STEP file.
 */
export function getMepIfcExportUrl(worldId: string, versionId?: string): string {
  const qs = versionId ? `?version_id=${encodeURIComponent(versionId)}` : "";
  return `${apiBaseUrl()}/api/worlds/${encodeURIComponent(worldId)}/mep/export/ifc${qs}`;
}

/**
 * URL to download 3D LineString GeoJSON network representation.
 */
export function getMepGeoJsonExportUrl(worldId: string, versionId?: string): string {
  const qs = versionId ? `?version_id=${encodeURIComponent(versionId)}` : "";
  return `${apiBaseUrl()}/api/worlds/${encodeURIComponent(worldId)}/mep/export/geojson${qs}`;
}

/**
 * Trigger direct download of IFC4X3 Building Services file.
 */
export async function downloadMepIfc(worldId: string, versionId?: string): Promise<void> {
  const url = getMepIfcExportUrl(worldId, versionId);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download IFC: ${res.statusText}`);
  const blob = await res.blob();
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = `mep_topology_${worldId}.ifc`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(blobUrl);
}

/**
 * Trigger direct download of 3D GeoJSON network representation.
 */
export async function downloadMepGeoJson(worldId: string, versionId?: string): Promise<void> {
  const url = getMepGeoJsonExportUrl(worldId, versionId);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download GeoJSON: ${res.statusText}`);
  const blob = await res.blob();
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = `mep_network_${worldId}.geojson`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(blobUrl);
}
