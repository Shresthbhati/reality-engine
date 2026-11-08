import { apiBaseUrl, apiDelete, apiGet, apiPost } from "./client";
import type {
  CollabMarkupsState,
  CollabPresenceState,
  PeerPresence,
  SpatialMarkup,
} from "@/lib/studio/collab-presence-model";

export interface HeartbeatPayload {
  client_id: string;
  user_name: string;
  role?: string;
  color?: string;
  camera_position: [number, number, number];
  look_at: [number, number, number];
  field_of_view?: number;
}

export interface CreateMarkupPayload {
  version_id?: string;
  markup_type: string;
  trade: string;
  title: string;
  description?: string;
  author_name?: string;
  author_id?: string;
  position: [number, number, number];
  normal?: [number, number, number];
  stroke_points?: [number, number, number][];
  camera_viewpoint?: {
    position?: [number, number, number];
    direction?: [number, number, number];
    fov?: number;
  };
  color?: string;
  status?: string;
}

/**
 * Retrieve active collaborators present in the 3D model.
 */
export async function fetchCollabPresence(worldId: string): Promise<CollabPresenceState> {
  return apiGet<CollabPresenceState>(`/api/worlds/${encodeURIComponent(worldId)}/collab/presence`);
}

/**
 * Send client heartbeat and 3D camera pose update.
 */
export async function sendCollabHeartbeat(
  worldId: string,
  payload: HeartbeatPayload,
): Promise<{ status: string; peer: PeerPresence; active_peers: PeerPresence[] }> {
  return apiPost(`/api/worlds/${encodeURIComponent(worldId)}/collab/heartbeat`, payload);
}

/**
 * URL for Server-Sent Events (SSE) stream.
 */
export function getCollabEventsUrl(worldId: string): string {
  return `${apiBaseUrl()}/api/worlds/${encodeURIComponent(worldId)}/collab/events`;
}

/**
 * Retrieve all persistent 3D spatial markups and annotations.
 */
export async function fetchCollabMarkups(
  worldId: string,
  versionId?: string,
): Promise<CollabMarkupsState> {
  const qs = versionId ? `?version_id=${encodeURIComponent(versionId)}` : "";
  return apiGet<CollabMarkupsState>(`/api/worlds/${encodeURIComponent(worldId)}/collab/markups${qs}`);
}

/**
 * Create or save a 3D spatial markup.
 */
export async function createSpatialMarkup(
  worldId: string,
  payload: CreateMarkupPayload,
): Promise<{ status: string; markup: SpatialMarkup; total_markups: number; version_id: string }> {
  return apiPost(`/api/worlds/${encodeURIComponent(worldId)}/collab/markups`, payload);
}

/**
 * Delete a 3D spatial markup by ID.
 */
export async function deleteSpatialMarkup(
  worldId: string,
  markupId: string,
  versionId?: string,
): Promise<{ status: string; markup_id: string; version_id: string }> {
  const qs = versionId ? `?version_id=${encodeURIComponent(versionId)}` : "";
  return apiDelete(`/api/worlds/${encodeURIComponent(worldId)}/collab/markups/${encodeURIComponent(markupId)}${qs}`);
}

/**
 * URL to download buildingSMART BCF 2.1 zip package.
 */
export function getCollabBcfExportUrl(worldId: string, versionId?: string): string {
  const qs = versionId ? `?version_id=${encodeURIComponent(versionId)}` : "";
  return `${apiBaseUrl()}/api/worlds/${encodeURIComponent(worldId)}/collab/export/bcf${qs}`;
}

/**
 * Trigger browser download of buildingSMART BCF 2.1 archive.
 */
export async function downloadCollabBcf(worldId: string, versionId?: string): Promise<void> {
  const url = getCollabBcfExportUrl(worldId, versionId);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download BCF package: ${res.statusText}`);
  const blob = await res.blob();
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = `spatial_markups_${worldId}.bcfzip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(blobUrl);
}
