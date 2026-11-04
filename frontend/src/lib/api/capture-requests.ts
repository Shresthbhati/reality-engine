import { apiGet, apiPatch } from "./client";

export interface SuggestedPose {
  position?: [number, number, number];
  rotation?: [number, number, number, number];
  description?: string;
  distance_m?: number;
  pitch_deg?: number;
  yaw_deg?: number;
}

export interface CaptureRequestItem {
  id: string;
  world_id: string;
  target_region?: {
    region_type?: string;
    bounds?: number[];
  } | null;
  target_ids?: string[];
  suggested_poses?: SuggestedPose[];
  expected_gain: number;
  instruction: string;
  status: "open" | "satisfied" | "dismissed";
  created_at?: string;
  updated_at?: string;
}

export interface CaptureRequestListResponse {
  items: CaptureRequestItem[];
  count: number;
}

/**
 * Fetch world-space capture requests from the Reality Engine API.
 */
export async function listCaptureRequests(
  worldId: string,
  status?: "open" | "satisfied" | "dismissed"
): Promise<CaptureRequestListResponse> {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  return apiGet<CaptureRequestListResponse>(`/api/worlds/${encodeURIComponent(worldId)}/capture-requests${query}`);
}

/**
 * Update a capture request status (satisfied or dismissed).
 */
export async function updateCaptureRequestStatus(
  worldId: string,
  requestId: string,
  status: "open" | "satisfied" | "dismissed"
): Promise<CaptureRequestItem> {
  return apiPatch<CaptureRequestItem>(
    `/api/worlds/${encodeURIComponent(worldId)}/capture-requests/${encodeURIComponent(requestId)}`,
    { status }
  );
}
