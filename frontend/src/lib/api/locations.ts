import { apiPost } from "./client";
import type { LocationPointPayload } from "@/lib/mobile/gps-track";

/** Post a batch of timestamped GPS fixes to a session (`POST /api/sessions/{id}/locations`). */
export function postSessionLocations(sessionId: string, points: LocationPointPayload[]): Promise<{ added: number; location_ids: string[] }> {
  return apiPost(`/api/sessions/${encodeURIComponent(sessionId)}/locations`, { points });
}
