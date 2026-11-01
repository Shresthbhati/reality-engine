/** Map context: imports OpenStreetMap structure around a coordinate as its own world. */
import { apiPost } from "./client";
import type { GeoContextInput, MapContextResult } from "@/lib/sites/compose-model";

export type { GeoContextInput } from "@/lib/sites/compose-model";

export interface GeoContextResponse extends MapContextResult {
  world_id: string;
  version_id: string;
}

export function loadGeoContext(input: GeoContextInput): Promise<GeoContextResponse> {
  return apiPost<GeoContextResponse>("/api/geo/context", input, { timeoutMs: 60_000 });
}
