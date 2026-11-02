/** GET /api/worlds/{id}/quality: per-entity quality bands plus the place-level summary (B-07). */
import { apiGet } from "./client";

export type Band = "good" | "fair" | "weak" | "unknown";

export interface EntityQuality {
  band: Band;
  provenance: string;
  confidence: number | null;
  type: string | null;
  name: string;
}

export interface WorldQuality {
  version_id: string;
  band: Band;
  /** Why the place has this band: few_placed, rough_model, partial_build, some_unplaced, or null (good / unknown). */
  band_reason: string | null;
  photos_total: number;
  photos_placed: number;
  scale_state: string | null;
  entity_counts: Record<Band, number>;
  open_conflicts: number;
  entities: Record<string, EntityQuality>;
}

export function getWorldQuality(worldId: string): Promise<WorldQuality> {
  return apiGet<WorldQuality>("/api/worlds/" + encodeURIComponent(worldId) + "/quality");
}
