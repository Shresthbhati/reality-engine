import { apiGet, apiPost } from "./client";
import { PRIOR_METHOD, type ScaleSuggestionDto } from "./scale-model";

export * from "./scale-model";

export interface ScaleSuggestionResponse {
  world_id: string;
  scale_state: string;
  suggestion: ScaleSuggestionDto | null;
  rejected: boolean;
  dimension_state: string | null;
  scale_basis: string | null;
  /** why there is no suggestion (null when there is one) */
  reason: string | null;
}

const base = (worldId: string) => `/api/worlds/${encodeURIComponent(worldId)}/scale`;

/** Suggest a typical-size scale from a detected door. Applies nothing. */
export function getScaleSuggestion(worldId: string) {
  return apiGet<ScaleSuggestionResponse>(`${base(worldId)}/suggestion`);
}

/** Accept the suggestion: the world is re-evaluated in metres and its dimensions are labelled ESTIMATED. */
export function acceptScaleSuggestion(worldId: string, s: Pick<ScaleSuggestionDto, "source_entity_id" | "suggested_value_m">) {
  return apiPost<{ job_id: string; dimension_state_after: string }>(base(worldId), {
    method: PRIOR_METHOD,
    source_entity_id: s.source_entity_id,
    distance_m: s.suggested_value_m,
  });
}

/** Reject it: nothing changes, the world stays RELATIVE. */
export function rejectScaleSuggestion(worldId: string, sourceEntityId: string) {
  return apiPost<{ rejected: boolean }>(`${base(worldId)}/suggestion/reject`, { source_entity_id: sourceEntityId });
}

/**
 * Known-length method (backend contract; the viewer point picker that calls it is wired separately):
 * two picked points in viewer/world coordinates + their real distance in metres, or model_length.
 */
export function setKnownLengthScale(
  worldId: string,
  body: {
    distance_m: number;
    point_a?: [number, number, number];
    point_b?: [number, number, number];
    model_length?: number;
  },
) {
  return apiPost<{ job_id: string; dimension_state_after: string }>(base(worldId), { method: "known_length", ...body });
}
