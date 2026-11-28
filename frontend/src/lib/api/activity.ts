/**
 * Activity — the platform's real event log.
 *
 * Events come from `GET /api/activity`, written by the API and the job worker
 * as things actually happen (session.created, evidence.created,
 * evidence.deleted, world.created, …).
 *
 * Contract: this function THROWS on failure. It must not return `[]` for an
 * unreachable backend, because an empty activity list is also the exact,
 * correct rendering for "nothing has happened yet" — the two are
 * indistinguishable to a user, so an outage would read as a quiet day.
 * Callers render `error` as a failure and `[]` as genuinely-no-events.
 */
import { apiGet } from "./client";
import { entityHref } from "./adapters";
import type { ActivityDto } from "./types";

export type ActivityType = "SESSION" | "WORLD" | "EVIDENCE" | "ANALYSIS" | "RESULT" | "REPORT";

export interface ActivityEvent {
  type: ActivityType;
  id: string;
  label: string;
  timestamp: string;
  href: string;
}

export interface ActivityResponse {
  items: ActivityDto[];
}

/** Server activity type → the vocabulary the feed renders. */
function activityTypeFrom(type: string): ActivityType {
  const prefix = type.split(".")[0]?.toLowerCase();
  switch (prefix) {
    case "world":
      return "WORLD";
    case "evidence":
      return "EVIDENCE";
    case "analysis":
      return "ANALYSIS";
    case "result":
      return "RESULT";
    case "report":
      return "REPORT";
    case "session":
    default:
      return "SESSION";
  }
}

/**
 * Real activity events, newest first.
 *
 * Throws (ApiError) if the backend could not be reached or returned an error.
 * Returns `[]` ONLY when the backend answered successfully with no events.
 */
export async function fetchActivity(limit = 50): Promise<ActivityEvent[]> {
  const data = await apiGet<ActivityResponse>("/api/activity");
  const rows = Array.isArray(data.items) ? data.items : [];
  return rows
    .map((row) => {
      const href = entityHref(row.entity_type, row.entity_id);
      if (!href) return null; // no product route for this entity type → no dead link
      return {
        type: activityTypeFrom(row.type),
        id: row.id,
        label: row.summary,
        timestamp: row.created_at ?? "—",
        href,
      } satisfies ActivityEvent;
    })
    .filter((e): e is ActivityEvent => e !== null)
    .slice(0, limit);
}

export const getActivity = fetchActivity;