/**
 * Activity — the platform's real event log.
 *
 * Events come from `GET /api/activity`, written by the API and the job worker
 * as things actually happen (session.created, evidence.created,
 * evidence.deleted, world.created, …). Nothing is derived from a static row
 * array any more.
 *
 * Failures propagate. This function used to `catch { return [] }`, which made
 * an unreachable backend render as "No activity recorded yet." — a confident,
 * plausible-looking claim that nothing had happened, when in fact the operator
 * simply could not reach the log. An empty feed and a broken feed are now
 * distinguishable to the caller, which owns the error state.
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

/** Throws `ApiError` when the log cannot be read; an empty array means "no events". */
export async function fetchActivity(limit = 50): Promise<ActivityEvent[]> {
  const data = await apiGet<ActivityResponse>("/api/activity");
  const rows = data.items;
  if (!Array.isArray(rows)) {
    throw new Error("Activity feed did not return an items array.");
  }
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