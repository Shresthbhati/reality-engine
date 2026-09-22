/**
 * Activity — the platform's real event log.
 *
 * Events come from `GET /api/activity`, written by the API and the job worker
 * as things actually happen (session.created, evidence.created,
 * evidence.deleted, world.created, …). Nothing is derived from a static row
 * array any more, and an unreachable backend yields an empty list plus a
 * thrown error the page can surface.
 */
import { entityHref, listActivity as fetchActivity } from "./api";

export type ActivityType = "SESSION" | "WORLD" | "EVIDENCE" | "ANALYSIS" | "RESULT" | "REPORT";

export interface ActivityEvent {
  type: ActivityType;
  id: string;
  label: string;
  timestamp: string;
  href: string;
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

export async function getActivity(limit = 50): Promise<ActivityEvent[]> {
  const rows = await fetchActivity();
  return rows
    .map((row) => {
      const href = entityHref(row.entityType, row.entityId);
      if (!href) return null; // no product route for this entity type → no dead link
      return {
        type: activityTypeFrom(row.type),
        id: row.id,
        label: row.summary,
        timestamp: row.at ?? "—",
        href,
      } satisfies ActivityEvent;
    })
    .filter((e): e is ActivityEvent => e !== null)
    .slice(0, limit);
}
