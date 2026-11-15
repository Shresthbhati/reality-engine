/**
 * "Needs attention" on Home, derived ONLY from real jobs (GET /api/jobs). Nothing here is invented: every item is a job
 * the backend reported, with its own error / degradation reasons / stage.
 *
 * Rules:
 * - one item per (job type, entity): the LATEST job decides, so a failure that a later run fixed disappears;
 * - failed -> partial -> running/queued, newest first within a group; cancelled and succeeded jobs never appear;
 * - failed/partial items older than STALE_AFTER_DAYS drop off (an old failure is history, not a to-do).
 */
import type { JobDto } from "../api/types";
import { toJobStatus } from "../api/job-status.ts";

export type AttentionKind = "failed" | "partial" | "running";

export interface AttentionItem {
  /** The job id. */
  id: string;
  kind: AttentionKind;
  title: string;
  detail: string | null;
  /** Where to look; null when the job names no entity the app can open. */
  href: string | null;
  at: string | null;
}

export const STALE_AFTER_DAYS = 14;

const LABEL: Record<string, string> = {
  RECONSTRUCT_SESSION: "Reconstruction",
  PROCESS_EVIDENCE: "Photo processing",
};

const TITLE: Record<AttentionKind, (label: string) => string> = {
  failed: (l) => `${l} failed`,
  partial: (l) => `${l} finished with problems`,
  running: (l) => `${l} in progress`,
};

const ORDER: Record<AttentionKind, number> = { failed: 0, partial: 1, running: 2 };

function kindOf(raw: string): AttentionKind | null {
  switch (toJobStatus(raw)) {
    case "failed":
      return "failed";
    case "partial":
      return "partial";
    case "running":
    case "queued":
      return "running";
    case "succeeded":
    case "cancelled":
      return null;
    case "unknown":
      return null; // not shown as attention: an unreadable status is neither a success nor a known problem
  }
}

function when(j: JobDto): string | null {
  return j.completed_at ?? j.started_at ?? j.created_at ?? null;
}

function hrefFor(j: JobDto): string | null {
  const worldId = typeof j.payload?.world_id === "string" ? j.payload.world_id : null;
  if (worldId) return `/worlds/${encodeURIComponent(worldId)}`;
  if (!j.entity_id) return null;
  if (j.entity_type === "session") return `/sessions/${encodeURIComponent(j.entity_id)}`;
  if (j.entity_type === "world") return `/worlds/${encodeURIComponent(j.entity_id)}`;
  if (j.entity_type === "evidence") return `/evidence/${encodeURIComponent(j.entity_id)}`;
  return null;
}

/**
 * The API stores a failed job's raw Python traceback (apps/api/jobs.py), which names local file paths and is unreadable.
 * Show only the final line, without the exception class (mirrors apps/api failure_message()). A plain message passes through.
 */
export function readableFailure(error: string | null | undefined): string | null {
  if (!error) return null;
  const lines = error.trim().split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return null;
  if (lines.length === 1 && !error.startsWith("Traceback")) return lines[0];
  return lines[lines.length - 1].replace(/^[A-Za-z_][\w.]*(?:Error|Exception|Insufficient\w*|Failed)?:\s+/, "") || null;
}

function detailFor(j: JobDto, kind: AttentionKind): string | null {
  if (kind === "failed") return readableFailure(j.error);
  if (kind === "partial") {
    const reasons = j.payload?.degraded;
    if (Array.isArray(reasons)) {
      const text = reasons.filter((r): r is string => typeof r === "string");
      if (text.length) return text.slice(0, 2).join("; ") + (text.length > 2 ? ` (+${text.length - 2} more)` : "");
    }
    return null;
  }
  return j.stage ? `Stage: ${j.stage}` : null;
}

export function attentionItems(
  jobs: JobDto[] | null | undefined,
  now: Date = new Date(),
  max = 6,
): AttentionItem[] {
  if (!jobs || jobs.length === 0) return [];
  const cutoff = now.getTime() - STALE_AFTER_DAYS * 86_400_000;

  // latest job per (type, entity); a missing timestamp sorts oldest, ties keep list order (the API lists newest first)
  const latest = new Map<string, JobDto>();
  const ts = (j: JobDto) => Date.parse(j.created_at ?? "") || 0;
  for (const j of jobs) {
    const key = `${j.type}\u0000${j.entity_id}`;
    const cur = latest.get(key);
    if (!cur || ts(j) > ts(cur)) latest.set(key, j);
  }

  const items: AttentionItem[] = [];
  for (const j of latest.values()) {
    const kind = kindOf(j.status);
    if (!kind) continue;
    const at = when(j);
    if (kind !== "running" && at && Date.parse(at) < cutoff) continue;
    items.push({
      id: j.id,
      kind,
      title: TITLE[kind](LABEL[j.type] ?? j.type),
      detail: detailFor(j, kind),
      href: hrefFor(j),
      at,
    });
  }
  items.sort((a, b) => ORDER[a.kind] - ORDER[b.kind] || (Date.parse(b.at ?? "") || 0) - (Date.parse(a.at ?? "") || 0));
  return items.slice(0, max);
}
