/**
 * Pure dashboard logic and formatters (Task 11.1 & Blueprint 14).
 * Handles real storage byte formatting and durable jobs categorization.
 */
import type { JobDto, StorageStatusDto } from "../api/types";
import { toJobStatus } from "../api/job-status.ts";

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes < 0) {
    return "0 B";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let val = bytes;
  let unitIndex = -1;
  while (val >= 1024 && unitIndex < units.length - 1) {
    val /= 1024;
    unitIndex++;
  }
  return `${val.toFixed(val < 10 ? 2 : 1)} ${units[unitIndex]}`;
}

export interface JobsSummary {
  active: JobDto[];
  failed: JobDto[];
  recent: JobDto[];
  counts: {
    total: number;
    running: number;
    failed: number;
    succeeded: number;
    queued: number;
  };
}

export function summarizeJobs(jobs: JobDto[] | null | undefined): JobsSummary {
  if (!jobs || jobs.length === 0) {
    return {
      active: [],
      failed: [],
      recent: [],
      counts: { total: 0, running: 0, failed: 0, succeeded: 0, queued: 0 },
    };
  }

  const active: JobDto[] = [];
  const failed: JobDto[] = [];
  let running = 0;
  let succeeded = 0;
  let queued = 0;

  for (const j of jobs) {
    switch (toJobStatus(j.status)) {
      case "running":
        running++;
        active.push(j);
        break;
      case "failed":
        failed.push(j);
        break;
      case "succeeded":
        succeeded++;
        break;
      case "queued":
        queued++;
        active.push(j);
        break;
      case "partial":
      case "cancelled":
        break;
      case "unknown":
        break; // counted only in total; never as succeeded
    }
  }

  // Sort jobs by creation / start time descending
  const sorted = [...jobs].sort((a, b) => {
    const ta = a.created_at ?? a.started_at ?? "";
    const tb = b.created_at ?? b.started_at ?? "";
    return tb.localeCompare(ta);
  });

  return {
    active,
    failed,
    recent: sorted.slice(0, 6),
    counts: {
      total: jobs.length,
      running,
      failed: failed.length,
      succeeded,
      queued,
    },
  };
}

export function jobDurationText(startedAt?: string | null, completedAt?: string | null): string | null {
  if (!startedAt) return null;
  const t0 = new Date(startedAt).getTime();
  if (Number.isNaN(t0)) return null;

  const t1 = completedAt ? new Date(completedAt).getTime() : Date.now();
  if (Number.isNaN(t1)) return null;

  const diffMs = Math.max(0, t1 - t0);
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s`;
  const diffMin = Math.floor(diffSec / 60);
  const remSec = diffSec % 60;
  return `${diffMin}m ${remSec}s`;
}
