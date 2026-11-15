/** Pipeline stage attempts for a job (mirrors apps/api/routes_jobs.py GET /{job_id}/stages). */
import { apiGet } from "./client.ts";

export type StageStatus = "succeeded" | "degraded" | "skipped" | "failed" | "unknown";

export interface JobStage {
  name: string;
  status: StageStatus;
  attempt: number;
  durationS: number | null;
  warnings: string[];
  error: string | null;
}

const KNOWN: ReadonlySet<string> = new Set(["succeeded", "degraded", "skipped", "failed"]);

export function parseJobStages(body: unknown): JobStage[] {
  const items = (body as { items?: unknown } | null)?.items;
  if (!Array.isArray(items)) throw new Error("job stages: unexpected response shape");
  return items.map((r: Record<string, unknown>) => ({
    name: String(r.name),
    status: typeof r.status === "string" && KNOWN.has(r.status) ? (r.status as StageStatus) : "unknown",
    attempt: Number(r.attempt),
    durationS: typeof r.duration_s === "number" ? r.duration_s : null,
    warnings: Array.isArray(r.warnings) ? r.warnings.map(String) : [],
    error: typeof r.error === "string" ? r.error : null,
  }));
}

export async function fetchJobStages(jobId: string): Promise<JobStage[]> {
  return parseJobStages(await apiGet<unknown>(`/api/jobs/${encodeURIComponent(jobId)}/stages`));
}
