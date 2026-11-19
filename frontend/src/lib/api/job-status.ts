/** Single source for reading a job status string from the API (mirrors apps/api/jobs.py JOB_*). */
export type JobStatus = "queued" | "running" | "partial" | "succeeded" | "failed" | "cancelled" | "unknown";

const KNOWN: ReadonlySet<string> = new Set(["queued", "running", "partial", "succeeded", "failed", "cancelled"]);

export function toJobStatus(raw: unknown): JobStatus {
  if (typeof raw !== "string") return "unknown";
  const s = raw.toLowerCase();
  return KNOWN.has(s) ? (s as JobStatus) : "unknown";
}

export function isTerminal(s: JobStatus): boolean {
  return s === "partial" || s === "succeeded" || s === "failed" || s === "cancelled";
}

export function isUsableResult(s: JobStatus): boolean {
  return s === "succeeded" || s === "partial";
}

/**
 * What a polled job means for the user. Mirrors `derive_state` in apps/api/reconstruction_state.py: the five outcomes a
 * person must be able to tell apart, plus `unrecognised` for a status this client does not know -- which is never
 * treated as "still running" (the poll would spin) nor as success.
 */
export type JobOutcome = "in_progress" | "succeeded" | "partial" | "needs_more_evidence" | "failed" | "cancelled" | "unrecognised";

export function jobOutcome(job: { status?: unknown; payload?: unknown } | null | undefined): JobOutcome {
  const s = toJobStatus(job?.status);
  if (s === "queued" || s === "running") return "in_progress";
  if (s === "unknown") return "unrecognised";
  if (s === "failed") {
    const kind = (job?.payload as { failure_kind?: unknown } | null | undefined)?.failure_kind;
    return kind === "insufficient_evidence" ? "needs_more_evidence" : "failed";
  }
  return s;
}
