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
