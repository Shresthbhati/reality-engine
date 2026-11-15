"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { RefreshCw, CheckCircle2, XCircle } from "lucide-react";
import { reconstructSession, getJob, isApiError } from "@/lib/api";
import type { JobDto } from "@/lib/api/types";
import { toJobStatus, isTerminal, isUsableResult, type JobStatus } from "@/lib/api/job-status";

/**
 * Starts a real reconstruction job for this Session (POST
 * /sessions/{id}/reconstruct), polls its real status (GET /jobs/{id})
 * until it finishes, and routes to the attached World on success --
 * no fabricated progress bar, no invented percentages.
 */
export default function ReconstructPanel({
  sessionId,
  worldId,
}: {
  sessionId: string;
  worldId: string | null;
}) {
  const router = useRouter();
  const [job, setJob] = useState<JobDto | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  function pollJob(jobId: string) {
    if (pollRef.current) clearInterval(pollRef.current);
    let failures = 0;
    pollRef.current = setInterval(async () => {
      try {
        const latest = await getJob(jobId);
        failures = 0;
        setError(null);
        setJob(latest);
        const st = toJobStatus(latest.status);
        if (isTerminal(st)) {
          if (pollRef.current) clearInterval(pollRef.current);
          if (isUsableResult(st) && worldId) {
            router.push(`/worlds/${worldId}`);
          }
        }
      } catch (err) {
        // A few failed polls are transient; a sustained outage must not leave "Reconstructing…" on screen forever.
        failures += 1;
        if (failures >= 5) {
          if (pollRef.current) clearInterval(pollRef.current);
          setError(
            `Lost contact with the API while the job was running (${isApiError(err) ? err.describe() : "request failed"}). ` +
              "The reconstruction may still be running; reload to check its status.",
          );
          setJob(null);
        }
      }
    }, 2000);
  }

  async function handleStart() {
    setError(null);
    setStarting(true);
    try {
      const started = await reconstructSession(sessionId);
      const initial = await getJob(started.job_id);
      setJob(initial);
      pollJob(started.job_id);
    } catch (err) {
      setError(isApiError(err) ? err.describe() : "Failed to start reconstruction.");
    } finally {
      setStarting(false);
    }
  }

  const status = job ? toJobStatus(job.status) : null;
  const isRunning = status === "queued" || status === "running";

  function renderOutcome(j: NonNullable<typeof job>, st: JobStatus) {
    switch (st) {
      case "queued":
      case "running":
        return null;
      case "succeeded":
        return (
          <p className="flex items-center gap-1.5 text-xs" style={{ color: "var(--success)" }}>
            <CheckCircle2 className="w-3.5 h-3.5" />
            Reconstruction complete — opening World…
          </p>
        );
      case "partial":
        return j.payload?.adopted === false ? (
          <p className="flex items-center gap-1.5 text-xs" style={{ color: "var(--warning)" }}>
            <XCircle className="w-3.5 h-3.5" />
            This rebuild was not adopted — the World keeps its current version. Opening World…
          </p>
        ) : (
          <p className="flex items-center gap-1.5 text-xs" style={{ color: "var(--warning)" }}>
            <CheckCircle2 className="w-3.5 h-3.5" />
            Reconstruction partially complete — opening World…
          </p>
        );
      case "cancelled":
        return (
          <p className="flex items-center gap-1.5 text-xs" style={{ color: "var(--text-tertiary)" }}>
            <XCircle className="w-3.5 h-3.5" />
            Reconstruction cancelled.
          </p>
        );
      case "failed":
        return (
          <p className="flex items-start gap-1.5 text-xs" style={{ color: "var(--error)" }}>
            <XCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <span>{j.error ?? "Reconstruction failed."}</span>
          </p>
        );
      case "unknown":
        return (
          <p className="flex items-center gap-1.5 text-xs" style={{ color: "var(--text-tertiary)" }}>
            <XCircle className="w-3.5 h-3.5" />
            Status unknown — refresh or check the job log
          </p>
        );
    }
  }

  return (
    <div className="flex flex-col gap-2">
      {!worldId && (
        <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          Attach this Session to a World before reconstructing — a compiled version needs
          somewhere to be committed.
        </p>
      )}

      <button
        type="button"
        onClick={handleStart}
        disabled={starting || isRunning || !worldId}
        className="flex items-center justify-center gap-2 h-9 px-4 rounded-md text-sm font-medium self-start disabled:opacity-40 transition-colors"
        style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
      >
        <RefreshCw className={`w-3.5 h-3.5 ${starting || isRunning ? "animate-spin" : ""}`} />
        {isRunning ? `Reconstructing (${job?.stage ?? status})…` : "Start Reconstruction"}
      </button>

      {job && status !== null && renderOutcome(job, status)}

      {error && (
        <p className="text-xs" style={{ color: "var(--error)" }}>
          {error}
        </p>
      )}
    </div>
  );
}
