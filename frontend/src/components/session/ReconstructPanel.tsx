"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { RefreshCw, CheckCircle2, XCircle } from "lucide-react";
import { reconstructSession, getJob, isApiError } from "@/lib/api";
import type { JobDto } from "@/lib/api/types";

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
    pollRef.current = setInterval(async () => {
      try {
        const latest = await getJob(jobId);
        setJob(latest);
        if (latest.status === "completed" || latest.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
          if (latest.status === "completed" && worldId) {
            router.push(`/worlds/${worldId}`);
          }
        }
      } catch {
        // Transient polling failure -- keep the last known status, try again.
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

  const isRunning = job != null && job.status !== "completed" && job.status !== "failed";

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
        {isRunning ? `Reconstructing (${job?.stage ?? job?.status})…` : "Start Reconstruction"}
      </button>

      {job?.status === "completed" && (
        <p className="flex items-center gap-1.5 text-xs" style={{ color: "var(--success, #2ecc71)" }}>
          <CheckCircle2 className="w-3.5 h-3.5" />
          Reconstruction complete — opening World…
        </p>
      )}

      {job?.status === "failed" && (
        <p className="flex items-start gap-1.5 text-xs" style={{ color: "var(--error, #e57373)" }}>
          <XCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          <span>{job.error ?? "Reconstruction failed."}</span>
        </p>
      )}

      {error && (
        <p className="text-xs" style={{ color: "var(--error, #e57373)" }}>
          {error}
        </p>
      )}
    </div>
  );
}
