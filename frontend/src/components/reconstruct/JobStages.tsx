"use client";

import React, { useEffect, useState } from "react";
import { fetchJobStages, type JobStage } from "@/lib/api/job-stages";

/** The pipeline stages the worker actually ran for this job (read from the server, never inferred): name, outcome, duration, retries,
 * warnings and the failure text. Polls while the job is running; an empty list means no stage has recorded a result yet. */
export default function JobStages({ jobId, active }: { jobId: string; active: boolean }) {
  const [stages, setStages] = useState<JobStage[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      fetchJobStages(jobId)
        .then((s) => {
          if (!cancelled) {
            setStages(s);
            setError(null);
          }
        })
        .catch((e) => {
          if (!cancelled) setError(e instanceof Error ? e.message : "Could not read stage progress.");
        });
    load();
    const t = active ? setInterval(load, 3000) : null;
    return () => {
      cancelled = true;
      if (t) clearInterval(t);
    };
  }, [jobId, active]);

  if (error) return <p className="text-[11px] text-warning" role="alert">Stage progress unavailable: {error}</p>;
  if (!stages) return null;
  if (stages.length === 0) return <p className="text-[11px] text-fg-muted">No stage has finished yet.</p>;
  return (
    <ol className="flex flex-col gap-0.5 text-[11px]" aria-label="Pipeline stages" data-testid="job-stages">
      {stages.map((s, i) => (
        <li key={`${s.name}-${s.attempt}-${i}`} className="flex flex-wrap gap-x-2">
          <span className="font-medium">{s.name}</span>
          <span className={s.status === "failed" ? "text-error" : s.status === "degraded" ? "text-warning" : "text-fg-muted"}>{s.status}</span>
          {s.durationS !== null && <span className="text-fg-muted">{s.durationS.toFixed(1)}s</span>}
          {s.attempt > 1 && <span className="text-fg-muted">attempt {s.attempt}</span>}
          {s.warnings.length > 0 && <span className="text-warning">{s.warnings.join("; ")}</span>}
          {s.error && <span className="text-error">{s.error}</span>}
        </li>
      ))}
    </ol>
  );
}
