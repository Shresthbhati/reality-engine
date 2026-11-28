/**
 * THE canonical reconstruction-outcome rule.
 *
 * This is the single place that decides whether a job represents a persisted
 * successful reconstruction. Screens, tests and harnesses all call it, so the
 * rule cannot drift between the UI and its tests.
 *
 * ---------------------------------------------------------------------------
 * RESOURCE CONTRACT MAP (the frontend side of the boundary)
 * ---------------------------------------------------------------------------
 * The backend endpoints live in `apps/api`. Every row below is enforced by a
 * harness under `frontend/tests/contract`.
 *
 *  Resource        Frontend route            Backend endpoint                Success proof
 *  --------------- ------------------------- ------------------------------- --------------------------
 *  Worlds list     api/worlds (GET)          GET  /api/worlds                200 { items: [...] }
 *  World create    api/worlds (POST)         POST /api/worlds (201)          201 + `id`
 *  World detail    (client direct)           GET  /api/worlds/{id}           200 { id, ... }
 *  Sessions list   api/sessions (GET)        GET  /api/sessions              200 { items: [...] }
 *  Reconstruct     api/sessions/{id}/        POST /api/sessions/{id}/        200 + `job_id`
 *                   reconstruct               reconstruct
 *  Job poll        api/jobs/{id} (GET)       GET  /api/jobs/{id}             200 { status, payload }
 *  Job cancel      api/jobs/{id}/cancel      POST /api/jobs/{id}/cancel     200/202 + `id`
 *  Evidence list   (client direct)           GET  /api/evidence              200 { items: [...] }
 *  Evidence delete api/evidence/{id} (DELETE) DELETE /api/evidence/{id}      204 (empty body)
 *  Upload          api/uploads (POST)        POST /api/uploads (201)         201 + `evidence_id`
 *  WorldIR         api/worlds/{id}/worldir   GET  /api/worlds/{id}/worldir   200 WorldIR, else 404
 *  WorldStore ver. api/worlds/{id}/versions  GET  /api/worlds/{id}/versions  200 { items: [...] }
 *  Diff            api/worlds/{id}/diff      GET  /api/worlds/{id}/diff      200 WorldDiff
 *  Points          api/worlds/{id}/points    GET  /api/worlds/{id}/points    200 octet-stream
 *  Cameras         api/worlds/{id}/cameras   GET  /api/worlds/{id}/cameras   200 { cameras: [...] }
 *  Commit          api/worlds/{id}/commit    POST /api/worlds/{id}/commit    200 + `version_id`
 *
 * Failure semantics, everywhere:
 *   4xx/5xx from the backend  -> forwarded with the REAL status and body
 *   transport failure          -> 503 with `available: false` (never 404)
 *   2xx without its proof key  -> 502, never reported as a success
 *
 * Cache invalidation: `useApi` clears on a dependency change and on failure, and
 * aborts the superseded request, so a world switch can never show the previous
 * world's data. See `hooks.ts`.
 * ---------------------------------------------------------------------------
 *
 * The rule (mirroring apps/api/jobs.py, which grades SUCCEEDED only when the
 * output validates and a version is actually adopted):
 *
 *   succeeded + version  -> "success"          fully clean, version adopted
 *   partial   + version  -> "partial"          version adopted, run degraded
 *   succeeded + no ver.  -> "failure"          claims success it cannot prove
 *   partial   + no ver.  -> "failure"          same
 *   failed    + version  -> "failure"          a version id never rescues a
 *   cancelled + version  -> "failure"            failed/terminal run
 *   queued    + version  -> "pending"          not finished; never "success"
 *   running   + version  -> "pending"
 *   (no payload at all)  -> "failure"
 *
 * A version id is necessary but NOT sufficient. `failed + version_id` is
 * incoherent lifecycle data and is rejected rather than fast-forwarded.
 */
import type { JobDto } from "./types";

export type ReconstructionOutcomeKind =
  /** Fully clean run; the worker adopted a version. */
  | "success"
  /** Version adopted but the run was degraded (partial registration, skips). */
  | "partial"
  /** Terminal and unsuccessful, or terminal-but-unprovable. */
  | "failure"
  /** Not finished yet (queued/running). */
  | "pending";

export interface ReconstructionOutcome {
  kind: ReconstructionOutcomeKind;
  /** The committed WorldStore version id, when one is proven. */
  versionId: string | null;
  /** Backend-reported degradation reasons, only meaningful for "partial". */
  degraded: string[];
  /** Why this was not a success, for display and for tests. */
  reason: string;
}

/** A job shape loose enough to accept a raw API payload as well as a JobDto. */
export interface ReconstructionJobLike {
  status: string;
  payload?: Record<string, unknown> | null;
  error?: string | null;
}

/**
 * Classify a reconstruction job. Never throws; unknown or malformed input is
 * "failure", never a success.
 */
export function classifyReconstruction(job: ReconstructionJobLike | null | undefined): ReconstructionOutcome {
  if (!job || typeof job.status !== "string") {
    return {
      kind: "failure",
      versionId: null,
      degraded: [],
      reason: "No job data was returned by the API.",
    };
  }

  const status = job.status;
  const payload = (job.payload ?? {}) as Record<string, unknown>;
  const versionId = typeof payload.version_id === "string" && payload.version_id.length > 0
    ? payload.version_id
    : null;
  const degraded = Array.isArray(payload.degraded)
    ? payload.degraded.filter((d): d is string => typeof d === "string")
    : [];

  if (status === "queued" || status === "running") {
    return {
      kind: "pending",
      versionId,
      degraded,
      reason: "The reconstruction is still running.",
    };
  }

  if (status === "failed") {
    return {
      kind: "failure",
      versionId: null,
      degraded,
      reason: job.error ?? "The reconstruction job failed.",
    };
  }

  if (status === "cancelled") {
    return {
      kind: "failure",
      versionId: null,
      degraded,
      reason: "The reconstruction was cancelled.",
    };
  }

  // From here the status is a terminal-but-claimant: succeeded or partial.
  // It may only be reported as a persisted result with a real version id.
  if (status !== "succeeded" && status !== "partial") {
    return {
      kind: "failure",
      versionId: null,
      degraded,
      reason: `Unrecognised job status '${status}'; no outcome can be claimed.`,
    };
  }

  if (!versionId) {
    return {
      kind: "failure",
      versionId: null,
      degraded,
      reason:
        `The job reported '${status}' but carries no committed WorldStore version id. ` +
        "Nothing was adopted, so this is not a successful reconstruction.",
    };
  }

  return {
    kind: status === "partial" ? "partial" : "success",
    versionId,
    degraded,
    reason: status === "partial" ? "Adopted with degraded stages." : "Reconstruction complete.",
  };
}

/** Convenience: did this job persist a usable result? */
export function isPersistedSuccess(outcome: ReconstructionOutcome): boolean {
  return outcome.kind === "success" || outcome.kind === "partial";
}

/** Convenience for a raw JobDto. */
export function classifyJob(job: JobDto | null | undefined): ReconstructionOutcome {
  return classifyReconstruction(job);
}
