"use client";

import { useState, useRef, useCallback } from "react";
import {
  Workflow,
  X,
  Play,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Layers,
  Box,
  Compass,
  Camera,
  Loader2,
  Ban,
} from "lucide-react";
import {
  classifyReconstruction,
  isPersistedSuccess,
  cancelJob,
  type CancelJobResult,
} from "@/lib/api";

/** A session attached to this world, as offered in the selector. */
export interface ConstructionSessionOption {
  id: string;
  name: string;
}

interface RoomConstructionModalProps {
  worldId: string;
  /** Sessions attached to this world; the job is enqueued for the selected one. */
  sessions: ConstructionSessionOption[];
  /** Currently selected session; null when the world has none attached. */
  sessionId: string | null;
  onSelectSession?: (sessionId: string | null) => void;
  isOpen: boolean;
  onClose: () => void;
  onReconstructionSuccess?: () => void;
}

const POLL_INTERVAL_MS = 2000;
const MAX_POLLS = 60; // ~2 minutes; the backend job is the source of truth

/**
 * The reconstruction outcome is decided by ONE canonical helper,
 * `classifyReconstruction` in lib/api/reconstruction.ts, so the UI and its
 * tests can never disagree about what counts as a persisted success.
 */
type ReconstructionOutcome = {
  status: "succeeded" | "partial";
  versionId: string;
  degraded: string[];
};

/**
 * Backend job stages (apps/api/jobs.py `job.stage`) mapped onto the modal's
 * progress steps. An unmapped stage is not a failure â€” the run continues and
 * simply shows no step highlight.
 */
const STAGE_TO_STEP: Record<string, number> = {
  checking_reconstruction_backend: 1,
  resolving_evidence: 1,
  evidence: 1,
  reconstructing: 2,
  sfm: 2,
  scale: 3,
  depth: 4,
  planes: 5,
  compiling: 6,
  worldir: 6,
  committing_version: 7,
  worldstore: 7,
};

function stepForStage(stage: string | null | undefined): number | null {
  if (!stage) return null;
  return STAGE_TO_STEP[stage] ?? null;
}

export type ReconstructionJobStatus =
  | "idle"
  | "queued"
  | "running"
  | "partial"
  | "succeeded"
  | "failed"
  | "cancelled";

export default function RoomConstructionModal({
  worldId,
  sessions,
  sessionId,
  onSelectSession,
  isOpen,
  onClose,
  onReconstructionSuccess,
}: RoomConstructionModalProps) {
  const [running, setRunning] = useState(false);
  const [jobStatus, setJobStatus] = useState<ReconstructionJobStatus>("idle");
  const [activeStage, setActiveStage] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<ReconstructionOutcome | null>(null);

  // --- Cancellation (P0) ------------------------------------------------
  // `jobIdRef` is the only handle the cancel button needs; `cancelState` is
  // what the UI renders. Crucially, `cancelState.kind` is NEVER set to
  // "cancelled" by the click -- only by a poll that actually observed the
  // backend report it.
  //
  // EVERY hook must stay ABOVE the `if (!isOpen) return null` early return.
  // A `useCallback` placed after it runs only while the modal is open, which
  // is a React hooks-order violation ("Rendered more hooks than during the
  // previous render") and crashes the modal exactly when it is opened.
  const jobIdRef = useRef<string | null>(null);
  const [cancelState, setCancelState] = useState<
    | { kind: "idle" }
    // Request sent, awaiting the backend's answer.
    | { kind: "requesting" }
    // 202: the worker was asked to stop and is winding down. NOT cancelled.
    | { kind: "requested"; detail: string }
    // 409/404/transport: the request was refused. The job is untouched.
    | { kind: "refused"; detail: string }
  >({ kind: "idle" });
  // Guards against duplicate cancels: a second click while one is in flight,
  // or after one has been accepted, is a no-op rather than a second request.
  const cancelInFlightRef = useRef(false);

  /**
   * Request cancellation of the running job.
   *
   * The click NEVER sets a cancelled state. It records that a request is in
   * flight, and then only "requested" (202/accepted) or "refused" (409/404/
   * transport) depending on what the backend answered. The modal reaches a
   * cancelled display solely because the polling loop observed
   * `status === "cancelled"` from `GET /api/jobs/{id}`.
   */
  const handleCancel = useCallback(async () => {
    const jobId = jobIdRef.current;
    if (!jobId) return;
    // Duplicate-request guard: a second click while one is in flight, or
    // after one was accepted, must not fire another cancel.
    if (cancelInFlightRef.current) return;
    if (cancelState.kind === "requesting" || cancelState.kind === "requested") return;

    cancelInFlightRef.current = true;
    setCancelState({ kind: "requesting" });
    try {
      const result: CancelJobResult = await cancelJob(jobId);
      if (result.outcome === "cancelling") {
        // 202: the worker was asked to stop. It is NOT cancelled yet.
        setCancelState({
          kind: "requested",
          detail:
            "Cancellation requested. The job is still running server-side and will " +
            "stop at its next stage boundary.",
        });
      } else if (result.outcome === "cancelled") {
        // 200: a QUEUED job stops immediately and is already terminal.
        setCancelState({ kind: "requested", detail: "Cancellation accepted." });
      } else {
        // 409 (already finished) / 404 (unknown) / transport failure.
        // The job is untouched; the poll loop will show the real state.
        setCancelState({
          kind: "refused",
          detail: result.reason ?? "The cancellation request was refused.",
        });
      }
    } finally {
      cancelInFlightRef.current = false;
    }
  }, [cancelState.kind]);

  if (!isOpen) return null;

  const STAGES = [
    {
      id: 1,
      title: "Evidence Ingestion & Validation",
      desc: "Parse camera telemetry, verify image integrity, check timestamp synchronization.",
      icon: Camera,
    },
    {
      id: 2,
      title: "SfM Multi-View Pose Estimation",
      desc: "Extract features, pair-wise matching, solve camera intrinsics & extrinsics.",
      icon: Compass,
    },
    {
      id: 3,
      title: "Metric Scale Calibration",
      desc: "Calibrate Euclidean metric scale against sensor baseline references.",
      icon: Layers,
    },
    {
      id: 4,
      title: "Monocular Depth Alignment & Fusion",
      desc: "Metricize monocular depth predictions using SfM sparse point cloud anchors.",
      icon: Cpu,
    },
    {
      id: 5,
      title: "Structural Plane Promotion",
      desc: "Extract planar primitives (floor, ceiling, bounding walls) from 3D points.",
      icon: Box,
    },
    {
      id: 6,
      title: "WorldIR Canonical Compilation",
      desc: "Compile entities, bounding boxes, level-of-detail meshes into intermediate representation.",
      icon: Workflow,
    },
    {
      id: 7,
      title: "WorldStore Version Snapshot",
      desc: "Commit immutable version lineage to WorldStore ledger with provenance audit.",
      icon: CheckCircle2,
    },
  ];

  const handleRunPipeline = async () => {
    if (!sessionId) {
      setError(
        "This world has no attached capture session to reconstruct. Attach a session with evidence first."
      );
      return;
    }

    setRunning(true);
    setJobStatus("queued");
    setError(null);
    setOutcome(null);
    setActiveStage(1);

    try {
      // Enqueue a real RECONSTRUCT_SESSION job for this session. The 200 is
      // an enqueue acknowledgement, not a result â€” the job id is polled below.
      const res = await fetch(
        `/api/sessions/${encodeURIComponent(sessionId)}/reconstruct`,
        { method: "POST" }
      );

      const data = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(
          data?.detail ??
            data?.error ??
            `Reconstruction could not be enqueued (HTTP ${res.status}).`
        );
      }

      const jobId = data?.job_id;
      if (!jobId) {
        throw new Error("API accepted the request but returned no job id to track.");
      }
      jobIdRef.current = jobId;

      let pollCount = 0;
      let lastStage: string | null = null;
      while (pollCount < MAX_POLLS) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        const jobRes = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);

        if (!jobRes.ok) {
          // A transient read failure is not a job failure; keep polling until
          // the budget is spent, then report honestly.
          pollCount++;
          continue;
        }

        const job = await jobRes.json();

        if (job.status === "running" || job.status === "queued") {
          setJobStatus(job.status);
        }

        if (job.stage && job.stage !== lastStage) {
          lastStage = job.stage;
          const step = stepForStage(job.stage);
          if (step) setActiveStage(step);
        }

        // ONE canonical decision, shared with the tests. A job is a persisted
        // result only when it is succeeded/partial AND carries the version id
        // the worker actually committed. A cancelled or failed run is never
        // promoted, and an unknown status is never read as success.
        const verdict = classifyReconstruction(job);

        if (verdict.kind === "pending") {
          pollCount++;
          continue;
        }

        if (isPersistedSuccess(verdict)) {
          setActiveStage(7);
          setJobStatus(verdict.kind === "partial" ? "partial" : "succeeded");
          setOutcome({
            status: verdict.kind === "partial" ? "partial" : "succeeded",
            versionId: verdict.versionId as string,
            degraded: verdict.degraded,
          });
          setCancelState({ kind: "idle" });
          setRunning(false);
          jobIdRef.current = null;
          onReconstructionSuccess?.();
          return;
        }

        // Terminal but not a persisted success: failed, cancelled, or a
        // terminal status that cannot be proven. `verdict.reason` says which.
        setJobStatus(job.status === "cancelled" ? "cancelled" : "failed");
        setActiveStage(null);
        setRunning(false);
        jobIdRef.current = null;
        // The cancel state is only resolved to its terminal form now, because
        // the BACKEND has now actually reported the terminal status.
        setCancelState(
          job.status === "cancelled"
            ? { kind: "requested", detail: "Cancelled by operator." }
            : { kind: "idle" },
        );
        setError(verdict.reason);
        return;
      }

      // Budget exhausted: the job is still running server-side. Say so instead
      // of reporting a success that was never observed.
      setRunning(false);
      setActiveStage(null);
      setError(
        `Job ${jobId} was still ${lastStage ?? "queued"} after ` +
          `${Math.round((MAX_POLLS * POLL_INTERVAL_MS) / 1000)}s of polling. ` +
          "The reconstruction may still finish server-side; re-open the world to see its version."
      );
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message || "Reconstruction failed for an unknown reason.");
      setRunning(false);
      setJobStatus("failed");
      setActiveStage(null);
      jobIdRef.current = null;
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm select-none"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl max-h-[90vh] rounded-lg flex flex-col overflow-hidden bg-[#0e1013] border border-[#1f222b] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 h-14 border-b border-[#1f222b] bg-[#12141a]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-md flex items-center justify-center bg-[#00e5ff]/10 text-[#00e5ff] border border-[#00e5ff]/20">
              <Workflow className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold text-white">Room Construction Pipeline</h3>
                {jobStatus !== "idle" && (
                  <span
                    className={`text-[9px] font-mono font-bold uppercase px-2 py-0.5 rounded border ${
                      jobStatus === "succeeded"
                        ? "bg-[#2ecc71]/20 text-[#2ecc71] border-[#2ecc71]/40"
                        : jobStatus === "partial"
                        ? "bg-[#e6a23c]/20 text-[#e6a23c] border-[#e6a23c]/40"
                        : jobStatus === "cancelled"
                        ? "bg-neutral-700/40 text-neutral-300 border-neutral-600"
                        : jobStatus === "running"
                        ? "bg-[#00e5ff]/20 text-[#00e5ff] border-[#00e5ff]/40 animate-pulse"
                        : jobStatus === "queued"
                        ? "bg-amber-500/20 text-amber-400 border-amber-500/40"
                        : jobStatus === "failed"
                        ? "bg-red-500/20 text-red-400 border-red-500/40"
                        : "bg-neutral-800 text-neutral-400 border-neutral-700"
                    }`}
                    data-testid="job-status-badge"
                  >
                    {jobStatus}
                  </span>
                )}

                {/* Cancel (P0). The label never says "Cancelled" -- that state
                    is only reachable once the backend reports it via polling. */}
                {running && cancelState.kind === "idle" && (
                  <button
                    type="button"
                    data-testid="cancel-job-button"
                    onClick={handleCancel}
                    className="flex items-center gap-1 text-[10px] font-mono uppercase px-2 py-0.5 rounded border border-red-500/40 bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
                  >
                    <Ban className="w-3 h-3" />
                    Cancel
                  </button>
                )}

                {cancelState.kind === "requesting" && (
                  <span className="flex items-center gap-1 text-[10px] font-mono uppercase px-2 py-0.5 rounded border border-amber-500/40 bg-amber-500/10 text-amber-400">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Requestingâ€¦
                  </span>
                )}

                {cancelState.kind === "requested" && (
                  <span
                    data-testid="cancel-requested-badge"
                    className="text-[10px] font-mono uppercase px-2 py-0.5 rounded border border-amber-500/40 bg-amber-500/10 text-amber-400"
                  >
                    Cancellation requested
                  </span>
                )}

                {cancelState.kind === "refused" && (
                  <span
                    data-testid="cancel-refused-badge"
                    className="text-[10px] font-mono uppercase px-2 py-0.5 rounded border border-red-500/40 bg-red-500/10 text-red-400"
                    title={cancelState.detail}
                  >
                    Cancel refused
                  </span>
                )}
              </div>
              <p className="text-[11px] font-mono text-neutral-400">
                Spatial World: <span className="text-[#00e5ff]">{worldId}</span>
              </p>
              {/* The honest status line for a pending cancellation. */}
              {cancelState.kind === "requested" && (
                <p className="text-[11px] text-amber-400 mt-1">{cancelState.detail}</p>
              )}
              {cancelState.kind === "refused" && (
                <p className="text-[11px] text-red-400 mt-1">
                  {cancelState.detail} The job is unchanged; its actual state is shown above.
                </p>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-md text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 min-h-0 overflow-y-auto p-5 space-y-5 text-xs">
          {/* Evidence source. The reconstruction job is per-Session, so this
              is the only input that actually changes what runs. */}
          <div className="p-3 rounded-lg border border-[#1f222b] bg-[#14161f]">
            <label className="text-[10px] text-neutral-400 uppercase tracking-wider block mb-1">
              Capture Session
            </label>
            <select
              value={sessionId ?? ""}
              onChange={(e) => onSelectSession?.(e.target.value || null)}
              disabled={running || sessions.length === 0}
              className="w-full h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white focus:border-[#00e5ff] focus:outline-none"
            >
              {sessions.length === 0 && (
                <option value="">No sessions attached to this world</option>
              )}
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name || s.id}
                </option>
              ))}
            </select>
            <p className="text-[10px] text-neutral-500 mt-1.5 leading-relaxed">
              Reconstruction runs on this session&apos;s stored evidence using the
              backend&apos;s own pipeline configuration (SfM backend, depth model,
              acceleration).
            </p>
          </div>

          {/* Canonical 7 Pipeline Stages */}
          <div className="space-y-2">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
              Canonical Pipeline Stages
            </h4>
            <div className="space-y-1.5">
              {STAGES.map((s) => {
                const Icon = s.icon;
                const isActive = activeStage === s.id;
                const isPast = activeStage !== null && activeStage > s.id;
                // Stages are only all-complete when the run actually
                // succeeded; a partial run keeps its unreached stages
                // showing as pending instead of claiming the pipeline
                // finished.
                const isFinished = outcome?.status === "succeeded";
                const isDone = isPast || isFinished;

                return (
                  <div
                    key={s.id}
                    className={`p-2.5 rounded-lg border transition-colors flex items-start gap-3 ${
                      isActive
                        ? "bg-[#182030] border-[#00e5ff]"
                        : isDone
                        ? "bg-[#14161f] border-[#2ecc71]/40"
                        : "bg-[#14161f] border-[#1f222b]"
                    }`}
                  >
                    <div
                      className={`p-1.5 rounded shrink-0 mt-0.5 ${
                        isActive
                          ? "bg-[#00e5ff]/20 text-[#00e5ff]"
                          : isDone
                          ? "bg-[#2ecc71]/20 text-[#2ecc71]"
                          : "bg-neutral-800 text-neutral-400"
                      }`}
                    >
                      {isActive ? (
                        <Loader2 className="w-4 h-4 animate-spin text-[#00e5ff]" />
                      ) : isDone ? (
                        <CheckCircle2 className="w-4 h-4 text-[#2ecc71]" />
                      ) : (
                        <Icon className="w-4 h-4" />
                      )}
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-white">{s.title}</span>
                        <span
                          className={`text-[10px] font-mono px-1.5 py-0.2 rounded ${
                            isActive
                              ? "bg-[#00e5ff]/20 text-[#00e5ff]"
                              : isPast || isFinished
                              ? "bg-[#2ecc71]/20 text-[#2ecc71]"
                              : "bg-neutral-800 text-neutral-500"
                          }`}
                        >
                          {isActive
                            ? "RUNNING"
                            : isDone
                            ? "COMPLETE"
                            : "PENDING"}
                        </span>
                      </div>
                      <p className="text-[11px] text-neutral-400 mt-0.5">{s.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Honest Error Status */}
          {error && (
            <div className="p-3 rounded-lg border border-[#e74c3c]/50 bg-[#2a1315] text-[#ff6b6b] flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="font-semibold text-xs">Reconstruction did not complete</div>
                <div className="text-[11px] leading-relaxed text-neutral-300 font-mono">
                  {error}
                </div>
              </div>
            </div>
          )}

          {/* Outcome: only a committed WorldStore version is reported */}
          {outcome && (
            <div
              className={`p-3 rounded-lg border flex items-start gap-2.5 ${
                outcome.status === "succeeded"
                  ? "border-[#2ecc71]/50 bg-[#12281a] text-[#2ecc71]"
                  : "border-[#e6a23c]/50 bg-[#2a2413] text-[#e6a23c]"
              }`}
            >
              {outcome.status === "succeeded" ? (
                <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
              ) : (
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              )}
              <div className="space-y-1 min-w-0">
                <div className="font-semibold text-xs">
                  {outcome.status === "succeeded"
                    ? "Room construction finished"
                    : "Version adopted with degraded stages"}
                </div>
                <div className="text-[11px] leading-relaxed font-mono break-all">
                  WorldStore version {outcome.versionId}
                </div>
                {outcome.degraded.length > 0 && (
                  <ul className="text-[11px] leading-relaxed list-disc list-inside">
                    {outcome.degraded.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 h-14 border-t border-[#1f222b] bg-[#12141a]">
          <div className="flex items-center gap-2 font-mono text-[11px]">
            <span className="text-neutral-500">Status:</span>
            <span
              className={`font-semibold uppercase ${
                jobStatus === "succeeded"
                  ? "text-[#2ecc71]"
                  : jobStatus === "partial"
                  ? "text-[#e6a23c]"
                  : jobStatus === "running"
                  ? "text-[#00e5ff]"
                  : jobStatus === "queued"
                  ? "text-amber-400"
                  : jobStatus === "failed"
                  ? "text-red-400"
                  : jobStatus === "cancelled"
                  ? "text-neutral-400"
                  : "text-neutral-500"
              }`}
            >
              {jobStatus === "idle" ? "READY" : jobStatus}
            </span>
            {outcome?.versionId && (
              <span className="text-neutral-400">Â· {outcome.versionId}</span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              disabled={running}
              className="px-3.5 py-1.5 rounded text-xs text-neutral-400 hover:text-white transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleRunPipeline}
              disabled={running}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded text-xs font-semibold bg-[#00e5ff] text-black hover:bg-[#33ebff] transition-colors disabled:opacity-50 cursor-pointer"
            >
              {running ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Executing...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Start Reconstruction</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
