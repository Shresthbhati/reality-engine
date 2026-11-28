"use client";

import { useState } from "react";
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
} from "lucide-react";

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
 * Backend job stages (apps/api/jobs.py `job.stage`) mapped onto the modal's
 * progress steps. An unmapped stage is not a failure — the run continues and
 * simply shows no step highlight.
 */
const STAGE_TO_STEP: Record<string, number> = {
  checking_reconstruction_backend: 1,
  resolving_evidence: 1,
  reconstructing: 2,
  compiling: 6,
  committing_version: 7,
};

function stepForStage(stage: string | null | undefined): number | null {
  if (!stage) return null;
  return STAGE_TO_STEP[stage] ?? null;
}

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
  const [activeStage, setActiveStage] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [completed, setCompleted] = useState(false);

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
    setError(null);
    setCompleted(false);
    setActiveStage(1);

    try {
      // Enqueue a real RECONSTRUCT_SESSION job for this session. The 200 is
      // an enqueue acknowledgement, not a result — the job id is polled below.
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

        if (job.stage && job.stage !== lastStage) {
          lastStage = job.stage;
          const step = stepForStage(job.stage);
          if (step) setActiveStage(step);
        }

        if (job.status === "completed") {
          setActiveStage(7);
          setCompleted(true);
          setRunning(false);
          onReconstructionSuccess?.();
          return;
        }

        if (job.status === "failed") {
          throw new Error(job.error || "Reconstruction job failed.");
        }

        pollCount++;
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
      setActiveStage(null);
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
              <h3 className="text-sm font-semibold text-white">Room Construction Pipeline</h3>
              <p className="text-[11px] font-mono text-neutral-400">
                Spatial World: <span className="text-[#00e5ff]">{worldId}</span>
              </p>
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
                const isFinished = completed;

                return (
                  <div
                    key={s.id}
                    className={`p-2.5 rounded-lg border transition-colors flex items-start gap-3 ${
                      isActive
                        ? "bg-[#182030] border-[#00e5ff]"
                        : isPast || isFinished
                        ? "bg-[#14161f] border-[#2ecc71]/40"
                        : "bg-[#14161f] border-[#1f222b]"
                    }`}
                  >
                    <div
                      className={`p-1.5 rounded shrink-0 mt-0.5 ${
                        isActive
                          ? "bg-[#00e5ff]/20 text-[#00e5ff]"
                          : isPast || isFinished
                          ? "bg-[#2ecc71]/20 text-[#2ecc71]"
                          : "bg-neutral-800 text-neutral-400"
                      }`}
                    >
                      {isActive ? (
                        <Loader2 className="w-4 h-4 animate-spin text-[#00e5ff]" />
                      ) : isPast || isFinished ? (
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
                            : isPast || isFinished
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

          {/* Success Banner */}
          {completed && (
            <div className="p-3 rounded-lg border border-[#2ecc71]/50 bg-[#12281a] text-[#2ecc71] flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span>Room construction finished. WorldIR compiled and persisted to WorldStore.</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 h-14 border-t border-[#1f222b] bg-[#12141a]">
          <span className="text-[11px] text-neutral-400 font-mono">
            {running ? "Processing reconstruction job..." : "Ready for execution"}
          </span>

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
