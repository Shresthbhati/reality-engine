/**
 * Reconstruction status derived from real signals only.
 *
 * The five states the product must distinguish are computed from things the
 * API actually reports -- the version's compile report, the job record, and
 * whether a WorldIR version exists at all. Nothing is inferred optimistically:
 * a world with a version but no room entities is reported as PARTIAL, never as
 * a completed reconstruction, because the engine compiled only part of what a
 * building needs.
 */
import type { InteriorModel } from "./model";
import type { PipelineReport } from "@/types/worldir";

export type ReconstructionState =
  | "processing"
  | "partial"
  | "complete"
  | "failed"
  | "insufficient_evidence"
  | "not_reconstructed";

export interface ReconstructionStatus {
  state: ReconstructionState;
  title: string;
  detail: string;
  /** Per-stage facts when the version carries a pipeline report. */
  stages: Array<{ name: string; status: string; note?: string }>;
  /** Counts backing a PARTIAL verdict, for the UI to show. */
  missing: string[];
}

interface JobLike {
  status?: string | null;
  stage?: string | null;
  error?: string | null;
}

/** Stage names the compile pipeline reports, in pipeline order. */
const STAGE_ORDER = [
  "detail",
  "depth",
  "perception",
  "fusion",
  "dense_mvs",
  "mesh",
] as const;

function stageList(report: PipelineReport | null): ReconstructionStatus["stages"] {
  const stages = report?.stages as Record<string, unknown> | undefined;
  if (!stages) return [];
  const out: ReconstructionStatus["stages"] = [];
  for (const name of Object.keys(stages)) {
    const value = stages[name];
    if (value && typeof value === "object") {
      const v = value as { status?: string; note?: string };
      out.push({ name, status: v.status ?? "unknown", note: v.note });
    } else {
      out.push({ name, status: String(value) });
    }
  }
  // Present pipeline order first, anything else after, stable.
  out.sort((a, b) => {
    const ia = STAGE_ORDER.indexOf(a.name as (typeof STAGE_ORDER)[number]);
    const ib = STAGE_ORDER.indexOf(b.name as (typeof STAGE_ORDER)[number]);
    if (ia === -1 && ib === -1) return a.name.localeCompare(b.name);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
  return out;
}

/**
 * Classify a world's reconstruction.
 *
 * `model` is null when no WorldIR version exists yet; `job` is the most
 * recent reconstruct job when one is known.
 */
export function classifyReconstruction(
  model: InteriorModel | null,
  report: PipelineReport | null,
  job: JobLike | null,
): ReconstructionStatus {
  const stages = stageList(report);

  // A failed job is the most specific thing we know: say so, with its reason.
  if (job?.status === "failed") {
    return {
      state: "failed",
      title: "Reconstruction failed",
      detail: job.error?.trim() || "The reconstruction job failed without a recorded reason.",
      stages,
      missing: [],
    };
  }

  // A job still running is processing, whatever partial geometry exists.
  if (job?.status === "queued" || job?.status === "running") {
    return {
      state: "processing",
      title: "Reconstruction in progress",
      detail: job.stage
        ? `Backend stage: ${job.stage}.`
        : "The reconstruction job has not reached a terminal state yet.",
      stages,
      missing: [],
    };
  }

  if (!model) {
    return {
      state: "not_reconstructed",
      title: "Not reconstructed yet",
      detail:
        "This world has no reconstruction version. Run reconstruction on an attached session with sufficient image evidence.",
      stages,
      missing: [],
    };
  }

  // A compiled version exists. Judge completeness against the classes an
  // interior needs -- not against the classes the engine can emit, or every
  // world would look "partial" forever.
  const missing: string[] = [];
  if (model.counts.room === 0) missing.push("no rooms");
  if (model.counts.wall === 0) missing.push("no walls");
  if (model.counts.floor === 0) missing.push("no floors");
  if (model.counts.ceiling === 0) missing.push("no ceilings");

  const failedStage = stages.find((s) => s.status === "failed");
  if (failedStage) {
    return {
      state: "partial",
      title: "Partial reconstruction",
      detail: `Stage "${failedStage.name}" failed${failedStage.note ? `: ${failedStage.note}` : "."} A version was committed with the stages that did succeed.`,
      stages,
      missing,
    };
  }

  if (missing.length > 0) {
    const isEvidenceShort = missing.length === 4;
    return {
      state: isEvidenceShort ? "insufficient_evidence" : "partial",
      title: isEvidenceShort
        ? "Insufficient evidence for interior structure"
        : "Partial reconstruction",
      detail: isEvidenceShort
        ? "A version was committed, but the engine closed no room from the evidence available, so no interior structure was inferred."
        : `A version was committed, but it is missing ${missing.join(", ")}.`,
      stages,
      missing,
    };
  }

  return {
    state: "complete",
    title: "Reconstruction complete",
    detail: "The version contains rooms, walls, floors, and ceilings inferred from the captured evidence.",
    stages,
    missing,
  };
}
