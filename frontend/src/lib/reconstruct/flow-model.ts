/**
 * Guided reconstruction flow (pure, node-testable). Five linear steps; every state and reason is derived from the
 * world status the API returned (`GET /api/worlds/{id}/status`). Nothing is invented: a stage name is the job's real
 * `stage`, a missing value becomes "not available" with the reason, never a default.
 */
import { describeScale } from "../api/scale-model.ts";

export type StepId = "photos" | "reconstruct" | "scale" | "review" | "measure";
export type StepState = "done" | "action" | "active" | "unavailable";
export type StepCta = "add_photos" | "open_world" | "set_scale" | "retry";

export interface FlowStep {
  id: StepId;
  n: number;
  title: string;
  state: StepState;
  /** one sentence: what is true and (for action / unavailable) why */
  reason: string;
  cta?: { kind: StepCta; label: string };
  /** the first step that still needs the user (or is running): where the user is now */
  current: boolean;
}

/** The subset of `WorldStatus` the flow reads (structural, so this file has no React/API imports). */
export interface FlowStatusInput {
  state: string;
  state_label?: string;
  in_progress: boolean;
  has_model: boolean;
  job?: { id: string; status: string; stage: string | null } | null;
  model?: {
    model_state?: string | null;
    images_used: number;
    images_registered: number;
    scale?: { state?: string | null; scale_basis?: string | null; dimension_state?: string | null; meters_per_unit?: number | null };
  } | null;
  evidence?: unknown[];
  evidence_summary?: { count: number };
  guidance?: { message: string }[];
  failure?: { message: string | null } | null;
}

/** Real job stage names (apps/api/jobs.py `job.stage = ...`) in plain words. Unknown names are shown humanised, as-is. */
const STAGE_LABEL: Record<string, string> = {
  verifying_artifact: "Checking the uploaded files",
  registering: "Registering the world",
  checking_reconstruction_backend: "Checking the reconstruction engine",
  resolving_evidence: "Collecting your photos",
  analyzing_evidence: "Analysing the photos",
  building_rough_model: "Building a first rough model",
  reconstructing: "Reconstructing the 3D structure",
  refining: "Refining the model",
  committing_version: "Saving the new version",
};

export function stageLabel(stage: string | null | undefined): string | null {
  if (!stage) return null;
  return STAGE_LABEL[stage] ?? stage.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

export const CAPTURE_TIPS = [
  "Cover every wall: stand in the middle or a corner and photograph each wall in turn, then the floor and ceiling edges.",
  "Overlap 60% or more between neighbouring photos so each one shares a clear patch with the next.",
  "Keep the camera steady and well lit. Avoid mirrors, glass and blank white walls where you can.",
];

export function buildFlowSteps(status: FlowStatusInput | null | undefined): FlowStep[] {
  const photoCount = status?.evidence_summary?.count ?? status?.evidence?.length ?? 0;
  const hasModel = Boolean(status?.has_model);
  const running = Boolean(status?.in_progress);
  const failed = status?.state === "FAILED" || Boolean(status?.failure);
  const cancelled = status?.state === "CANCELLED";

  const photos: FlowStep =
    !status || photoCount === 0
      ? {
          id: "photos", n: 1, title: "Add photos", state: "action", current: false,
          reason: "Add photos of the room. Cover every wall and overlap neighbouring photos by 60% or more.",
          cta: { kind: "add_photos", label: "Add photos" },
        }
      : {
          id: "photos", n: 1, title: "Add photos", state: "done", current: false,
          reason: `${photoCount} photo${photoCount === 1 ? "" : "s"} added. More views from other angles improve the model.`,
          cta: hasModel ? { kind: "add_photos", label: "Add more" } : undefined,
        };

  let reconstruct: FlowStep;
  if (!status || photoCount === 0) {
    reconstruct = { id: "reconstruct", n: 2, title: "Reconstruct", state: "unavailable", current: false, reason: "Starts automatically once photos are added." };
  } else if (running) {
    const stage = stageLabel(status.job?.stage);
    reconstruct = {
      id: "reconstruct", n: 2, title: "Reconstruct", state: "active", current: false,
      reason: stage ? `${stage}… (${status.state_label ?? status.state})` : `${status.state_label ?? "Working"}…`,
    };
  } else if (failed || cancelled) {
    reconstruct = {
      id: "reconstruct", n: 2, title: "Reconstruct", state: "action", current: false,
      reason: failed ? (status.failure?.message ?? "The reconstruction failed.") : "The reconstruction was cancelled. Your photos are kept.",
      cta: { kind: "retry", label: "Try again" },
    };
  } else if (hasModel) {
    const m = status.model;
    reconstruct = {
      id: "reconstruct", n: 2, title: "Reconstruct", state: "done", current: false,
      reason: m
        ? `Model built: ${m.images_registered} of ${m.images_used} photos placed${m.model_state ? ` (${m.model_state.toLowerCase()})` : ""}.`
        : "Model built.",
    };
  } else {
    reconstruct = {
      id: "reconstruct", n: 2, title: "Reconstruct", state: "action", current: false,
      reason: status.guidance?.[0]?.message ?? "No model yet. Add more overlapping photos.",
      cta: { kind: "add_photos", label: "Add photos" },
    };
  }

  let scale: FlowStep;
  const scaleView = describeScale(status?.model?.scale);
  if (!hasModel) {
    scale = { id: "scale", n: 3, title: "Set scale", state: "unavailable", current: false, reason: "Needs a reconstructed model first." };
  } else if (scaleView.kind === "relative") {
    scale = { id: "scale", n: 3, title: "Set scale", state: "action", current: false, reason: scaleView.cta, cta: { kind: "set_scale", label: "Set scale" } };
  } else if (scaleView.kind === "estimated") {
    scale = { id: "scale", n: 3, title: "Set scale", state: "done", current: false, reason: `${scaleView.label}. Enter a real distance to replace the estimate.` };
  } else if (scaleView.kind === "measured") {
    scale = { id: "scale", n: 3, title: "Set scale", state: "done", current: false, reason: scaleView.label };
  } else {
    scale = { id: "scale", n: 3, title: "Set scale", state: "unavailable", current: false, reason: "This version did not report a scale state." };
  }

  const review: FlowStep = hasModel
    ? {
        id: "review", n: 4, title: "Review the room", state: "action", current: false,
        reason: "Open the 3D view and switch between Coloured, Wireframe and Confidence.",
        cta: { kind: "open_world", label: "Open viewer" },
      }
    : { id: "review", n: 4, title: "Review the room", state: "unavailable", current: false, reason: "Needs a reconstructed model first." };

  const measure: FlowStep = !hasModel
    ? { id: "measure", n: 5, title: "Measure / export", state: "unavailable", current: false, reason: "Needs a reconstructed model first." }
    : scaleView.kind === "relative"
      ? { id: "measure", n: 5, title: "Measure / export", state: "action", current: false, reason: "You can measure now, but distances are in model units until the scale is set." }
      : { id: "measure", n: 5, title: "Measure / export", state: "action", current: false, reason: "Pick two points to measure a distance, or export the model." };

  const steps = [photos, reconstruct, scale, review, measure];
  const cur = steps.find((s) => s.state === "active" || s.state === "action");
  if (cur) cur.current = true;
  return steps;
}

/** Where a step CTA leads from the reconstruct page. No world yet => never a route (that would be a dead link). */
export type CtaTarget = { type: "route"; href: string } | { type: "photos" } | { type: "retry" };

export function ctaTarget(kind: StepCta, worldId: string | null | undefined): CtaTarget {
  if (kind === "retry") return { type: "retry" };
  if (kind === "add_photos" || !worldId) return { type: "photos" };
  const base = `/worlds/${encodeURIComponent(worldId)}`;
  return { type: "route", href: kind === "set_scale" ? `${base}?step=scale` : base };
}

/** Consistent back / next links between the reconstruct page and the viewer (null when there is no world to link to). */
export function flowNav(
  worldId: string | null | undefined,
  hasModel: boolean,
): { back: { href: string; label: string } | null; next: { href: string; label: string } | null } {
  if (!worldId) return { back: null, next: null };
  const href = `/worlds/${encodeURIComponent(worldId)}`;
  return {
    back: { href, label: "Back to the 3D viewer" },
    next: hasModel ? { href, label: "Next: review the room" } : null,
  };
}
