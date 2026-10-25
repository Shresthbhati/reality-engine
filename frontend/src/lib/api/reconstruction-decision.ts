/**
 * How a version was built and what changed in it, as plain words. Pure and dependency-free (type imports only) so it
 * is testable under `node --test`, and so the Studio only ever states what the status API actually reported: a
 * field that was not measured yields `null`, never a sentence that sounds like a measurement.
 */
import type { WorldStatus } from "./reconstructions";

type Model = NonNullable<WorldStatus["model"]>;

/** The ten relations between consecutive versions, in the order the Studio shows them. */
export const CHANGE_KINDS = [
  "preserved",
  "refined",
  "extended",
  "reduced",
  "split",
  "merge",
  "regrouped",
  "ambiguous",
  "removed",
  "new",
] as const;
export type ChangeKind = (typeof CHANGE_KINDS)[number];

export const CHANGE_LABEL: Record<ChangeKind, string> = {
  preserved: "Preserved",
  refined: "Refined",
  extended: "Extended",
  reduced: "Partly reproduced",
  split: "Split",
  merge: "Merged",
  regrouped: "Regrouped",
  ambiguous: "Ambiguous",
  removed: "Removed",
  new: "New",
};

/** One entry per category, zero included: a category with no changes reads 0 instead of being absent. */
export function changeCategoryCounts(
  counts: Partial<Record<ChangeKind, number>> | null | undefined,
): { kind: ChangeKind; label: string; count: number }[] {
  return CHANGE_KINDS.map((kind) => ({ kind, label: CHANGE_LABEL[kind], count: counts?.[kind] ?? 0 }));
}

export interface BuildDescription {
  /** how the sparse model was produced */
  method: string | null;
  /** why that was chosen (the arbitration's deciding fact when two candidates were judged, else the strategy's reason) */
  why: string | null;
  frame: string | null;
  /** what each candidate placed, when both were built */
  comparison: string | null;
  /** accepted / accepted with open questions */
  decision: string | null;
  dense: string | null;
}

const METHOD: Record<string, string> = {
  incremental: "Added the new photos to the existing model",
  full: "Re-solved all photos together",
  reused: "Reused the existing model (nothing new to place)",
};

type Strategy = NonNullable<Model["strategy"]>;

/**
 * The frame sentence prefers what was MEASURED (the chosen candidate's frame shift against the current version) over
 * the strategy's mode label: an incremental build preserves COLMAP's raw frame, but the world's visible frame is
 * what the user sees, and only the measurement can say whether it moved.
 */
function frameText(s: Strategy): string {
  const winner = s.arbitration ? s.arbitration[s.arbitration.choice] : null;
  const fs = winner?.measures?.frame_shift;
  if (fs && fs.changed === true) {
    const parts = [
      typeof fs.rotation_deg === "number" ? `rotated ${fs.rotation_deg.toFixed(1)}°` : null,
      fs.scale_change > 0.005 ? `rescaled ${(fs.scale_change * 100).toFixed(1)}%` : null,
      typeof fs.translation_rel === "number" && fs.translation_rel > 0.01
        ? `shifted ${(fs.translation_rel * 100).toFixed(0)}% of the scene` : null,
    ].filter(Boolean);
    return `Coordinate frame changed (${parts.join(", ") || "measured shift"}): positions can differ from the previous version`;
  }
  if (fs && fs.changed === false) return "Coordinate frame kept: everything already placed stays where it was";
  return s.frame === "preserved"
    ? "Coordinate frame kept: everything already placed stays where it was"
    : "Coordinate frame re-solved: positions can differ from the previous version";
}

export function describeBuild(model: Pick<Model, "strategy" | "verdict" | "dense">): BuildDescription {
  const s = model.strategy ?? null;
  const arb = s?.arbitration ?? null;
  return {
    method: s ? (METHOD[s.mode] ?? null) : null,
    why: arb?.why ?? (s?.reason || null),
    frame: s ? frameText(s) : null,
    comparison:
      s && typeof s.incremental_registered === "number" && typeof s.full_registered === "number"
        ? `Extending placed ${s.incremental_registered} photo${s.incremental_registered === 1 ? "" : "s"}; ` +
          `re-solving everything placed ${s.full_registered}`
        : null,
    decision:
      model.verdict === "ACCEPT"
        ? "Accepted: it keeps what was established and adds to it"
        : model.verdict === "ACCEPT_WITH_UNCERTAINTY"
          ? "Accepted with open questions (listed below)"
          : null,
    dense: model.dense
      ? `${model.dense.state === "dense" ? "Dense surface built" : "Sparse only"}${model.dense.detail ? `: ${model.dense.detail}` : ""}`
      : null,
  };
}
