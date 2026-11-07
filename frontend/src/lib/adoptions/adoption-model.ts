/**
 * View-model helpers for World Adoption Review UI (Task U.7 / Decision 027).
 */

import type { AdoptionChangeSet, AlignmentTier } from "@/lib/api/adoptions";

export const ALIGNMENT_TIER_META: Record<
  AlignmentTier,
  { label: string; fg: string; bg: string; description: string }
> = {
  HIGH: {
    label: "High Confidence",
    fg: "var(--success)",
    bg: "var(--success-subtle)",
    description: "Strong feature correspondence and scale match (≥ 85%). Automatic merge ready.",
  },
  MODERATE: {
    label: "Moderate Confidence",
    fg: "var(--warning)",
    bg: "var(--warning-subtle)",
    description: "Acceptable spatial overlap (60%–84%). Review uncertain and conflicting items.",
  },
  LOW: {
    label: "Low Confidence",
    fg: "var(--error)",
    bg: "var(--error-subtle)",
    description: "Insufficient correspondence (< 60%). Automatic alignment refused by policy.",
  },
};

export const CHANGE_CATEGORY_META = {
  new: {
    label: "New",
    description: "Unseen entities detected in session to be added as first-class objects.",
    badgeFg: "var(--accent)",
  },
  matched: {
    label: "Matched",
    description: "Existing entities confirmed within 2 cm tolerance. Tightens measurement certainty.",
    badgeFg: "var(--success)",
  },
  updated: {
    label: "Updated",
    description: "Existing unlocked entities refined with denser observations.",
    badgeFg: "var(--info, #38bdf8)",
  },
  uncertain: {
    label: "Uncertain",
    description: "Partially observed or boundary-ambiguous entities. Choose to include or drop.",
    badgeFg: "var(--warning)",
  },
  conflicting: {
    label: "Conflicting",
    description: "Geometric clashes or clashes with protected manual edits (Rule 18).",
    badgeFg: "var(--error)",
  },
} as const;

export type ChangeCategoryKey = keyof typeof CHANGE_CATEGORY_META;

export function formatConfidencePercent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function defaultConflictResolutions(
  changeSet: AdoptionChangeSet,
): Record<string, "keep_edit" | "accept_reconstruction"> {
  const resolutions: Record<string, "keep_edit" | "accept_reconstruction"> = {};
  for (const id of changeSet.conflicting) {
    // Rule 18 default: keep manual edit unless user explicitly chooses to overwrite
    resolutions[id] = "keep_edit";
  }
  return resolutions;
}

export function getTotalChangesCount(changeSet: AdoptionChangeSet): number {
  return (
    changeSet.new.length +
    changeSet.matched.length +
    changeSet.updated.length +
    changeSet.uncertain.length +
    changeSet.conflicting.length
  );
}
