/**
 * Change-proposal review logic (pure; no React, no network): which changes are pre-selected, what an acceptance
 * request contains, and why the Accept button may be disabled. Mirrors the server rules in worldstore/proposals.py:
 * supported changes apply by default; needs_review changes apply only when named; conflicts additionally need an
 * explicit override and a written reason. The server re-checks everything -- this only keeps the UI honest.
 *
 * Run with: npm run test:contract
 */
import type { ProposalAssessmentDto, ProposalSummaryDto, ProposalVerdict } from "../api/proposals";

export const VERDICT_RANK: Record<ProposalVerdict, number> = { conflict: 0, needs_review: 1, supported: 2 };

export const VERDICT_LABEL: Record<ProposalVerdict, string> = {
  conflict: "Conflict",
  needs_review: "Needs review",
  supported: "Supported",
};

/** Conflicts first, then needs_review, then supported; stable by entity id within a group. */
export function sortAssessments(items: ProposalAssessmentDto[]): ProposalAssessmentDto[] {
  return [...items].sort(
    (a, b) => VERDICT_RANK[a.verdict] - VERDICT_RANK[b.verdict] || a.entity_id.localeCompare(b.entity_id),
  );
}

/** What the server would apply with no explicit selection: every supported change, nothing else. */
export function defaultSelection(items: ProposalAssessmentDto[]): string[] {
  return items.filter((a) => a.verdict === "supported").map((a) => a.entity_id).sort();
}

export interface AcceptPlan {
  entity_ids: string[];
  override_conflicts: string[];
  reason: string;
  /** Selected changes whose verdict is conflict: they are applied only because the user chose to. */
  overridden: string[];
  /** Why the request cannot be sent yet; empty means it can. */
  blockers: string[];
}

/** The acceptance request implied by the current selection. Every selected conflict is an explicit override. */
export function planAcceptance(items: ProposalAssessmentDto[], selected: Iterable<string>, reason: string): AcceptPlan {
  const known = new Map(items.map((a) => [a.entity_id, a]));
  const entity_ids = [...new Set(selected)].filter((id) => known.has(id)).sort();
  const overridden = entity_ids.filter((id) => known.get(id)!.verdict === "conflict");
  const blockers: string[] = [];
  if (entity_ids.length === 0) blockers.push("Select at least one change to apply.");
  if (overridden.length > 0 && reason.trim() === "") {
    blockers.push(
      `Overriding ${overridden.length} conflicting change${overridden.length === 1 ? "" : "s"} needs a written reason.`,
    );
  }
  return { entity_ids, override_conflicts: overridden, reason: reason.trim(), overridden, blockers };
}

export function summaryLine(s: ProposalSummaryDto): string {
  if (s.changes === 0) return "No changes";
  const parts: string[] = [];
  if (s.conflict) parts.push(`${s.conflict} conflict${s.conflict === 1 ? "" : "s"}`);
  if (s.needs_review) parts.push(`${s.needs_review} need review`);
  if (s.supported) parts.push(`${s.supported} supported`);
  return parts.join(" · ");
}

/** "0.90 → 0.40", "new 0.80", "was 0.70"; null when neither side has a value. */
export function confidenceText(a: ProposalAssessmentDto): string | null {
  const f = (n: number) => n.toFixed(2);
  if (a.old_confidence != null && a.new_confidence != null) return `${f(a.old_confidence)} → ${f(a.new_confidence)}`;
  if (a.new_confidence != null) return `new ${f(a.new_confidence)}`;
  if (a.old_confidence != null) return `was ${f(a.old_confidence)}`;
  return null;
}
