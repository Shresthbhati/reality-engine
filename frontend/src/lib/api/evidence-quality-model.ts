/**
 * Evidence Quality Firewall view model. Pure and dependency-free so it is testable under `node --test`.
 *
 * Everything here reshapes what GET /api/evidence/quality returned: counters, tab filters, tones and labels. No value is
 * computed from pixels or invented -- check results, scores and contribution labels all come from the server.
 */

export type QualityDecision = "accepted" | "accepted_with_warning" | "needs_review" | "rejected";
export type QualityResult = QualityDecision | "not_inspected";
export type CheckStatus = "pass" | "warn" | "fail";

export interface QualityCheck {
  name: string;
  status: CheckStatus;
  value: string | null;
  threshold: string | null;
  message: string;
}

export interface QualityContribution {
  available: boolean;
  reason?: string;
  label?: string | null;
  status?: string | null;
  verified_matches?: number | null;
  median_shift_fraction?: number | null;
  best_partner_id?: string | null;
  best_partner_name?: string | null;
  basis?: string | null;
}

export interface QualityPolicy {
  resolution_fail_mp: number;
  resolution_warn_mp: number;
  sharpness_fail: number;
  sharpness_warn: number;
  warn_penalty: number;
  fail_penalty: number;
}

export interface EvidenceQualityItem {
  id: string;
  name: string;
  session_id: string | null;
  world_id: string | null;
  source: string;
  source_video: string | null;
  added_at: string | null;
  mime_type: string | null;
  size: number | null;
  width: number | null;
  height: number | null;
  sharpness: number | null;
  evidence_class: string | null;
  class_basis: string | null;
  gps: [number, number] | null;
  inspected: boolean;
  checks: QualityCheck[];
  score: number | null;
  auto_result: QualityDecision | null;
  key_issues: string[];
  explanation: string;
  decision: { decision: QualityDecision; reason: string | null; decided_at: string | null } | null;
  result: QualityDecision | null;
  contribution: QualityContribution;
  policy?: QualityPolicy;
}

export type QualityTab = "all" | QualityResult;

export const RESULT_LABEL: Record<QualityResult, string> = {
  accepted: "Accepted",
  accepted_with_warning: "Accepted with warning",
  needs_review: "Needs review",
  rejected: "Rejected",
  not_inspected: "Not inspected",
};

export const TABS: QualityTab[] = ["all", "accepted", "accepted_with_warning", "needs_review", "rejected", "not_inspected"];

export function resultOf(item: Pick<EvidenceQualityItem, "result">): QualityResult {
  return item.result ?? "not_inspected";
}

export function countResults(items: Pick<EvidenceQualityItem, "result">[]): Record<QualityResult, number> & { total: number } {
  const out = { accepted: 0, accepted_with_warning: 0, needs_review: 0, rejected: 0, not_inspected: 0, total: items.length };
  for (const i of items) out[resultOf(i)] += 1;
  return out;
}

export function filterItems<T extends Pick<EvidenceQualityItem, "result" | "name">>(items: T[], tab: QualityTab, query: string): T[] {
  const q = query.trim().toLowerCase();
  return items.filter((i) => (tab === "all" || resultOf(i) === tab) && (!q || i.name.toLowerCase().includes(q)));
}

/** Share of the total as a whole percent; "0%" for an empty list (never NaN). */
export function percentOf(part: number, total: number): string {
  return total > 0 ? `${Math.round((part / total) * 100)}%` : "0%";
}

export type Tone = "success" | "warning" | "error" | "neutral";

export function resultTone(result: QualityResult): Tone {
  return result === "accepted" ? "success" : result === "rejected" ? "error" : result === "not_inspected" ? "neutral" : "warning";
}

export function checkTone(status: CheckStatus): Tone {
  return status === "pass" ? "success" : status === "warn" ? "warning" : "error";
}

/** Ring colour for a 0-100 score: the same bands as the formula's penalties (a failed check costs 40, a warning 15). */
export function scoreTone(score: number | null): Tone {
  if (score === null) return "neutral";
  return score >= 85 ? "success" : score >= 60 ? "warning" : "error";
}

export function formatBytes(n: number | null): string {
  if (n === null || n === undefined) return "Unknown";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDimensions(i: Pick<EvidenceQualityItem, "width" | "height">): string {
  return i.width && i.height ? `${i.width} x ${i.height}` : "Unknown";
}

const CONTRIBUTION_LABEL: Record<string, string> = {
  first: "First image (nothing earlier to compare)",
  new_view: "New viewpoint",
  redundant: "Near-duplicate viewpoint",
  disconnected: "No verified overlap with other images",
  unknown: "Not measurable",
};

/** One line describing the measured spatial contribution; says why when it was not measured. */
export function contributionSummary(c: QualityContribution): string {
  if (!c.available) return `Not measured: ${c.reason ?? "no contribution record"}`;
  const label = CONTRIBUTION_LABEL[c.label ?? "unknown"] ?? (c.label ?? "Not measurable");
  const n = c.verified_matches ?? 0;
  return n > 0 ? `${label} (${n} verified feature matches)` : label;
}

/** Position of `value` on a bar that spans 0..max, clamped to 0..100 (percent). */
export function barPercent(value: number, max: number): number {
  return max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
}
