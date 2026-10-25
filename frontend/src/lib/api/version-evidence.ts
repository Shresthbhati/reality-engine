/**
 * What an EARLIER version was built from. Pure and dependency-free (type imports only) so it is testable under
 * `node --test`, and so the evidence panel never shows the current model's per-photo facts as if they described the
 * version being inspected.
 *
 * A version records the photographs it used (`evidence_ids`) and the ones it placed (`registered_ids`). A version that
 * predates that record has `null`: the answer is then "unknown", never a guess.
 */
import type { EvidenceStatus, VersionStatus } from "./reconstructions";

/** How one photograph relates to an inspected version. */
export type VersionRelation = "placed" | "waiting" | "added_later" | "unknown";

export function evidenceInVersion(
  evidenceId: string,
  version: Pick<VersionStatus, "evidence_ids" | "registered_ids"> | null,
): VersionRelation {
  if (!version || !version.evidence_ids) return "unknown";
  if (!version.evidence_ids.includes(evidenceId)) return "added_later";
  return version.registered_ids?.includes(evidenceId) ? "placed" : "waiting";
}

/** One line for an inspected version: photos it used / placed / left waiting, and how many came after it. */
export function versionEvidenceSummary(
  evidence: Pick<EvidenceStatus, "id">[],
  version: Pick<VersionStatus, "label" | "evidence_ids" | "registered_ids">,
): string {
  if (!version.evidence_ids) return `${version.label} did not record which photos it used.`;
  const rel = evidence.map((e) => evidenceInVersion(e.id, version));
  const placed = rel.filter((r) => r === "placed").length;
  const waiting = rel.filter((r) => r === "waiting").length;
  const later = rel.filter((r) => r === "added_later").length;
  const used = placed + waiting;
  return (
    `${version.label}: ${used} photo${used === 1 ? "" : "s"} used, ${placed} placed` +
    `${waiting ? `, ${waiting} waiting` : ""}${later ? ` · ${later} added since` : ""}`
  );
}
