/**
 * What a correction commit actually did. Pure and dependency-free so it is testable under `node --test`.
 *
 * The API answers a byte-identical retry with 200 and `duplicate: true` and the CURRENT version: nothing was
 * persisted. The Studio must say so, rather than present it as a fresh commit whose parent is itself.
 */
export interface CommitOutcomeInput {
  version_id: string;
  parent_version_id?: string;
  duplicate?: boolean;
}

export type CommitOutcome =
  | { kind: "created"; versionId: string; parentVersionId?: string }
  | { kind: "duplicate"; versionId: string };

export function commitOutcome(res: CommitOutcomeInput): CommitOutcome {
  if (res.duplicate === true) return { kind: "duplicate", versionId: res.version_id };
  return { kind: "created", versionId: res.version_id, parentVersionId: res.parent_version_id };
}
