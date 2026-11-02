/**
 * Identity model (plan §6 "overlap view"): what the association engine recorded about WHY an entity is one entity.
 * Reads `entity.custom_properties.association`, written by the pipeline's object-resolution stage:
 *   { resolver, observations, images, decision, ambiguous_with: string[] }
 * Pure (no React) so it is unit-testable under Node. A missing record is reported as "not recorded", never as a
 * confident merge, and an AMBIGUOUS decision is never presented as settled.
 */

export type IdentityState = "merged" | "single" | "ambiguous" | "not_recorded";

export interface IdentityEvidence {
  state: IdentityState;
  photoCount: number | null;
  /** other entities this one could not be told apart from (only for state "ambiguous") */
  ambiguousWith: string[];
  /** one short line for the card */
  text: string;
}

type Props = Record<string, unknown>;
const isRec = (v: unknown): v is Props => typeof v === "object" && v !== null && !Array.isArray(v);
const count = (v: unknown): number | null => (typeof v === "number" && Number.isInteger(v) && v >= 0 ? v : null);

export function buildIdentityEvidence(cp: Props | undefined): IdentityEvidence {
  const a = cp?.association;
  if (!isRec(a)) return { state: "not_recorded", photoCount: null, ambiguousWith: [], text: "Identity not recorded" };

  const photos = count(a.images);
  const ambiguousWith = Array.isArray(a.ambiguous_with) ? a.ambiguous_with.filter((x): x is string => typeof x === "string") : [];
  if (a.decision === "AMBIGUOUS" || ambiguousWith.length > 0) {
    return {
      state: "ambiguous", photoCount: photos, ambiguousWith,
      text: ambiguousWith.length ? `Could be the same as ${ambiguousWith.join(", ")}: unresolved` : "Could be another object: unresolved",
    };
  }
  if (photos !== null && photos >= 2) {
    return { state: "merged", photoCount: photos, ambiguousWith: [], text: `Same object in ${photos} photos` };
  }
  if (photos === 1) return { state: "single", photoCount: 1, ambiguousWith: [], text: "Seen in 1 photo" };
  return { state: "not_recorded", photoCount: photos, ambiguousWith: [], text: "Identity not recorded" };
}
