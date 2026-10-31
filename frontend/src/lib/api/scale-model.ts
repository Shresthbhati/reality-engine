/**
 * Scale presentation model (pure, no imports so `npm run test:contract` can run it).
 *
 * The honesty rule from Decision 023: a world is RELATIVE (model units), METRIC measured, or METRIC but
 * ESTIMATED from a typical size the user accepted. Nothing here ever turns a missing value into a number.
 */

export interface ScaleStageDto {
  state?: string | null;
  meters_per_unit?: number | null;
  scale_basis?: string | null;
  dimension_state?: string | null;
}

export interface ScaleSuggestionDto {
  source_entity_id: string;
  source_kind: string;
  measured_model_units: number;
  typical_range_m: [number, number];
  suggested_value_m: number;
  implied_scale: number;
  basis: string;
  n_candidates: number;
}

export type ScaleView =
  | { kind: "relative"; cta: string }
  | { kind: "estimated"; label: string }
  | { kind: "measured"; label: string }
  | { kind: "unknown" };

export const RELATIVE_CTA = "Dimensions are not in metres. Enter one known distance.";
export const PRIOR_METHOD = "prior_accepted_by_user";

export function describeScale(scale: ScaleStageDto | null | undefined): ScaleView {
  const state = scale?.state?.toLowerCase();
  if (state === "metric") {
    if (scale?.dimension_state === "ESTIMATED" || scale?.scale_basis === PRIOR_METHOD) {
      return { kind: "estimated", label: "ESTIMATED from a typical door size" };
    }
    return { kind: "measured", label: "Metric, from a distance you measured" };
  }
  if (state === "relative" || state === "unknown") return { kind: "relative", cta: RELATIVE_CTA };
  return { kind: "unknown" };
}

/** The sentence the suggestion card shows. Every number comes from the suggestion; nothing is defaulted. */
export function suggestionSentence(s: ScaleSuggestionDto): string {
  const [lo, hi] = s.typical_range_m;
  return (
    `The ${s.source_kind} is ${s.measured_model_units.toFixed(2)} model units tall. ` +
    `Typical ${s.source_kind}s are ${lo.toFixed(1)}\u2013${hi.toFixed(1)} m. ` +
    `Use ${s.suggested_value_m.toFixed(2)} m as scale?`
  );
}
