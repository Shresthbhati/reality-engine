/**
 * Scale presentation model: RELATIVE shows the call to action, an accepted prior is labelled ESTIMATED,
 * and the suggestion sentence uses only the numbers the API returned.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { RELATIVE_CTA, describeScale, suggestionSentence, type ScaleSuggestionDto } from "../../src/lib/api/scale-model.ts";

test("a relative or unknown world shows the call to action, never metres", () => {
  assert.deepEqual(describeScale({ state: "relative" }), { kind: "relative", cta: RELATIVE_CTA });
  assert.equal(RELATIVE_CTA, "Dimensions are not in metres. Enter one known distance.");
  assert.equal(describeScale({ state: "unknown", meters_per_unit: null }).kind, "relative");
  assert.equal(describeScale(null).kind, "unknown");
});

test("a metric world scaled by an accepted prior is ESTIMATED, a measured one is not", () => {
  const est = describeScale({ state: "metric", meters_per_unit: 2.2, scale_basis: "prior_accepted_by_user", dimension_state: "ESTIMATED" });
  assert.deepEqual(est, { kind: "estimated", label: "ESTIMATED from a typical door size" });
  assert.equal(describeScale({ state: "metric", scale_basis: "prior_accepted_by_user" }).kind, "estimated");
  assert.equal(describeScale({ state: "metric", scale_basis: "known_length", dimension_state: "MEASURED" }).kind, "measured");
});

test("the suggestion sentence states the object, the range and the value from the API", () => {
  const s: ScaleSuggestionDto = {
    source_entity_id: "door-1", source_kind: "door", measured_model_units: 0.93, typical_range_m: [2.0, 2.1],
    suggested_value_m: 2.05, implied_scale: 2.204, basis: "x", n_candidates: 1,
  };
  assert.equal(suggestionSentence(s), "The door is 0.93 model units tall. Typical doors are 2.0\u20132.1 m. Use 2.05 m as scale?");
});
