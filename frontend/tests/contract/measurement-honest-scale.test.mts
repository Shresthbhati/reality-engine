import test from "node:test";
import assert from "node:assert/strict";
import {
  resolveUnitContext,
  formatValue,
  formatUncertainty,
  unitFor,
  unitLabelFromRow,
  stateFor,
} from "../../src/lib/viewport/measure-model.ts";

test("resolveUnitContext honestly reflects scaleCalibrationState as METRIC_CALIBRATED or UNSCALED_RELATIVE", () => {
  // Metric world
  const metricWorld = {
    metadata: { scale: { state: "metric" } },
    entities: { e1: { provenance: "OBSERVED" } },
  };
  const metricCtx = resolveUnitContext(metricWorld);
  assert.equal(metricCtx.scaleCalibrationState, "METRIC_CALIBRATED");
  assert.equal(metricCtx.basis, "metric");
  assert.equal(metricCtx.lengthUnit, "m");
  assert.equal(metricCtx.areaUnit, "m²");
  assert.equal(metricCtx.state, "MEASURED");
  assert.equal(metricCtx.badge, null);

  // Authored procedural world (exact)
  const authoredWorld = {
    entities: {
      room: { provenance: "GENERATED" },
      wall: { provenance: "GENERATED" },
    },
  };
  const authoredCtx = resolveUnitContext(authoredWorld);
  assert.equal(authoredCtx.scaleCalibrationState, "METRIC_CALIBRATED");
  assert.equal(authoredCtx.basis, "authored");
  assert.equal(authoredCtx.lengthUnit, "m");
  assert.equal(authoredCtx.uncertainty, 0);

  // Relative / Unscaled world
  const relativeWorld = {
    metadata: { scale: { state: "relative" } },
    entities: { e1: { provenance: "RECONSTRUCTED" } },
  };
  const relCtx = resolveUnitContext(relativeWorld);
  assert.equal(relCtx.scaleCalibrationState, "UNSCALED_RELATIVE");
  assert.equal(relCtx.basis, "unknown");
  assert.equal(relCtx.lengthUnit, "model units");
  assert.equal(relCtx.areaUnit, "model units²");
  assert.equal(relCtx.state, "RECONSTRUCTED");
  assert.equal(relCtx.badge, "scale unknown");
  assert.equal(relCtx.uncertainty, null);
});

test("measurement units and uncertainty strings never invent precision", () => {
  const metricCtx = resolveUnitContext({ metadata: { scale: { state: "metric" } } });
  assert.equal(formatValue(2.05, unitFor("distance", metricCtx), 2), "2.05 m");
  assert.equal(formatValue(2.05, unitFor("distance", metricCtx), 3), "2.050 m");
  assert.equal(formatUncertainty(0.015, metricCtx.lengthUnit), "Uncertainty: ± 0.015 m");
  assert.equal(formatUncertainty(null, metricCtx.lengthUnit), "Uncertainty: not estimated");
  assert.equal(formatUncertainty(0, metricCtx.lengthUnit), "Uncertainty: exact (authored)");

  const relCtx = resolveUnitContext({ metadata: { scale: { state: "relative" } } });
  assert.equal(formatValue(5.42, unitFor("distance", relCtx), 2), "5.42 model units");
  assert.equal(formatUncertainty(null, relCtx.lengthUnit), "Uncertainty: not estimated");
});

test("stateFor maps states honestly without allowing relative model-unit dimensions to claim MEASURED", () => {
  // A relative dimension is forced to RECONSTRUCTED even if claimed MEASURED
  assert.equal(stateFor({ state: "MEASURED", unit: "model_units" }), "RECONSTRUCTED");
  assert.equal(stateFor({ state: "VERIFIED", unit: "model_units" }), "RECONSTRUCTED");

  // In calibrated metric scale, MEASURED, ESTIMATED, and VERIFIED are honored
  assert.equal(stateFor({ state: "MEASURED", unit: "m" }), "MEASURED");
  assert.equal(stateFor({ state: "ESTIMATED", unit: "m" }), "ESTIMATED");
  assert.equal(stateFor({ state: "VERIFIED", unit: "m" }), "VERIFIED");
  assert.equal(unitLabelFromRow("m"), "m");
  assert.equal(unitLabelFromRow("model_units"), "model units");
});
