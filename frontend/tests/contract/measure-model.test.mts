/**
 * Measurement view-model: unit/scale honesty and preview math.
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  MEASURE_TOOLS,
  computePreview,
  formatUncertainty,
  formatValue,
  histogram,
  resolveUnitContext,
  unitFor,
  unitLabelFromRow,
} from "../../src/lib/viewport/measure-model.ts";

const reconstructed = (state?: string) => ({
  metadata: state ? { scale: { state } } : {},
  entities: { a: { provenance: "RECONSTRUCTED" } },
});

test("unknown / relative / missing / nominal scale never yields metres", () => {
  for (const w of [reconstructed("relative"), reconstructed("unknown"), reconstructed("nominal"), reconstructed("unscaled"), reconstructed(), null, undefined]) {
    const ctx = resolveUnitContext(w);
    assert.equal(ctx.basis, "unknown");
    assert.equal(ctx.lengthUnit, "model units");
    assert.ok(!/\bm\b|m²/.test(`${unitFor("distance", ctx)} ${unitFor("area", ctx)}`), "no metre unit");
    assert.equal(ctx.badge, "scale unknown");
    assert.equal(ctx.uncertainty, null);
  }
});

test("metric-anchored world reads metres with no invented uncertainty", () => {
  const ctx = resolveUnitContext(reconstructed("metric"));
  assert.equal(ctx.basis, "metric");
  assert.equal(unitFor("distance", ctx), "m");
  assert.equal(unitFor("area", ctx), "m²");
  assert.equal(ctx.uncertainty, null);
  assert.equal(formatUncertainty(ctx.uncertainty, "m"), "Uncertainty: not estimated");
});

test("procedural (all GENERATED) world is authored metres, exact", () => {
  const ctx = resolveUnitContext({ entities: { r: { provenance: "GENERATED" }, w: { provenance: "GENERATED" } } });
  assert.equal(ctx.basis, "authored");
  assert.equal(ctx.lengthUnit, "m");
  assert.equal(ctx.method, "authored geometry (exact)");
  assert.equal(ctx.uncertainty, 0);
  assert.equal(formatUncertainty(0, "m"), "Uncertainty: exact (authored)");
});

test("a mixed generated + reconstructed world with unknown scale is NOT authored", () => {
  const ctx = resolveUnitContext({ entities: { r: { provenance: "GENERATED" }, w: { provenance: "RECONSTRUCTED" } } });
  assert.equal(ctx.basis, "unknown");
});

test("stored row units map without inventing metres", () => {
  assert.equal(unitLabelFromRow("model_units"), "model units");
  assert.equal(unitLabelFromRow("model_units2"), "model units²");
  assert.equal(unitLabelFromRow("m2"), "m²");
  assert.equal(formatValue(Number.NaN, "m"), "—");
  assert.equal(formatUncertainty(0.02, "m"), "Uncertainty: ± 0.020 m");
});

test("preview math matches the server rules", () => {
  assert.equal(computePreview("distance", [[0, 0, 0], [3, 4, 12]]), 13);
  assert.equal(computePreview("height", [[0, 5, 0], [9, 2, 9]]), 3);
  assert.equal(computePreview("elevation_difference", [[0, 5, 0], [9, 2, 9]]), -3);
  assert.ok(Math.abs((computePreview("area", [[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]]) ?? 0) - 12) < 1e-9);
  assert.equal(computePreview("area", [[0, 0, 0], [1, 0, 0]]), null);
});

test("unavailable tools are disabled with the honest title; available ones are not", () => {
  const off = MEASURE_TOOLS.filter((t) => t.unavailable).map((t) => t.label);
  assert.deepEqual(off, ["Volume", "Road Width", "Building Footprint"]);
  assert.ok(MEASURE_TOOLS.filter((t) => t.unavailable).every((t) => t.unavailable === "Not available yet"));
});

test("histogram bins every value once", () => {
  const h = histogram([1, 2, 2, 9], 4);
  assert.equal(h.reduce((s, b) => s + b.count, 0), 4);
  assert.deepEqual(histogram([]), []);
  assert.equal(histogram([5, 5]).length, 1);
});
