/**
 * Contract tests for Surface Quality Analysis Model.
 * Verifies ASTM E1155 classifications, deflection codes, token styling, and formatters.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  getASTMClassificationMeta,
  getDeflectionCodeMeta,
  formatFFNumber,
  formatDeviationMm,
} from "../../src/lib/studio/surface-quality-model.ts";

test("ASTM classification metadata provides WCAG-compliant design token styling", () => {
  const superFlat = getASTMClassificationMeta("super_flat");
  assert.equal(superFlat.label, "Super Flat");
  assert.ok(superFlat.badgeClass.includes("text-success"));

  const veryFlat = getASTMClassificationMeta("very_flat");
  assert.equal(veryFlat.label, "Very Flat");
  assert.ok(veryFlat.badgeClass.includes("text-accent"));

  const flat = getASTMClassificationMeta("flat");
  assert.equal(flat.label, "Flat");

  const modFlat = getASTMClassificationMeta("moderately_flat");
  assert.equal(modFlat.label, "Moderately Flat");
  assert.ok(modFlat.badgeClass.includes("text-warning"));

  const nonCompliant = getASTMClassificationMeta("non_compliant");
  assert.equal(nonCompliant.label, "Non-Compliant");
  assert.ok(nonCompliant.badgeClass.includes("text-danger"));
});

test("Deflection code metadata categorizes structural criteria accurately", () => {
  const strict = getDeflectionCodeMeta("L/480");
  assert.equal(strict.label, "Strict (L/480)");
  assert.ok(strict.badgeClass.includes("text-success"));

  const standard = getDeflectionCodeMeta("L/360");
  assert.equal(standard.label, "Standard (L/360)");
  assert.ok(standard.badgeClass.includes("text-accent"));

  const permissible = getDeflectionCodeMeta("L/240");
  assert.equal(permissible.label, "Permissible (L/240)");
  assert.ok(permissible.badgeClass.includes("text-warning"));

  const excessive = getDeflectionCodeMeta("excessive");
  assert.ok(excessive.label.includes("Excessive"));
  assert.ok(excessive.badgeClass.includes("text-danger"));
});

test("formatFFNumber formats floating point FF values to single decimal place", () => {
  assert.equal(formatFFNumber(35.267), "35.3");
  assert.equal(formatFFNumber(50), "50.0");
  assert.equal(formatFFNumber(0.04), "0.0");
});

test("formatDeviationMm formats signed millimeter deviations with polarity prefix", () => {
  assert.equal(formatDeviationMm(8.34), "+8.3 mm");
  assert.equal(formatDeviationMm(-14.21), "-14.2 mm");
  assert.equal(formatDeviationMm(0.0), "0.0 mm");
});
