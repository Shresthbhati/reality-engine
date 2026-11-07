/** Furniture design controls: arithmetic and wording. A catalog size is only offered where metres mean something. */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  catalogLabel, degreesToRadians, deleteConfirmation, isFurniture, isMetricScale, normaliseDegrees, parseNumber,
  replaceBlockedReason, yawDegrees, type CatalogItem,
} from "../../src/lib/design/design-model.ts";

test("only furniture gets design controls", () => {
  assert.equal(isFurniture({ type: "furniture" }), true);
  assert.equal(isFurniture({ type: "wall" }), false);
  assert.equal(isFurniture(null), false);
});

test("a catalog size needs a metric world", () => {
  assert.equal(isMetricScale({ scale: { state: "metric" } }), true);
  assert.equal(isMetricScale({ scale: { state: "RELATIVE" } }), false);
  assert.equal(isMetricScale({}), false);
  assert.equal(isMetricScale(undefined), false);
  assert.equal(replaceBlockedReason(true), null);
  assert.match(replaceBlockedReason(false) ?? "", /Set scale/);
});

test("angles are kept in (-180, 180] and survive the radian round trip", () => {
  assert.equal(normaliseDegrees(270), -90);
  assert.equal(normaliseDegrees(-190), 170);
  assert.equal(normaliseDegrees(180), 180);
  assert.equal(normaliseDegrees(-180), 180);
  assert.equal(normaliseDegrees(720), 0);
  assert.ok(Math.abs(degreesToRadians(90) - Math.PI / 2) < 1e-12);
  const e = { custom_properties: { yaw_rad: Math.PI / 2 } };
  assert.equal(yawDegrees(e), 90);
  assert.equal(yawDegrees({ custom_properties: {} }), 0);
  assert.equal(yawDegrees(null), 0);
  // turning right by 90 from 170 wraps instead of reaching 260
  assert.equal(normaliseDegrees(170 + 90), -100);
});

test("typed numbers are never silently zero", () => {
  assert.equal(parseNumber("2.5"), 2.5);
  assert.equal(parseNumber("-1"), -1);
  assert.equal(parseNumber(""), null);
  assert.equal(parseNumber("   "), null);
  assert.equal(parseNumber("abc"), null);
  assert.equal(parseNumber("Infinity"), null);
});

test("wording says what will happen", () => {
  const item: CatalogItem = { id: "armchair", name: "Armchair", category: "armchair", length_m: 0.85, width_m: 0.85, height_m: 0.85, basis: "nominal" };
  assert.equal(catalogLabel(item), "Armchair - 0.85 x 0.85 x 0.85 m");
  assert.match(deleteConfirmation("Sofa"), /Delete Sofa .* undo/);
  assert.match(deleteConfirmation(""), /this piece/);
});
