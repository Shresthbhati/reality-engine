import test from "node:test";
import assert from "node:assert/strict";

import {
  formatCoordinates,
  formatStyleDisplay,
  formatPlausibilityStatus,
  isGeoreferenced,
  type SpatialIntelligenceSummary,
  type GeodeticOrigin,
  type PhysicalPlausibilitySummary,
} from "../../src/lib/studio/spatial-intelligence-model.ts";

test("formatCoordinates produces readable coordinates with directions and elevation", () => {
  const originNorthWest: GeodeticOrigin = {
    latitude_deg: 37.774929,
    longitude_deg: -122.419416,
    altitude_m: 24.3,
  };
  const strNW = formatCoordinates(originNorthWest);
  assert.equal(strNW, "37.77493° N, 122.41942° W (24m alt)");

  const originSouthEast: GeodeticOrigin = {
    latitude_deg: -33.86882,
    longitude_deg: 151.20929,
    altitude_m: 10.0,
  };
  const strSE = formatCoordinates(originSouthEast);
  assert.equal(strSE, "33.86882° S, 151.20929° E (10m alt)");

  assert.equal(formatCoordinates(null), "Local / Arbitrary Origin");
  assert.equal(formatCoordinates(undefined), "Local / Arbitrary Origin");
});

test("formatStyleDisplay transforms SCREAMING_SNAKE to Title Case", () => {
  assert.equal(formatStyleDisplay("MODERN_MINIMALIST"), "Modern Minimalist");
  assert.equal(formatStyleDisplay("INDUSTRIAL_LOFT"), "Industrial Loft");
  assert.equal(formatStyleDisplay("SCANDINAVIAN"), "Scandinavian");
  assert.equal(formatStyleDisplay("MID_CENTURY_MODERN"), "Mid Century Modern");
  assert.equal(formatStyleDisplay(null), "Vernacular / Contemporary");
});

test("formatPlausibilityStatus accurately returns badge variant and anomaly counts", () => {
  const validReport: PhysicalPlausibilitySummary = {
    is_plausible: true,
    collisions_count: 0,
    floating_count: 0,
  };
  const validBadge = formatPlausibilityStatus(validReport);
  assert.equal(validBadge.variant, "success");
  assert.equal(validBadge.label, "Physics Verified");

  const anomalyReport: PhysicalPlausibilitySummary = {
    is_plausible: false,
    collisions_count: 2,
    floating_count: 1,
  };
  const anomalyBadge = formatPlausibilityStatus(anomalyReport);
  assert.equal(anomalyBadge.variant, "error");
  assert.equal(anomalyBadge.label, "3 Anomalies");

  const singleAnomalyReport: PhysicalPlausibilitySummary = {
    is_plausible: false,
    collisions_count: 1,
    floating_count: 0,
  };
  assert.equal(formatPlausibilityStatus(singleAnomalyReport).label, "1 Anomaly");
  assert.equal(formatPlausibilityStatus(null).variant, "warning");
});

test("isGeoreferenced checks for ENU coordinate frame and georeferenced status", () => {
  const georefSummary: SpatialIntelligenceSummary = {
    world_id: "world-geo-1",
    coordinate_frame: "ENU",
    transforms_count: 2,
    georeference: {
      status: "georeferenced",
      frame: "ENU",
      origin: { latitude_deg: 37.77, longitude_deg: -122.42, altitude_m: 10 },
    },
  };
  assert.equal(isGeoreferenced(georefSummary), true);

  const localSummary: SpatialIntelligenceSummary = {
    world_id: "world-local-1",
    coordinate_frame: "WORLD",
    transforms_count: 0,
    georeference: null,
  };
  assert.equal(isGeoreferenced(localSummary), false);
  assert.equal(isGeoreferenced(null), false);
});
