import test from "node:test";
import assert from "node:assert/strict";
import {
  type GeoAnchor,
  formatGeoAnchorCoordinates,
  isIndoorGpsAccuracy,
} from "../../src/lib/api/geo-anchor-model.ts";

test("geo anchor formats coordinates with cardinal directions", () => {
  const sfAnchor: GeoAnchor = {
    latitude: 37.77493,
    longitude: -122.41942,
    altitude_m: 15.0,
    heading_deg: 0,
    crs: "EPSG:4326",
    accuracy_m: 4.5,
    source: "exif_gps",
  };

  const formatted = formatGeoAnchorCoordinates(sfAnchor);
  assert.equal(formatted, "37.77493° N, 122.41942° W");

  const sydneyAnchor: GeoAnchor = {
    latitude: -33.86882,
    longitude: 151.2093,
    altitude_m: 58.0,
    heading_deg: 180,
    crs: "EPSG:4326",
    accuracy_m: 8.0,
    source: "exif_gps",
  };
  assert.equal(formatGeoAnchorCoordinates(sydneyAnchor), "33.86882° S, 151.20930° E");
});

test("isIndoorGpsAccuracy flags coarse indoor GPS uncertainty", () => {
  // Blueprint 16: Indoor GPS is noisy prior (>15m)
  assert.equal(isIndoorGpsAccuracy(3.5), false);
  assert.equal(isIndoorGpsAccuracy(12.0), false);
  assert.equal(isIndoorGpsAccuracy(18.5), true);
  assert.equal(isIndoorGpsAccuracy(45.0), true);
  assert.equal(isIndoorGpsAccuracy(undefined), false);
});
