/**
 * Contract tests for 3D Surface-Projected & Polygon Area Tool (Epic E09-2).
 * Verifies accuracy against server-side world_ir/measure.py rules within +/- 0.1%.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  computeWallProjectedDistance,
  computePolygonMetrics,
  computeBuildingFootprint,
  computeRoadWidth,
  projectPointToPlane,
  distance,
  dot,
  cross,
  normalize,
  resolveUnitContext,
  unitFor,
  formatValue,
} from "../../src/lib/studio/measure-model.ts";

test("Euclidean 3D distance matches 3D Pythagorean theorem exactly", () => {
  const d = distance([0, 0, 0], [3, 4, 12]);
  assert.equal(d, 13);
});

test("Wall projection projects points onto vertical wall plane", () => {
  // Vertical wall facing +Z with normal [0, 0, 1] at z = 2.0
  const wallPlane = {
    origin: [0, 0, 2.0] as [number, number, number],
    normal: [0, 0, 1.0] as [number, number, number],
  };

  const p = projectPointToPlane([1.0, 3.0, 5.0], wallPlane);
  // Point should have same x, y but z = 2.0
  assert.equal(p[0], 1.0);
  assert.equal(p[1], 3.0);
  assert.equal(p[2], 2.0);
});

test("Wall-projected distance computes projected run, horizontal run, and vertical run", () => {
  // Wall facing normal [0, 0, 1] at z = 0
  const wallPlane = {
    origin: [0, 0, 0] as [number, number, number],
    normal: [0, 0, 1] as [number, number, number],
  };

  // Pick point A floating at [1, 2, 0.5] and point B at [4, 6, 1.2]
  // On wall: A' = [1, 2, 0], B' = [4, 6, 0]
  // Projected delta: dx = 3, dy = 4, dz = 0
  // Projected distance = 5.0m
  // Horizontal run dx = 3.0m
  // Vertical run dy = 4.0m
  const res = computeWallProjectedDistance([1, 2, 0.5], [4, 6, 1.2], wallPlane);

  assert.ok(Math.abs(res.projectedDistance - 5.0) < 1e-9);
  assert.ok(Math.abs(res.horizontalRun - 3.0) < 1e-9);
  assert.ok(Math.abs(res.verticalRun - 4.0) < 1e-9);
  assert.ok(Math.abs(res.perpendicularOffsets[0] - 0.5) < 1e-9);
  assert.ok(Math.abs(res.perpendicularOffsets[1] - 1.2) < 1e-9);
});

test("Wall-projected distance on angled 45-degree wall plane", () => {
  // Wall normal [1/sqrt(2), 0, 1/sqrt(2)]
  const invSqrt2 = 1 / Math.SQRT2;
  const wallPlane = {
    origin: [0, 0, 0] as [number, number, number],
    normal: [invSqrt2, 0, invSqrt2] as [number, number, number],
  };

  const res = computeWallProjectedDistance([1, 0, 0], [0, 0, 1], wallPlane);
  // P1 proj is (0.5, 0, -0.5) and P2 proj is (-0.5, 0, 0.5), distance is sqrt(2)
  assert.ok(Math.abs(res.projectedDistance - Math.SQRT2) < 1e-9);
});

test("3D Polygon area matches server-side Newell method within 0.1%", () => {
  // 1. Horizontal rectangle: 4m x 3m = 12 m^2
  const rect = computePolygonMetrics([
    [0, 0, 0],
    [4, 0, 0],
    [4, 0, 3],
    [0, 0, 3],
  ]);
  assert.ok(Math.abs(rect.area - 12.0) < 1e-9);
  assert.ok(Math.abs(rect.perimeter - 14.0) < 1e-9);
  assert.ok(rect.isPlanar);

  // 2. Tilted triangle with known exact area
  // Vertices (0,0,0), (2,0,0), (0,2,2)
  // Cross product of (2,0,0) and (0,2,2) is (0, -4, 4), norm = sqrt(32) = 4*sqrt(2)
  // Area = 0.5 * 4 * sqrt(2) = 2 * sqrt(2) ~ 2.828427...
  const tri = computePolygonMetrics([
    [0, 0, 0],
    [2, 0, 0],
    [0, 2, 2],
  ]);
  const expectedArea = 2 * Math.SQRT2;
  const relError = Math.abs(tri.area - expectedArea) / expectedArea;
  assert.ok(relError < 0.001, `Relative error ${relError} must be < 0.1%`);
  assert.ok(tri.isPlanar);
});

test("Non-planar polygon fails validation honestly", () => {
  // 4 vertices where 4th vertex is warped off-plane by 0.5m
  assert.throws(() => {
    computePolygonMetrics([
      [0, 0, 0],
      [4, 0, 0],
      [4, 0, 3],
      [0, 0.5, 3], // warped out of plane
    ]);
  }, /Non-planar polygon/);
});

test("Degenerate collinear polygon is rejected", () => {
  assert.throws(() => {
    computePolygonMetrics([
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
    ]);
  }, /Degenerate polygon/);
});

test("Building footprint computes 2D ground area and bounds", () => {
  // L-shaped footprint
  const footprint = computeBuildingFootprint([
    [0, 0, 0],
    [10, 0, 0],
    [10, 0, 4],
    [5, 0, 4],
    [5, 0, 8],
    [0, 0, 8],
  ]);
  // Total area = (10 * 4) + (5 * 4) = 40 + 20 = 60 m^2
  assert.ok(Math.abs(footprint.area - 60.0) < 1e-9);
  assert.equal(footprint.width, 10);
  assert.equal(footprint.depth, 8);
  assert.equal(footprint.perimeter, 10 + 4 + 5 + 4 + 5 + 8);
});

test("Road width computes perpendicular distance across curb edges", () => {
  // Baseline curb along X axis from [0, 0, 0] to [10, 0, 0]
  // Opposite curb point at [5, 0, 7.5]
  // Perpendicular width must be 7.5m
  const width = computeRoadWidth([0, 0, 0], [10, 0, 0], [5, 0, 7.5]);
  assert.ok(Math.abs(width - 7.5) < 1e-9);
});

test("Honesty rules: scale unknown never yields metres", () => {
  const ctx = resolveUnitContext({
    metadata: { scale: { state: "unknown" } },
    entities: { e1: { provenance: "RECONSTRUCTED" } },
  });
  assert.equal(ctx.basis, "unknown");
  assert.equal(ctx.lengthUnit, "model units");
  assert.equal(unitFor("area", ctx), "model units²");
  assert.equal(formatValue(12.5, unitFor("area", ctx)), "12.500 model units²");
});
