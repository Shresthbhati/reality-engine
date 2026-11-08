/**
 * Contract tests for Architectural Section & Elevation Drawings Model.
 * Verifies drawing types, categories, scale formats, and design token styling.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  getDrawingTypeMeta,
  formatDrawingScale,
} from "../../src/lib/studio/drawing-sheet-model.ts";

test("Drawing type metadata provides accurate architectural labels and WCAG badge styling", () => {
  const longSec = getDrawingTypeMeta("section_longitudinal");
  assert.equal(longSec.label, "Section A-A (Longitudinal)");
  assert.equal(longSec.category, "section");
  assert.ok(longSec.badgeClass.includes("text-accent"));

  const transSec = getDrawingTypeMeta("section_transverse");
  assert.equal(transSec.label, "Section B-B (Transverse)");
  assert.equal(transSec.category, "section");

  const northElev = getDrawingTypeMeta("elevation_north");
  assert.equal(northElev.label, "North Elevation");
  assert.equal(northElev.category, "elevation");
  assert.ok(northElev.badgeClass.includes("text-success"));

  const eastElev = getDrawingTypeMeta("elevation_east");
  assert.equal(eastElev.label, "East Elevation");
  assert.equal(eastElev.category, "elevation");
  assert.ok(eastElev.badgeClass.includes("text-warning"));

  const customSec = getDrawingTypeMeta("custom_section");
  assert.equal(customSec.label, "Custom Section Cut");
  assert.equal(customSec.category, "section");
});

test("formatDrawingScale normalizes architectural drawing scale strings", () => {
  assert.equal(formatDrawingScale("1:50"), "1:50");
  assert.equal(formatDrawingScale("1:100"), "1:100");
  assert.equal(formatDrawingScale("200"), "1:200");
});
