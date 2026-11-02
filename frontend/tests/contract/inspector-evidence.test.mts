import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildEntityEvidence, buildDimensionEvidence, buildColourEvidence, buildHypothesesEvidence,
  hasConflictingHypotheses, isEvidenceTarget,
  fmtUncertainty, fmtConfidence, NOT_RECORDED,
} from "../../src/lib/viewport/inspector-evidence-model.ts";

test("full evidence is carried through verbatim", () => {
  const ev = buildEntityEvidence({
    id: "wall-1", type: "wall", confidence: 0.82, provenance: "OBSERVED",
    custom_properties: {
      dimensions: { length_m: { value: 4.2, unit: "m", state: "measured", basis: "plane fit", uncertainty_m: 0.05 } },
      appearance: { color_rgb: [200, 190, 180], color_state: "measured", views_used: ["IMG_1", "IMG_2"], coverage: 0.7, texture: { uri: "a.png" } },
    },
  });
  assert.equal(ev.isTarget, true);
  assert.deepEqual(ev.dimensions[0], { key: "length_m", value: 4.2, unit: "m", state: "MEASURED", basis: "plane fit", uncertaintyM: 0.05 });
  assert.equal(ev.confidence, 0.82);
  assert.equal(ev.provenance, "OBSERVED");
  assert.equal(ev.colour.state, "measured");
  assert.equal(ev.colour.hasTexture, true);
  assert.deepEqual(ev.supportingViews, ["IMG_1", "IMG_2"]);
});

test("missing data is null / not recorded, never guessed", () => {
  const ev = buildEntityEvidence({ id: "d1", type: "door" });
  assert.deepEqual(ev.dimensions, []);
  assert.equal(ev.confidence, null);
  assert.equal(ev.provenance, null);
  assert.equal(ev.colour.state, "not_measured");
  assert.equal(ev.colour.rgb, null);
  assert.deepEqual(ev.supportingViews, []);
  assert.equal(fmtConfidence(ev.confidence), NOT_RECORDED);
  assert.equal(fmtUncertainty(null), NOT_RECORDED);
});

test("dimension without state/basis/uncertainty keeps them null; unknown state is not promoted", () => {
  const [d] = buildDimensionEvidence({ dimensions: { height_m: { value: 2.4, state: "GREAT" } } });
  assert.equal(d.state, null);
  assert.equal(d.basis, null);
  assert.equal(d.uncertaintyM, null);
  assert.deepEqual(buildDimensionEvidence({ dimensions: { x: { unit: "m" }, y: "bad" } }), []);
});

test("colour: measured state without rgb, or not_measured with rgb, is not measured", () => {
  assert.equal(buildColourEvidence({ appearance: { color_state: "measured" } }).state, "not_measured");
  const c = buildColourEvidence({ appearance: { color_rgb: [1, 2, 3], color_state: "not_measured" } });
  assert.equal(c.state, "not_measured");
  assert.equal(c.rgb, null);
});

test("evidence targets", () => {
  for (const t of ["wall", "door", "window", "floor", "ceiling"]) assert.equal(isEvidenceTarget(t), true);
  assert.equal(isEvidenceTarget("furniture"), false);
  assert.equal(isEvidenceTarget(undefined), false);
});

test("hypotheses: preserves candidate alternatives and surfaces conflicts", () => {
  const ev = buildEntityEvidence({
    id: "obj-1",
    type: "structure",
    custom_properties: {
      hypotheses: {
        type: {
          active_id: "h1",
          has_conflict: true,
          candidates: [
            { id: "h1", source_engine: "sam2", value: "chair", confidence: 0.85, evidence_count: 3, is_active: true },
            { id: "h2", source_engine: "mask_rcnn", value: "table", confidence: 0.72, evidence_count: 2, is_active: false },
          ],
        },
      },
    },
  });

  assert.equal(ev.hypotheses.length, 1);
  assert.equal(ev.hypotheses[0].domain, "type");
  assert.equal(ev.hypotheses[0].hasConflict, true);
  assert.equal(ev.hypotheses[0].activeId, "h1");
  assert.equal(ev.hypotheses[0].candidates.length, 2);
  assert.equal(ev.hypotheses[0].candidates[0].value, "chair");
  assert.equal(ev.hypotheses[0].candidates[0].isActive, true);
  assert.equal(hasConflictingHypotheses(ev), true);
});
