import test from "node:test";
import assert from "node:assert/strict";

import {
  BLUEPRINT_STAGE_LABELS,
  getStageLabel,
  cardView,
  type PlaceCardInput,
} from "../../src/lib/places.ts";
import {
  PIPELINE_DEFS,
  buildPipelineNodes,
} from "../../src/lib/api/pipeline-graph.ts";

test("BLUEPRINT_STAGE_LABELS matches the 14 pipeline stages from 12_PIPELINE.md", () => {
  assert.equal(BLUEPRINT_STAGE_LABELS["s1_ingest"], "Checking photos");
  assert.equal(BLUEPRINT_STAGE_LABELS["s2_grouping"], "Finding photos of the same place");
  assert.equal(BLUEPRINT_STAGE_LABELS["s3_sfm"], "Calculating camera positions");
  assert.equal(BLUEPRINT_STAGE_LABELS["s4_scale"], "Working out real-world scale");
  assert.equal(BLUEPRINT_STAGE_LABELS["s5_dense"], "Building room geometry");
  assert.equal(BLUEPRINT_STAGE_LABELS["s7_perception"], "Recognising and reconstructing objects");
  assert.equal(BLUEPRINT_STAGE_LABELS["s11_diagnostics"], "Checking the result");
  assert.equal(BLUEPRINT_STAGE_LABELS["s13_completeness"], "Finding what's missing");
  assert.equal(BLUEPRINT_STAGE_LABELS["s14_saving"], "Saving the world");
});

test("getStageLabel returns Blueprint label or clean humanized fallback", () => {
  assert.equal(getStageLabel("checking_reconstruction_backend"), "Checking photos");
  assert.equal(getStageLabel("reconstructing"), "Calculating camera positions");
  assert.equal(getStageLabel("committing_version"), "Saving the world");
  assert.equal(getStageLabel("custom_neural_feature_pass"), "Custom Neural Feature Pass");
  assert.equal(getStageLabel(null), "Preparing");
});

test("cardView reflects honest stageLabel when job is active", () => {
  const input: PlaceCardInput = {
    photo_count: 24,
    current_version_id: "v-1",
    active_job: {
      status: "running",
      stage: "reconstructing",
      progress: 60,
    },
  };
  const view = cardView(input, Date.now());
  assert.ok(view.building);
  assert.equal(view.building?.step, "Building the 3D place");
  assert.equal(view.stageLabel, "Calculating camera positions");
  assert.equal(view.building?.percent, 60);
});

test("PIPELINE_DEFS and buildPipelineNodes carry blueprint_label for user clarity", () => {
  const nodes = buildPipelineNodes({ job: null });
  assert.equal(nodes.length, 5);
  assert.equal(nodes[0].blueprint_label, "Checking photos");
  assert.equal(nodes[1].blueprint_label, "Checking photos");
  assert.equal(nodes[2].blueprint_label, "Finding photos of the same place");
  assert.equal(nodes[3].blueprint_label, "Calculating camera positions");
  assert.equal(nodes[4].blueprint_label, "Saving the world");
});
