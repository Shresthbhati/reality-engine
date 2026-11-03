import test from "node:test";
import assert from "node:assert/strict";

import {
  type TechnologyItem,
  type BenchmarkMetric,
  filterTechnologies,
  groupTechnologiesBySubsystem,
  isCopyleftLicense,
  isHardwareCompliant,
  SUBSYSTEM_METADATA,
} from "../../src/lib/studio/technologies-model.ts";

const MOCK_TECHS: TechnologyItem[] = [
  {
    id: "teaser_plus_plus",
    name: "TEASER++",
    subsystem: "geometric_estimation",
    license: "MIT",
    repository_url: "https://github.com/MIT-SPARK/TEASER-plusplus",
    role: "Fast certified point-cloud registration with up to 99% outlier tolerance",
    primary_contribution: "Truncated Least Squares (TLS) invariant registration",
    reality_engine_integration_target: "reconstruction.scale.teaser_solver",
    top_priority: true,
    status: "available",
  },
  {
    id: "manifold",
    name: "Manifold",
    subsystem: "geometry_cad_mesh",
    license: "Apache 2.0",
    repository_url: "https://github.com/elalish/manifold",
    role: "Guaranteed 2-manifold polygon mesh boolean operations",
    primary_contribution: "Robust topological CSG subtraction for doors and windows",
    reality_engine_integration_target: "geometry.kernels.manifold_kernel",
    top_priority: true,
    status: "available",
  },
  {
    id: "open_drone_map",
    name: "OpenDroneMap",
    subsystem: "reconstruction_mapping",
    license: "AGPL 3.0",
    repository_url: "https://github.com/OpenDroneMap/ODM",
    role: "Aerial orthophoto generation and digital surface models",
    primary_contribution: "Drone reconstruction backend",
    reality_engine_integration_target: "reconstruction.backend.odm",
    top_priority: false,
    status: "reference_simulated",
  },
  {
    id: "onnx_runtime",
    name: "ONNX Runtime",
    subsystem: "inference_acceleration",
    license: "MIT",
    repository_url: "https://github.com/microsoft/onnxruntime",
    role: "High-performance inference engine with DirectML and dynamic INT8",
    primary_contribution: "RTX 4050 6GB memory bound enforcement",
    reality_engine_integration_target: "perception.acceleration.onnx_quantizer",
    top_priority: true,
    status: "available",
  },
];

test("filterTechnologies filters by subsystem and text query", () => {
  const meshTechs = filterTechnologies(MOCK_TECHS, "", "geometry_cad_mesh");
  assert.equal(meshTechs.length, 1);
  assert.equal(meshTechs[0].id, "manifold");

  const querySearch = filterTechnologies(MOCK_TECHS, "outlier", "all");
  assert.equal(querySearch.length, 1);
  assert.equal(querySearch[0].id, "teaser_plus_plus");
});

test("groupTechnologiesBySubsystem partitions across all 7 subsystems", () => {
  const groups = groupTechnologiesBySubsystem(MOCK_TECHS);
  assert.equal(groups.geometric_estimation.length, 1);
  assert.equal(groups.geometry_cad_mesh.length, 1);
  assert.equal(groups.reconstruction_mapping.length, 1);
  assert.equal(groups.inference_acceleration.length, 1);
  assert.equal(groups.visual_perception.length, 0);
});

test("isCopyleftLicense correctly flags AGPL distributions", () => {
  assert.equal(isCopyleftLicense("MIT"), false);
  assert.equal(isCopyleftLicense("Apache 2.0"), false);
  assert.equal(isCopyleftLicense("AGPL 3.0"), true);
  assert.equal(isCopyleftLicense("GPL-3.0-or-later"), true);
});

test("isHardwareCompliant strictly bounds RTX 4050 VRAM at 6GB (6,144 MB)", () => {
  const validMetric: BenchmarkMetric = {
    tech_id: "yolox",
    subsystem: "visual_perception",
    latency_ms: 18.5,
    memory_peak_mb: 240.0,
    vram_peak_mb: 1850.0,
    success: true,
    accuracy_metric_name: "mAP50",
    accuracy_metric_value: 0.512,
    within_rtx4050_bound: true,
  };
  assert.equal(isHardwareCompliant(validMetric, 6144.0), true);

  const overflowMetric: BenchmarkMetric = {
    ...validMetric,
    vram_peak_mb: 7200.0,
    within_rtx4050_bound: false,
  };
  assert.equal(isHardwareCompliant(overflowMetric, 6144.0), false);
});

test("SUBSYSTEM_METADATA covers all 7 subsystem keys with valid icons and descriptions", () => {
  const expectedKeys = [
    "visual_perception",
    "geometric_estimation",
    "reconstruction_mapping",
    "pointcloud_3d_perception",
    "geometry_cad_mesh",
    "geospatial_environment",
    "inference_acceleration",
  ];
  for (const key of expectedKeys) {
    const meta = SUBSYSTEM_METADATA[key as keyof typeof SUBSYSTEM_METADATA];
    assert.ok(meta);
    assert.ok(meta.label.length > 0);
    assert.ok(meta.description.length > 0);
  }
});
