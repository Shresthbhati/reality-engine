# Dataset reconstruction benchmark: infrastructure

Status: **infrastructure implemented and unit-tested on generated fixtures. No real dataset is present, downloaded or verified.
No reconstruction accuracy number exists yet.** The proposed scene selection (section 6) is `PROPOSED`, not available.

Code: `benchmarks/datasets/` (manifest, adapters, licences, splits, selection, runner) and `benchmarks/metrics/geometry.py`.
Tests: `tests/test_benchmark_geometry_metrics.py`, `tests/test_benchmark_dataset_adapters.py`, `tests/test_benchmark_dataset_runner.py`
(fixtures: `tests/benchmark_dataset_fixtures.py`). It reuses `engine/multiengine/benchmark_manifest.py` for hash, duplicate-image and
split-leakage checks; it does not replace the measured-rooms benchmark (`benchmarks/measured_rooms/`, `docs/engineering/MEASURED_ROOMS_*.md`),
which remains the only benchmark with our own physically measured ground truth.

## 1. SceneManifest (`benchmarks/datasets/manifest.py`, schema `scene_manifest.schema.json`)

One JSON document per scene. Original image paths are kept relative to the dataset root and are never copied or rewritten; each carries its
sha256 (`validate_scene` re-hashes and fails on any change). Per scene: intrinsics and camera-to-world poses per image, depth, GT mesh / point
cloud, semantic / instance labels, 3D boxes, part annotations (MultiScan), coordinate frame (units, `unit_to_metre`, up axis, camera axes),
split, group, licence record, free-form notes (including `needs export` / `UNVERIFIED` items).

Two classifications are kept **separate** and never merged into one number:

| Field | Values | Meaning |
|---|---|---|
| `annotation.provenance_class` (also `pose_provenance`, `depth_provenance`) | `measured` / `estimated` / `fitted` / `synthetic` | measured = independent physical instrument (laser scanner, tape); estimated = algorithm or human judgement from the same sensor data (SfM/SLAM poses, fused RGB-D meshes, human labels); fitted = model fitted to other data; synthetic = authored/rendered |
| `scene.data_kinds` | `synthetic` / `real_image` / `rgbd` / `physical_gt` | what the input is; `physical_gt` is only allowed when the scene has a `measured` annotation, and a `measured` annotation requires `physical_gt` (and is rejected in a synthetic scene) |

Aggregates are grouped by `split | data_kinds | gt_provenance`. `to_benchmark_manifest` converts to the dict
`benchmark_manifest.validate_manifest` checks, and only passes `dimension` items that are `measured`.

## 2. Adapters (`adapters.py`) and access

Adapters read LOCAL paths only: no network, no download, no archive extraction, no video decoding. Constructing one needs
`terms_acknowledged=True` (a user attestation recorded in the manifest, not a check). Layouts below were read from the official documentation
on 2026-11-23; anything not confirmed is marked UNVERIFIED in the adapter docstring and in the manifest `notes`.

| Dataset | Layout source | Frames | GT classes in the manifest | Notable UNVERIFIED |
|---|---|---|---|---|
| ScanNet | README (scene%04d_%02d, `.sens`, `_vh_clean_2.ply`, aggregation/segs json) | need user SensReader export | mesh `estimated` (BundleFusion), labels `estimated`; data kind `rgbd`; **no independent measurement** | exported-frame layout, pose direction, depth unit, camera axes |
| ScanNet++ | documentation page (data/<scene>/{scans,dslr}, split/, nerfstudio json) | DSLR undistorted + `transforms_undistorted.json` (OpenGL axes, metric) | laser point cloud `measured`, mesh `fitted`, labels `estimated`; `real_image + physical_gt` | nerfstudio field names, file_path matching; nvs_test scenes have no scan data |
| ARKitScenes | DATA.md + threedod/README (3dod layout) | `lowres_wide` + `.traj` + `.pincam` | ARKit mesh `estimated`, boxes `estimated`; `rgbd`; laser point clouds (raw) **not wired** | .traj direction (assumed world-to-camera), frame naming, camera axes |
| Structured3D | README + data_organization.md | `perspective/full/<pos>/rgb_rawlight.png`, `camera_pose.txt` | everything `synthetic`; data kind `synthetic` | none for the layout; the OpenCV pose/intrinsics are derived by us |
| 3RScan | README (`<scanId>/`, `3RScan.json`) | user must unzip `sequence.zip`; unpacked layout UNVERIFIED | mesh/labels `estimated`; `rgbd`; reference + rescans share one group | sequence file names, `_info.txt`, pose direction, frames stored rotated |
| MultiScan | docs (file system, acquired, annotation) | **mp4 only**: user decodes to `<scene>/frames/NNNNNN.png` (our convention) | mesh `estimated`, OBB + parts + articulations `estimated`; `rgbd` | frames convention / count agreement |

### What YOU must do for access (nothing here accepts terms, creates accounts or downloads)

Full text in `licences.py`. Every licence record has `commercial_use=None` (not read by this tooling): treat all six as research / internal
benchmark only until you have read the terms; several restrict commercial use and redistribution, and that matters for a commercial product.

- **ScanNet**: read the Terms of Use; send the signed agreement from an institutional address to scannet@googlegroups.com; download with the script you receive; export frames with SensReader.
- **ScanNet++**: read the terms and submit the access request on the ScanNet++ site; download the DSLR 2 MP and meshes/semantics groups (or default 1.5 TB); procedure UNVERIFIED.
- **ARKitScenes**: read the repository LICENSE (not verified here); run the official `download_data.py` for only the selected video ids (3dod is 623.4 GB in total).
- **Structured3D**: read the Terms of Use and fill the agreement form linked from the README; download the selected scenes.
- **3RScan**: read the project terms and follow its access procedure (UNVERIFIED); unzip `sequence.zip` per scan yourself.
- **MultiScan**: accept any gating on the Hugging Face dataset page (UNVERIFIED which apply); `unzip "*.zip"`; decode `.mp4` to frames.

## 3. Splits (`splits.py`)

`assign_splits(scenes, seed=..., fractions=...)` assigns whole **groups** (ScanNet space, MultiScan environment, 3RScan reference + rescans) by
`sha256(seed:group)` order with largest-remainder cuts: reproducible, independent of input order, every split non-empty when possible.
`leakage_problems` reports group straddling, unassigned scenes and everything `validate_manifest` finds (hash mismatch, same image bytes in two
scenes or splits). Official dataset splits are kept in `extra.official_split` for reference only. Choose scenes and thresholds before looking at results.

## 4. Metrics (`benchmarks/metrics/geometry.py`)

All in metres; numerically tested against analytic answers.
Chamfer (accuracy, completeness, symmetric) and F-score at several thresholds (point sets); exact point-to-triangle-surface distance and
mesh accuracy / completeness (area-weighted samples); camera registration after Sim(3) (Umeyama) alignment (ATE, rotation error, registered
fraction, recovered scale) and trajectory ATE + RPE; wall / floor / ceiling area completeness (plus spurious area) and door / window recall;
`dimension_errors`, which accepts **only** `measured` items with a finite uncertainty and lists everything else as excluded. Anything without
its required ground truth returns `{"status": "BLOCKED", "reason": ...}`, never a number.

## 5. Runner (`run_benchmark.py`)

`python -m benchmarks.datasets.run_benchmark --manifests DIR --root DATASET_ROOT --out OUT --splits test` writes `OUT/scenes/<scene>.json` and
`OUT/aggregate.json`.

Production entry: `engine.pipeline.vertical_slice.vertical_slice(evidence_items, VerticalSliceOptions())`, the function `apps/api/jobs.py`
calls in the multi-view rung, fed `EvidenceItem(id, PHOTO, source_uri=file URI, sha256)` as the job builds them. **Not exercised**: the
progressive ladder's lower rungs, the API's open-vocabulary detectors / dense options, evidence-quality gating, persistence. GT intrinsics and
poses are never given to the pipeline. Importing the real entry pulls torch (about 20 s).

Statuses: `PASSED` = the pipeline ran, returned a world with >= 1 registered camera, **and** at least one metric was computed against GT (this
means "processed and measured", not "accurate"; there are no thresholds); `BLOCKED` = not scored, with a reason (invalid or missing data, fewer
than 3 images, pipeline unavailable, or processed but nothing scorable); `FAILED` = the pipeline raised or returned nothing usable. Rotation
error is withheld when the dataset's camera-axes convention is unknown. The aggregate states `production_pipeline`; a run with an injected
pipeline carries a warning and must not be quoted as a reconstruction result.

Not wired yet (reported BLOCKED): predicted-mesh extraction for mesh accuracy / completeness, WorldIR planar structure and opening
extraction (`structure_extractor` hook), named dimension extraction (`dimension_extractor` hook), bounded GT structure elements (Structured3D
planes are unbounded; no dataset here provides independent room dimensions).

## 6. Initial proposed selection: PROPOSED, NOT AVAILABLE

| Dataset | Proposed count | Status |
|---|---|---|
| ScanNet | 30 scenes | PROPOSED |
| ScanNet++ | 15 scenes | PROPOSED |
| ARKitScenes | 20 scenes | PROPOSED |
| MultiScan | 10 scenes | PROPOSED |
| Structured3D | 30 scenes (synthetic: report separately) | PROPOSED |
| 3RScan | 10 reference + rescan pairs | PROPOSED |

Nothing is downloaded, hashed or split. Scene ids are chosen later, by a fixed seeded rule per group, from the licensed data
(`selection.py`: `AVAILABLE = False`).

## 7. Known limits

- Only ScanNet++ (laser point cloud) offers independent physical GT among these datasets; for ScanNet, ARKitScenes, 3RScan and MultiScan the
  GT is produced from the same sensor data, so agreement with it measures consistency with another algorithm, not metric truth. Dimension error
  is BLOCKED for all of them.
- Pose conventions and the GT-mesh-in-pose-frame assumption are verified only where stated; metrics carry the caveat in their output.
- None of the dataset terms were read in full; commercial-use status is unknown for all six.
