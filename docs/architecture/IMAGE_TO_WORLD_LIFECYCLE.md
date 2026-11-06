# Reality Engine — Image-to-World Lifecycle

**Status of this document:** target process + evidence-based status per step.
The target describes what Reality Engine should do from one uploaded photograph to a stored,
displayed, editable 3D World. It is **not** a claim that every step exists. Each step carries a
status taken from the code and from recorded runs (`docs/audits/REALITY_ENGINE_FINAL_ACCEPTANCE_REPORT.md`,
2026-11-05; later fixes in `docs/audits/REALITY_ENGINE_REPAIR_ROADMAP.md`).

Companion docs (not duplicated here): stage order and output set in `PIPELINE.md`; stage contract in
`docs/blueprint/12_PIPELINE.md`; call graph in `docs/audits/REALITY_ENGINE_EXECUTION_GRAPH.md`.

## Status key

| Status | Meaning |
|---|---|
| **WORKING** | Real implementation, called by the product path, observed in a recorded run |
| **UNINTEGRATED** | Code exists; no production caller found (or the caller cannot reach it) |
| **PARTIAL** | Works with a stated limit, or is heuristic |
| **UNAVAILABLE** | Implemented but blocked here by a missing dependency/environment |
| **MISSING** | No implementation found in the repo |

"Not verified" means exactly that: I did not run it.

## What one image can and cannot give

A single photograph supports analysis, monocular *relative* depth and a visible-surface hypothesis.
It cannot give a complete or metric environment. `engine/pipeline/single_image.py` encodes this: output is
LEVEL 0, scale `RELATIVE`, confidence capped at 0.35, unseen surfaces recorded as UNKNOWN, camera from
EXIF 35 mm focal length or an *assumed* 60° HFOV (recorded on the world). `engine/pipeline/progressive.py`
defines the cumulative levels the product may claim, and only the level actually achieved:

| Level | Meaning | Needs |
|---|---|---|
| 0 | single-image visual hypothesis | 1 photo |
| 1 | rough multi-image (SfM placed part of the set) | overlapping views |
| 2 | sparse photogrammetric (all images placed) | overlap + parallax |
| 3 | dense (dense MVS contributed points) | ≥ gate in `engine/pipeline/dense_gate.py` |
| 4 | semantic/topological refinement | evidence for rooms/storeys |
| 5 | incremental refinement | prior world + new evidence |

Metric scale additionally needs a measurement (known distance, RGB-D/LiDAR, calibrated target, GNSS).

---

## Part A — Per-image processing

### Stage 1 — Intake and asset creation (steps 1–7)

| Step | Target | Status | Evidence / gap |
|---|---|---|---|
| 1 | Receive, record name/size/MIME/time/owner/session/asset id; keep the original | **WORKING** | `POST /api/reconstructions` → evidence rows (acceptance #1: 32 files → 202 in 4.1 s) |
| 2 | Signature check, size/pixel limits, malformed data, bombs | **PARTIAL** | Magic-byte typing, never extension/MIME (`apps/api/routes_misc.py` `_classify`), `UPLOAD_MAX_BYTES`, per-session file cap, 413 on overflow. **No explicit pixel cap**: decode errors (incl. bombs) become `ok=False` in `evidence/image_check.py:113`, relying on Pillow's default limit |
| 3 | Decode; record w/h, channels, bit depth, alpha, colour profile | **PARTIAL** | `inspect_image` records width/height only. HEIC/WebP support depends on the Pillow build — not verified |
| 4 | Apply EXIF orientation, keep original EXIF | **MISSING** in the geometry path | `ImageOps.exif_transpose` is used only in `apps/api/place_summary.py`; `inspect_image` does not, so reported size is pre-rotation. COLMAP reads the raw pixels |
| 5 | Extract camera/lens/focal/exposure/ISO/time/GPS; never invent | **PARTIAL** | `image_check.py` keeps make, model, `focal_length_35mm`, GPS (only if present). The folder importer (`evidence/importers.py::_decode_photo_exif`) reads time and more. Lens/aperture/ISO are not carried into the API evidence row |
| 6 | Derivatives (thumb, analysis sizes, full-res) with pixel mapping | **MISSING** | Only an internal 256 px copy for statistics; no stored derivative set or coordinate map |
| 7 | Hash + similarity descriptors; flag but do not auto-drop duplicates | **PARTIAL** | Exact duplicate via sha256 is flagged (`quality_checks.evaluate_checks` "Duplicate" → `rejected`). Perceptual hashes exist (`evidence/dedup.py`: `compute_phash/dhash`) but have **no production caller** — **UNINTEGRATED**. Near-duplicate detection is therefore not active on upload |

### Stage 2 — Image quality assessment (steps 8–13)

| Step | Target | Status | Evidence / gap |
|---|---|---|---|
| 8 | Resolution / usable detail | **WORKING** | MP check (`RESOLUTION_FAIL_MP 0.25`, `WARN 1.0`), policy values not calibrated. Ground-sample detail: `perception/quality/assessment.py` + `detail_budget.py` (needs intrinsics, set-level) |
| 9 | Blur (motion/defocus/local) | **PARTIAL** | One global Laplacian variance on a 256 px copy; **relative ranking only**, not a blur model, no local map |
| 10 | Exposure, clipping, uneven light, backlight | **MISSING** | `quality_checks.py` states exposure is not measured |
| 11 | Noise, compression, banding | **MISSING** | none |
| 12 | Viewpoint usefulness | **PARTIAL** | Set-level only: view-angle diversity, coverage degrees, "adds a new viewpoint / mostly repeats" (acceptance #3, heuristic) |
| 13 | Separate scores per task (recognition, matching, depth, geometry, measurement) | **MISSING** | One 0–100 score, documented as "NOT a probability". Statuses accepted / accepted_with_warning / needs_review / rejected exist; there is no "unresolved" and no per-task score |

### Stage 3 — Initial image understanding (steps 14–20)

| Step | Target | Status | Evidence / gap |
|---|---|---|---|
| 14 | Scene classification | **MISSING** | no scene classifier in `perception/` |
| 15 | Semantic segmentation | **UNAVAILABLE** | `OneFormer`/`SAM` backends raise `SegmentationBackendUnavailableError` (no fabricated masks); `segment-anything` not installed in the venv |
| 16 | Object detection + instance masks | **PARTIAL / UNAVAILABLE** | Mask R-CNN (COCO classes) ran as a component probe only: 10 detections in 2/8 photos; never reached a World. Open-vocabulary (`perception/detection/open_vocab.py`) needs `transformers`, not installed. Detector choice (OWLv2 vs Grounding DINO) is an open user decision |
| 17 | Fine-grained subtypes, no forced labels | **UNINTEGRATED** | `perception/fine_grained/*` exists; no real-data run reached it |
| 18 | Corners/edges/contours/landmarks | **PARTIAL** | Features come from COLMAP SIFT inside SfM; no 2D contour/junction detector in the pipeline |
| 19 | Structural candidates from the image | **PARTIAL** | Planes/walls/floors/openings are inferred from the **3D** point cloud (`perception/geometry/planes.py`, `perception/architecture/*`), not from image-space hypotheses. Heuristic; credible-looking rooms/corridors were produced from an exterior facade (acceptance #11) |
| 20 | 2D relationships (in front of, within, beside) | **MISSING** | none |

### Stage 4 — Camera understanding and geometric preparation (steps 21–30)

| Step | Target | Status | Evidence / gap |
|---|---|---|---|
| 21 | Camera model knowledge | **PARTIAL** | `world_ir/camera.py::CameraIntrinsics`; EXIF-only focal is treated as a hint |
| 22 | Intrinsics, with source recorded (measured/calibrated/EXIF/estimated) | **PARTIAL** | API derives fx from EXIF 35 mm focal with basis `exif_focal_length_35mm` (`apps/api/jobs.py:369-374`); CLI takes manifest intrinsics. COLMAP's own estimate is not yet fed back to depth/perception on a first run (roadmap P1-7, open). No calibration workflow |
| 23 | Lens-distortion correction with coordinate mapping | **MISSING** | Outside COLMAP's own undistorter (dense only) there is no undistort step; the CLI run explicitly "ignored distortion" |
| 24 | Optional, evidence-preserving enhancement | **MISSING (by design safe)** | No enhancement exists, so original pixels always reach SfM |
| 25 | Image pyramids | **MISSING** | no engine-side pyramid; resolution is whatever COLMAP's extractor does |
| 26 | Local features | **WORKING** | COLMAP SIFT, robust mode (affine shape + domain-size pooling) in `colmap_backend.py` |
| 27 | Learned descriptors | **PARTIAL** | `reconstruction/matching/lightglue.py` backs scene grouping only. XFeat/DeDoDe previously ran AKAZE/BRISK under their names; now truthfully labelled with honest probes. Not used in the main SfM path |
| 28 | 2D lines, vanishing points, planar regions | **MISSING** | no vanishing-point or line code in the repo |
| 29 | Monocular depth, relative vs metric kept distinct | **UNAVAILABLE here** | `perception/depth/*` (MiDaS, DPT, Depth-Anything, metric backend). `timm` not installed → depth fails on this machine. Metric-by-alignment to the SfM fit is the only metric route |
| 30 | A "prepared image asset" bundling all of the above | **MISSING** | data is spread over evidence rows, `metadata_json["image_facts"]` and per-stage artifacts; no single per-image record |

---

## Part B — Multi-image reconstruction

### Stage 5 — Grouping and cross-view matching (steps 31–38)

| Step | Target | Status | Evidence / gap |
|---|---|---|---|
| 31 | Decide which images share a capture area | **PARTIAL** | Visual grouping: `reconstruction/scene_identity.py::group_images` (SIFT or LightGlue pair verifier). Time/GPS grouping: `evidence/session_grouping.py` — **UNINTEGRATED** (exported, no caller) |
| 32 | Candidate pairs (retrieval, GPS, sequence) | **PARTIAL** | Only COLMAP `exhaustive_matcher` is used (`colmap_backend.py`) — O(n²); no vocab-tree/sequential/spatial matching. Fine for tens of images, a ceiling beyond that |
| 33 | Feature matching | **WORKING** | COLMAP matching, optional guided matching |
| 34 | Geometric verification / outlier rejection | **WORKING** (opaque) | COLMAP two-view verification; not separately reported |
| 35–37 | Epipolar geometry, relative motion, pair consistency | **WORKING** (inside COLMAP) | not exposed as artifacts |
| 38 | Image connectivity graph | **PARTIAL** | Not exposed. COLMAP may emit several sub-models; the backend merges them (`_parse_sparse_dir`) and reports registered/unplaced images |

### Stage 6 — Camera poses and global reconstruction (steps 39–48)

| Step | Target | Status | Evidence / gap |
|---|---|---|---|
| 39–44 | Initial pair, registration (PnP), triangulation, tracks, bundle adjustment | **WORKING** | COLMAP `mapper`. Observed: South Building 22/32 registered via API (69 %), 25/32 via CLI; 5,396 sparse points; 118 s |
| 45 | Outlier removal with recorded reasons | **PARTIAL** | COLMAP filters; rejected data not recorded as such |
| 46 | Re-register difficult images | **PARTIAL** | Incremental path adds images then runs `bundle_adjuster`, falling back to a full mapper (`colmap_backend.py`). Registration-rate improvement (ALIKED/LightGlue, GLOMAP) is open, roadmap P1-8 |
| 47 | Iterative refinement | **WORKING** (COLMAP) | — |
| 48 | Completeness metrics | **PARTIAL** | `reconstruction/completeness.py` (registration ratio, mean track length, redundancy), `scene_identity.diagnose/capture_quality`. Surfacing of reprojection error in the report: not verified |

### Stage 7 — Scale, orientation, coordinate frames (steps 49–53)

| Step | Target | Status | Evidence / gap |
|---|---|---|---|
| 49 | Scale from a measurement | **PARTIAL** | `reconstruction/scale/base.py::anchor_metric_scale`, `metric_anchor.py`; honest `relative` fallback. On the only real set the world stayed **relative** |
| 50 | Orientation (gravity, ground plane) | **PARTIAL** | `reconstruction/frame.py` canonicalises from the dominant plane (+Y up). No IMU/gravity input |
| 51 | Local frame | **WORKING** | see `COORDINATE_FRAMES.md` |
| 52 | Georeference (ECEF/ENU) | **UNINTEGRATED / not verified** | `reconstruction/scale/georeferencing.py` (Umeyama Sim3, RANSAC); `evidence/geo_anchor.py` is only re-exported, with no production caller (roadmap #9). No real GNSS run |
| 53 | Record source, residuals, uncertainty | **PARTIAL** | scale state and diagnostics exist; end-to-end uncertainty propagation not verified |

### Stage 8 — Dense geometry (beyond step 53)

| Capability | Status | Evidence / gap |
|---|---|---|
| Gate: dense runs only if justified | **WORKING** | `engine/pipeline/dense_gate.py`: ≥ 80 % registered, median track ≥ 3, coverage ≥ 15°, ≥ 300 structure points (EXPERIMENTAL thresholds; env override). Acceptance: refused at 69 %. A wrong "pass" is judged afterwards (`dense_judge.py`) |
| Dense MVS | **PARTIAL** | COLMAP `patch_match_stereo` → `stereo_fusion` (`reconstruction/dense_pipeline.py`). Earlier docs claim a GPU run; the acceptance run did not reproduce it |
| Depth fusion | **UNAVAILABLE here** | needs depth (above) |
| Surface/mesh | **WORKING** (on earlier runs) | `reconstruction/meshing/*`: Poisson via COLMAP, validation, repair, simplification (execution graph: 240k verts / 482k faces). The acceptance API run skipped meshing for lack of intrinsics — fixed since for EXIF photos, re-run not recorded |
| Texturing | **UNINTEGRATED** | `reconstruction/texturing/projection.py`; roadmap #9 (10.1): no production caller |

### Stage 9 — Semantic and architectural understanding

| Capability | Status | Evidence / gap |
|---|---|---|
| Planes → walls/floors/ceilings/openings | **PARTIAL (heuristic)** | `perception/architecture/*` → 21 walls, 3 floors, 3 ceilings, 2 windows (conf 0.05), 1 room, 1 corridor (0.90 INFERRED) from an **exterior** facade — over-claim (roadmap P1-10: add an evidence gate) |
| Rooms/levels/stairs/corridors, topology | **PARTIAL** | implemented and unit/golden tested (`perception/architecture/topology.py`, room graph); never judged on real interior data (none in repo) |
| Cross-view object identity | **UNAVAILABLE on real data** | `perception/instances/association.py`, `epipolar.py`; world had **0 objects** because detection/segmentation/depth did not run |
| 2D detection → persistent 3D entity | **not demonstrated** | no real run carried a detection into WorldIR |

### Stage 10 — World construction

| Capability | Status | Evidence / gap |
|---|---|---|
| Compile to WorldIR, validate | **WORKING** | acceptance #12, #15 (32 entities, 12 warnings, `validated: true`) |
| Provenance / evidence links | **PARTIAL** | entities carry supporting evidence ids; provenance route is coarser than the data; confidence vectors null; correction-commit loss of cameras/report/sessions fixed (roadmap §1), live re-run not recorded |
| Versioned WorldStore, corrections, diff, restart | **WORKING** | acceptance #16, #19–#23 |
| Session → World alignment; World ↔ World composition | **UNINTEGRATED / not demonstrated** | `reconstruction/merge.py`, `apps/api/routes_registration.py` exist; no real two-session alignment recorded |
| Session ≠ World invariant | **not verified end to end** | domain model in `docs/blueprint/11_WORLDIR_CONTRACT.md` |

### Stage 11 — Validation, storage, delivery

| Capability | Status | Evidence / gap |
|---|---|---|
| Honest job states (never SUCCEEDED without valid outputs) | **WORKING** | job ended PARTIAL (352 s); no false SUCCEEDED |
| Exports (glTF/GLB, PLY, OBJ, STL, IFC, USD, CityJSON, GeoJSON…) | **WORKING** | 9 formats re-parsed independently (acceptance #24–25); some entities omitted from some formats |
| Studio display | **not verified** | no browser run recorded; Simple mode hides inspector/corrections (roadmap P3) |
| Resource limits (6 GiB GPU) | **PARTIAL** | peak 5,902/6,141 MiB with an OOM warning; API holds ~5.8 GiB idle (roadmap P0-4, open) |

---

## Mapping to the consolidation gates

| Gate | Current position (evidence-based) |
|---|---|
| 0 Repository truth | Largely met by the audit set in `docs/audits/`; keep one ledger (`30_BUILD_PLAN.md` + `.agent/TASKS.yaml`) |
| 1 Foundations | World/Session/versioning/persistence **WORKING**; tenant isolation beyond `routes_worlds.py` open (P0-2) |
| 2 One reconstruction pipeline genuinely works | Sparse **WORKING** (outdoor, 22–25/32). Dense, depth, mesh, texture not demonstrated on the same run; **no real indoor dataset** |
| 3 Intelligence | Detection/segmentation/association **UNAVAILABLE**; 2D→3D entity path not demonstrated |
| 4 Composable Worlds | **not demonstrated** |
| 5 UX | not verified in a browser |
| 6 Reliability | honest failure states **WORKING**; VRAM lifetime open |
| 7 Expansion | deferred until 2–4 pass |

**Release-blocking acceptance test** (from the consolidation brief): 30–50 real photographs of one room →
quality report → cameras → sparse + dense → mesh → objects and architecture → associated across views →
structured room → persist → restart → second Session aligned without identity loss → edit → export →
exported geometry matches the stored World. **Blocked on the user for real interior captures and tape
measurements** (roadmap P2-12).

## Highest-value gaps to close first (in dependency order)

1. Install `timm`, `transformers`, `segment-anything`, `jsonschema`; re-run the acceptance journey (unblocks depth, detection, segmentation).
2. Apply EXIF orientation in `inspect_image` and the geometry input (step 4) and add an explicit pixel cap (step 2).
3. Wire perceptual-hash near-duplicate flagging (step 7) and time/GPS session grouping (step 31), or delete them.
4. Surface a per-image prepared-asset record and per-task quality scores (steps 13, 30).
5. Evidence gate before rooms/corridors are inferred from exteriors (Stage 9).
6. Real interior capture with tape measurements, then Gates 2–4.
