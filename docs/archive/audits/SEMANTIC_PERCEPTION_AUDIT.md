# Semantic Perception & 3D Object Understanding Audit

**Date:** 2026-09-13. Direct repository inspection (Prompt 3 Phase 0).

| Area | Status | Evidence |
|---|---|---|
| Detection (2D object detection) | **MISSING** | `perception/detection/` is a README placeholder only. No detector backend, real or adapter, exists anywhere. |
| Segmentation | PARTIAL, UNVERIFIED | `perception/segmentation/interface.py` (real interface) + `sam_backend.py` (real SAM adapter code). **Not runnable in this environment as-is**: `torch` 2.14.0 (CPU) is installed, but the `segment_anything` package and any SAM checkpoint are not — confirmed by direct import attempt. Installing/downloading those is a real, separate task (checkpoint is several hundred MB to a few GB), not attempted this session. |
| Tracking | MISSING | `perception/tracking/` is a README placeholder; `perception/instances/interface.py` defines `ITrackBackend`/`InstanceTrack` types but zero implementations. |
| 2D→3D lifting | **MISSING before this session, IMPLEMENTED now** | See below. |
| Object entity resolution (multi-view merge) | MISSING | No code merges multiple `ObjectHypothesis3D`-shaped observations of the same object across frames/views. |
| Structural perception (floor/ceiling/wall/door/window) | IMPLEMENTED (pre-existing) | `perception/geometry/planes.py` + `orientation.py` + `evidence/promote_planes.py`/`promote_rooms.py` — real, tested, unchanged this session. |
| Material perception | MISSING | `perception/materials/` is a README placeholder. |
| Ontology mapping | PARTIAL (pre-existing) | `world_ir/schema_v1.py`'s `EntityType` enum is the canonical ontology; nothing yet maps a segmentation backend's raw label string onto it (moot until a real detector/segmenter actually runs). |
| Confidence separation (model/geometric/fusion/world) | PARTIAL | `ObjectHypothesis3D.confidence` (new, below) is explicitly `segmentation_confidence * depth_valid_fraction`, both factors visible in the result rather than collapsed into an opaque number — a real, if narrow, example of the separation the campaign asks for. |
| Model disagreement | MISSING | No code represents two models producing conflicting labels for the same evidence. |

## This session's change

**`perception/instances/lifting.py`** (Phase 4, "2D → 3D") — `lift_region_to_3d(region, depth, camera)`: converts one `SegmentedRegion` (2D mask, from any `ISegmentationBackend`) plus a `DepthMap` (from any `IDepthBackend`) plus the real `PinholeCamera` (added in the prior reconstruction-hardening session) into an `ObjectHypothesis3D` — unprojects every valid-depth mask pixel, takes the resulting point cloud's centroid and AABB. Deterministic geometry composition, not a learned model — operates on backend *output types*, so it needs no SAM/detector install to run or test, the same way `evidence/promote_planes.py` doesn't need COLMAP installed to be tested against a synthetic `ReconstructionResult`.

Honesty rules enforced: a relative (non-metric) depth map is refused (`LiftingError`), not silently treated as meters; too few valid-depth mask pixels returns `None` (insufficient evidence), never a low-quality fabricated hypothesis; the result is always `Provenance.INFERRED` with a confidence that is the visible product of segmentation confidence and depth coverage, not a black-box number.

10 tests (`tests/test_2d_3d_lifting.py`): known-depth planar lift, full-vs-partial valid-depth confidence scaling, insufficient-evidence `None` path, empty-mask edge case, mismatched-evidence-id/dimensions/non-metric-depth error paths, tight AABB bounds, plain-data serialization.

**Verification:** full suite 1012 passed, 1 skipped (1002 baseline + 10), zero regressions.

## Honest gap against Prompt 3's Definition of Done

The full chain `images -> semantics -> 3D -> WorldIR` cannot be run end-to-end this session: there is no real detection backend at all, and the one real segmentation backend (SAM) cannot execute in this environment (missing `segment_anything` package and model checkpoint — a real installation/download task, not a code gap). This session's lifting module is real and tested against hand-built fixtures shaped exactly like real backend output, but it has never consumed an actual SAM/detector result. That remains the honest, named blocker for Phases 1-2 and, transitively, for exercising Phase 4 on real backend output rather than fixtures.
