# ML Reconstruction, Overlap Intelligence & Export: Execution Plan

**Date:** 2026-11-01 · **Status:** PLAN ONLY, nothing implemented · **Branch:** `claude/reality-engine-ml-reconstruction-c6a6fe`

Goal: every photo is evidence of **one** physical world. No duplicated walls, no image-space overlap mistaken for intersection, no invented geometry labelled as observed. Fast, measurably accurate, and a UI that shows rather than tells.

---

## How to use this plan (brief for the executing coding agent)

You are executing this plan in the **Reality Engine** repo (Python backend + Next.js frontend). This document is self-contained: you do not need the conversation it came from. Read it fully before starting.

**What the user asked for, in their words:** systems that make accurate reconstruction possible ("engines, transformers, for all I care"), "super super super accurate" reconstruction of everything, fast ("shouldn't take too much time"), a UI with **fewer words and a better view**, and a fix for **export, which is not working**.

**Before writing code:**
1. Read `CLAUDE.md` (repo root). It is binding: inspect before changing, no fake completion, verify by running things, keep diffs focused, never claim a result you did not observe.
2. Read `docs/implementation/PENDING_IMPLEMENTATION.md`, `docs/implementation/IMPLEMENTATION_STATUS.md`, `docs/engineering/DESIGN_DECISIONS.md` and `docs/implementation/SCENE_IDENTITY.md`. Do not relitigate settled decisions.
3. Re-verify §0 below against the code. It was accurate on 2026-11-01 but the repo moves.
4. Check whether the `room-reconstruction-phase-7` work (open-vocabulary detection, `perception/entities/`) has been merged to `main`. If not, merge or port it before Phase 1 instead of rebuilding it.

**Execution rules:**
- Work in the order of §9. Finish, test and measure each step before starting the next.
- **Preserve working code.** Extend `scene_identity.py`, `object_resolution.py`, `entity_reid.py`, `incremental.py`, `guidance.py`, COLMAP orchestration, WorldIR and WorldStore. Do not replace them without a measured reason written into the PR.
- All new WorldIR fields are **additive**. Existing exports, tests and stored worlds must keep working.
- Every model sits behind an interface with a classical fallback (pattern: `perception/depth/interface.py`, `perception/segmentation/interface.py`). Never hard-require the GPU.
- Download model weights only from official sources (Hugging Face org of the model authors, or the PyTorch hub). Record name, version, licence and SHA in a model registry (`perception/model_registry.py` exists; extend it).
- Report thresholds and results separately. Never tune a threshold on the test split.
- Commit per step on a branch off `main`, with one PR per phase. No attribution lines in commits or PRs.

**How to run things (Windows):**
| What | Command |
|---|---|
| Backend tests | `python -m pytest tests -q` (focused: `python -m pytest tests/test_export_route.py -q`) |
| Full app (API :8100 + UI :3100) | `start.bat`, stop with `start.bat stop` (logs in `logs/api.log`) |
| API only, isolated data | `DATABASE_URL=sqlite+aiosqlite:///./data/dev.db STORAGE_ROOT=./data/dev_artifacts WORLDSTORE_ROOT=./data/dev_worldstore python -m uvicorn apps.api.main:app --port 8100` |
| Frontend type-check / build | `npm --prefix frontend run build` (in a git worktree, `frontend/node_modules` may be missing; run `npm --prefix frontend install` or work in the main checkout) |
| Frontend lint | `npm run lint` (root `package.json` proxies to `frontend/`) |
| Sample data | `datasets/south_building/images` (outdoor COLMAP sample, the only bundled set). Indoor test rooms come from D3 (§7). |

Before starting the API, check **port 8100 is free** (`netstat -ano | findstr :8100`). A stale API process held it on 2026-11-01.

**Definition of done for every step:** code + tests pass + measured result recorded + behaviour seen running in the real app (UI steps: open the browser and look) + `PENDING_IMPLEMENTATION.md` / `IMPLEMENTATION_STATUS.md` status advanced with evidence (MISSING → FOUNDATION → PARTIAL → EXPERIMENTAL → IMPLEMENTED → VERIFIED). If something cannot be finished, mark it PARTIAL/BLOCKED with the reason. Never mark it done.

---

## 0. What already exists (audited 2026-11-01, read-only)

| Capability | Where | State | Gap |
|---|---|---|---|
| Image grouping by verified correspondences (SIFT + epipolar RANSAC) | `reconstruction/scene_identity.py` | IMPLEMENTED, tested | Only classical SIFT; no learned matcher |
| COLMAP SfM / MVS orchestration, sub-model merge | `reconstruction/orchestrator.py`, `merge.py`, `colmap_session.py` | IMPLEMENTED | `reconstruction/{features,matching,sfm,registration}/` are **README-only stubs** |
| Cross-view depth consistency gate | `reconstruction/consistency.py` | IMPLEMENTED | Not tied to per-entity confidence |
| 2D mask → 3D object lifting | `perception/instances/lifting.py` | IMPLEMENTED | Uses MiDaS (relative depth) and has no scale gate |
| Multi-view object merge | `perception/instances/object_resolution.py` | PARTIAL, **unreliable** | Union-find over label + centroid distance + reprojection. **Transitive chaining:** A~B and B~C merges A with C even if A≠C. No Hungarian, no AMBIGUOUS state |
| Cross-session entity re-ID | `world_ir/entity_reid.py` | IMPLEMENTED (geometric, no embeddings) | Snapshot-to-snapshot only, not per-observation |
| Planes, room box, openings, windows, stairs, topology | `perception/geometry/`, `perception/architecture/` | IMPLEMENTED | Topology edges not geometrically re-validated after merges |
| Incremental "affected region" closure | `world_ir/incremental.py` | IMPLEMENTED | Not wired to new-photo ingestion |
| Measured capture guidance | `engine/pipeline/guidance.py` | PARTIAL | Image-level advice, not per-region world-space requests |
| Real/synthetic quality firewall | `world_ir/quality_state.py` | IMPLEMENTED | Needs OBSERVED / INFERRED / PRIOR_GENERATED per surface |
| Completion proposals (generated interiors) | `apps/api/routes_completion.py` | IMPLEMENTED | Not a geometry-completion system |
| Detection / segmentation / depth backends | `perception/{detection,segmentation,depth}` (Mask R-CNN, SAM, MiDaS) | PARTIAL | No transformer depth/matching. Open-vocab detector work (OWLv2, switchboards) lives in the **unmerged** `room-reconstruction-phase-7` worktree |
| Runtime | `torch 2.14.0+cpu`, CUDA **unavailable** | constraint | "Fast transformers" needs a GPU decision (§7, D1) |
| Export: format exporters + API | `exporters/`, `apps/api/routes_export.py` | IMPLEMENTED, 14/14 route + e2e tests pass | Real-app failure **not yet reproduced** (§1) |

---

## 1. Export fix (do first, small)

**Facts found:**
- Backend export tests pass (`tests/test_export_route.py`, `tests/test_export_pipeline_e2e.py`: 14 passed).
- `SpatialWorkstation.handleExport` (`frontend/src/components/workspace/SpatialWorkstation.tsx:261`) swallows every failure in `console.error`. The user sees **nothing** when WorldIR / PLY / cameras / report export fails.
- `ExportPanel` posts straight to `apiBaseUrl()` (`:8100`). The four artifact exports go through Next proxy routes instead, so there are two different paths.
- On this machine **port 8100 was already taken** when I tried to start the API (a stale API process is likely, so the UI may be talking to old code). `next` is not installed in this worktree.

**Steps:**
1. Reproduce: run `start.bat`, open a world that has a version, try every export option, capture the network status and body.
2. Classify the cause: stale process on 8100 / CORS / 409 "no version" / proxy timeout / exporter exception.
3. Fix at the shared root, not per button. One `downloadArtifact()` helper used by both paths, with errors surfaced in the UI (toast plus reason).
4. Show export as an **icon grid of formats**, each disabled with a one-line reason when the world can't export it (no version, IFC lib missing).
5. Add a regression test: an e2e that exports gltf + PLY from a freshly compiled world and asserts a non-empty download.

**Done when:** every format downloads a valid file from a real reconstructed world in the browser, and every failure shows a visible reason.

---

## 2. Architecture: one evidence graph, many models

```
 photos ─▶ [A] Image understanding ──┐  (quality, features, masks, depth, normals, uncertainty)
                                     ▼
          [B] Camera graph  ◀── learned matches ─▶ geometric verification (RANSAC, BA)
                                     ▼
          [C] Geometry  (COLMAP sparse/dense + planes + metric scale)
                                     ▼
          [D] OBSERVATION LEDGER ──▶ ASSOCIATION ENGINE ──▶ ENTITY REGISTRY (persistent IDs)
              every 2D detection        gated Hungarian /        WorldIR entities + evidence refs
              lifted to world frame     AMBIGUOUS kept           + per-property confidence
                                     ▼
          [E] Coverage map ─▶ missing regions ─▶ capture requests ─▶ new photo ─▶ relocalise ─▶ local update
                                     ▼
          [F] Validation: reprojection, duplicates, contradictions ─▶ versioned WorldIR + quality report
```

**Rule:** neural nets *propose*, geometry *verifies*. No model writes WorldIR directly. Everything goes through the Observation Ledger, which carries `source_image`, `pixel_region`, `model@version`, and `confidence`.

### New data contracts (additive to WorldIR, no breaking change)

| Type | Fields (key) | Lives in |
|---|---|---|
| `Observation` | id, image_id, mask/bbox, label, model@ver, world_pos ± cov, normal, depth_source (`metric`/`relative`/`mvs`), visibility | `world_ir/observations.py` (new) |
| `AssociationDecision` | candidates, probability, geom_verified, reproj_err_px, dist_m, normal_diff_deg, occlusion_state, **state** ∈ {SAME_ENTITY_CONFIRMED, SAME_SURFACE_CONFIRMED, DIFFERENT_ENTITY, PHYSICAL_INTERSECTION, IMAGE_OCCLUSION, PARTIAL_OVERLAP, AMBIGUOUS, INSUFFICIENT_EVIDENCE}, reason | same |
| `EvidenceClass` per surface/property | OBSERVED, PARTIALLY_OBSERVED, OCCLUDED, UNOBSERVED, INFERRED, PRIOR_GENERATED, CONFLICTED, UNKNOWN | extend `quality_state.py` |
| `ConfidenceVector` | classification, identity, position, orientation, dimension, geometry, completeness, material, colour, topology, cross_view | on entity |
| `CaptureRequest` | region_id, world_bbox, normal, suggested camera pose(s), anchor features, plain-language instruction, thumbnail | `engine/pipeline/guidance.py` (extend) |

---

## 3. Model selection (benchmark before adopting; licence-checked)

| Stage | Candidate (primary → fallback) | Why | Licence note | Cost (CPU / GPU est.) |
|---|---|---|---|---|
| Local features + matching | **ALIKED/SuperPoint + LightGlue** → LoFTR → existing SIFT | LightGlue is fast, handles wide baselines and blank walls better than SIFT | SuperPoint weights are non-commercial; ALIKED + LightGlue are permissive | slow / ~20 ms per pair |
| Pose bootstrap for hard sets | **MASt3R** (transformer, metric pointmaps) → COLMAP | Registers low-texture rooms where SfM fails | CC BY-NC (**blocks commercial use**) | very slow / ~0.3 s per pair |
| Metric depth | **Depth Anything V2 (metric-indoor)** → Metric3D v2 → MiDaS (current) | Transformer depth, indoor metric head | V2-Small Apache, larger sizes NC | ~1 s per image / ~40 ms |
| Normals | Metric3D v2 or DSINE | Plane and edge refinement | check | — |
| Segmentation | **Mask2Former / OneFormer (ADE20K)** for walls, floor, ceiling, doors, windows; **SAM 2** for instance masks | Architectural classes + clean instance boundaries | Apache | ~0.5 s per image GPU |
| Open-vocab objects | **OWLv2 / Grounding-DINO** (merge existing phase-7 work) | Switches, sockets, vents, fixtures | Apache | — |
| Re-ID embedding | **DINOv2** patch features per mask | Appearance cue *inside* the geometric gate only | Apache | ~30 ms per image GPU |
| Completion (later) | Plane/boundary extrapolation (classical) first; PoinTr/SnowflakeNet only if benchmarks justify | Constrained, auditable | check | background job |

Fall back to classical methods at every stage, so the pipeline still runs on CPU-only machines and simply reports lower confidence.

---

## 4. Phased execution (priority order; each phase is tested, measured and documented before the next)

### Phase 1: Overlap and identity foundation (top priority)
1. `Observation` ledger: lift every detection/mask with camera + depth into world frame, with covariance (depth_source `relative` widens covariance heavily).
2. **Association engine** replacing union-find chaining in `object_resolution.py`:
   - gate: label compatibility (hierarchy-aware), 3D Mahalanobis distance, normal difference, bbox IoU in 3D, epipolar/reprojection check, *visibility* (a camera that should see it but doesn't is negative evidence)
   - score: weighted log-likelihood (geometry dominant, DINOv2 appearance secondary)
   - assign: Hungarian per image vs. registry; **no transitive merges**; under-margin cases → `AMBIGUOUS`
   - every decision persisted as `AssociationDecision` with its reason
3. Surface association: plane parameter test (normal angle, point-to-plane distance, overlap of 2D extents on the plane), then fuse compatible observations and keep conflicting ones as `CONFLICTED`.
4. Intersection vs. occlusion: depth ordering along camera rays. Masks overlapping in an image but separated in depth → `IMAGE_OCCLUSION`. Volumes touching or penetrating in 3D → `PHYSICAL_INTERSECTION` (wall-floor, frame-in-wall, mounted-on).
5. Topology graph re-validated geometrically after every merge (reuse `perception/architecture/topology.py`).
6. Tests: adversarial scenarios 1-9, 13, 15, 16 from the brief as synthetic fixtures with ground-truth IDs.
- **Metrics:** identity precision/recall, duplicate-entity rate, incorrect-merge rate, incorrect-split rate.

### Phase 2: Camera and geometry accuracy
1. Fill the `reconstruction/{features,matching}` stubs with a `IMatcher` interface: SIFT (existing), ALIKED+LightGlue, MASt3R (optional, NC-gated).
2. Benchmark on `datasets/south_building`, your tape-measured rooms (D3) and ScanNet++ / ARKitScenes samples: rotation/translation error, registration rate, reprojection error.
3. Metric scale: reconcile MVS scale, metric-depth priors and any known dimension (door height prior only as INFERRED, never OBSERVED).
4. Plane + corner refinement with edge-preserving robust losses (no smoothing across corners or trim).
- **Metrics:** Chamfer, F-score@5 cm, wall-length error, opening-dimension error, before vs. after.

### Phase 3: Persistent entity identity across sessions
Wire the ledger + registry to `entity_reid.py` and the WorldStore versions so an object seen in 5 photos and 2 sessions keeps one ID.

### Phase 4: Missing-region intelligence + guided capture
1. Coverage map: voxel/surfel visibility from registered cameras → per-region EvidenceClass.
2. Separate "real opening" (doorway with frame evidence) from "unobserved hole".
3. Next-best-view: sample candidate poses, score by expected newly-observed area × registrability (shared anchor features visible).
4. `CaptureRequest` rendered as **a 3D highlight + suggested camera frustum + a short sentence**.
5. New photo: quality check → relocalise against the existing model (matcher + PnP) → reject with reason if it fails (wrong room, no shared features) → associate → `apply_incremental_update` on the affected closure only → new WorldStore version + visual diff.
- **Metrics:** registration success of follow-up photos, recovery rate, local-update time vs. full rebuild.

### Phase 5: Constrained completion
Plane/boundary extrapolation for walls/floors (INFERRED), learned completion only as PRIOR_GENERATED proposals behind acceptance gates (continuity, normals, collision, topology, reprojection where evidence exists). Never auto-promoted.

### Phase 6: Appearance fidelity
Per-surface colour from multi-view median in linear space with white-balance/exposure uncertainty; material class separate from observed colour.

### Phase 7: Whole-journey validation
End-to-end run of the 20 adversarial scenarios; quality report per world.

---

## 5. Speed architecture

| Pass | Runs | Target (set after baseline) |
|---|---|---|
| **Fast** (seconds) | quality, low-res seg, feature extraction, matching, initial poses, room box | first 3D view quickly |
| **Focused** | high-res seg + depth only on uncertain regions, local association | progressive refinement in view |
| **Background** | dense MVS, global BA, completion proposals, calibration | streamed updates |

- Cache features / masks / depth by `sha256(image)+model@ver` (content-addressed store already exists).
- One backbone pass shared by seg + re-ID (DINOv2 features), not separate models per task.
- Batch on GPU, ONNX/fp16 export for the hot models.
- **No speed claims without measured numbers on named hardware.**

---

## 6. UI: "show, don't tell"

Replace text-heavy panels in the reconstruction view with:
- **Evidence overlay toggle:** surfaces coloured by EvidenceClass (observed solid, inferred hatched, unobserved red outline, conflicted amber). The legend is icons, not prose.
- **Overlap view:** click an entity to see thumbnails of every photo that sees it, with the mask outlined. Ambiguous merges show side-by-side for a one-click Same / Different.
- **Capture requests as 3D markers:** a camera ghost where to stand, plus the target region glowing. One line of text max.
- **Change diff after a new photo:** before/after slider, changed regions pulse.
- **Export:** format icon grid (§1.4).
- Text budget: ≤ 1 short line per card; details behind a hover or expand.

---

## 7. Decisions (answered 2026-11-01)

| # | Decision | Answer | Consequence for execution |
|---|---|---|---|
| D1 | Compute | **Local NVIDIA GPU**: RTX 4050 Laptop, **6 GB VRAM**, driver 617.42 (detected) | Step 0: replace `torch 2.14.0+cpu` with the CUDA build in the project venv; verify `torch.cuda.is_available()`. A 6 GB budget means **one heavy model resident at a time** (load, run batch, unload), fp16 everywhere, and small/base variants: Depth Anything V2-S/B, Mask2Former Swin-T/S (not Swin-L), DINOv2-S/B, SAM 2-small, MASt3R at 512 px only. The CPU path stays as a fallback and is reported as lower confidence. |
| D2 | Licences | **Permissive by default; non-commercial behind a flag** | Default stack: ALIKED + LightGlue, Depth Anything V2-Small, Mask2Former/OneFormer, SAM 2, DINOv2, OWLv2. `REALITY_ALLOW_NC_MODELS=1` enables MASt3R, SuperPoint and larger Depth Anything. Every Observation records `model@version` + licence, so NC-derived geometry is traceable and can be excluded from commercial exports. |
| D3 | Data | **Public datasets + your own captures** | Public: ScanNet++ and ARKitScenes (access requests needed, filed at Phase-0), Structured3D for layout. Yours: 3-5 rooms photographed with the capture guide **and tape-measured** (wall lengths, ceiling height, door/window sizes) as ground truth. Splits by room/property. Fine-tuning only after the baseline shows where the error is. |

---

## 8. Acceptance (numbers set after Phase-0 baseline, reported separately from targets)

Required metric families: camera rot/trans error, registration rate, reprojection error; Chamfer, F-score@τ, wall/opening dimension error; identity P/R, duplicate rate, wrong-merge/split rate, occlusion accuracy; per-class IoU, instance AP, small-object recall; missing-region recovery, false-completion rate; ΔE colour under controlled capture; end-to-end time and memory per hardware tier.

Splits are **by scene/property**, never by image.

---

## 9. Order of work

0. CUDA PyTorch in the venv + GPU smoke test (D1); file dataset access requests (D3)
1. Export fix (§1)
2. Baseline benchmark harness + adversarial fixtures (+ your tape-measured rooms)
3. Phase 1 overlap/identity
4. Phase 2 matcher + depth upgrades (after D1/D2)
5. Phase 4 missing regions + guided capture
6. UI evidence overlay + capture markers (alongside 3-5)
7. Phases 3, 5, 6, 7

Each step: implement → test → measure → update `docs/implementation/PENDING_IMPLEMENTATION.md` and `IMPLEMENTATION_STATUS.md` with evidence.

### Step 0 detail: GPU setup
1. Check the CUDA version the driver supports: `nvidia-smi` (driver 617.42 detected).
2. Install the CUDA build of PyTorch matching the installed torch version, using the selector at pytorch.org/get-started/locally (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX`). Install it in the same Python the API runs under; `launch_ui.bat` prefers `.venv\Scripts\python.exe` in the repo or main checkout, otherwise system `python`.
3. Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` must print `True NVIDIA GeForce RTX 4050 Laptop GPU`.
4. Add a `perception/device.py` helper (cuda → cpu fallback, fp16 flag, free-VRAM check) used by every model backend, plus one test that passes on both CPU and GPU.

---

## Appendix A: Adversarial test scenarios (each must be a repeatable test)

Each test must check that the system **makes the correct decision, records its uncertainty, and does not claim false success**.

1. Two images showing the same wall from different angles → one wall entity.
2. The same door appearing in five photographs → one door, 5 evidence refs.
3. The same window partially visible in three photographs → one window, PARTIALLY_OBSERVED where applicable.
4. Two visually similar but physically different doors → two entities (DIFFERENT_ENTITY).
5. Two overlapping furniture detections that are actually different objects → two entities.
6. Furniture hiding part of a wall → wall region OCCLUDED, not missing, not merged.
7. A wall corner visible from only one viewpoint → corner kept, lower confidence.
8. A room photographed from opposite directions → one consistent room.
9. Two capture sessions of the same room → same entity IDs carried across.
10. A new photograph with a large viewpoint change → registers or is rejected with a reason.
11. A new photograph with no reliable shared features → rejected with a reason and a better capture request, never a separate coordinate system merged by appearance.
12. A missing wall region supplied later by the user → local update only; the rest of the world unchanged.
13. A mirror causing false correspondences → no phantom room behind the mirror.
14. Glass and transparent surfaces → not reconstructed as solid walls without evidence.
15. Blank white walls with few features → handled (learned matcher / planes) or flagged UNKNOWN.
16. Repeated floor or wall patterns → no false matches collapsing distinct regions.
17. Low-light and motion-blurred photos → quality-flagged, down-weighted.
18. Incorrect camera pose estimates → detected by reprojection or consistency checks.
19. A photo of an adjacent room presented as the current room → rejected or routed to a new room, not merged.
20. A photo that contradicts existing geometry → CONFLICTED with both observations kept, never silently overwritten.

## Appendix B: Non-negotiable rules

1. A 2D detection is not a 3D reconstruction.
2. Monocular depth is not metric geometry without calibration or scale evidence.
3. Never merge two objects solely because they look similar.
4. Image-space overlap is not physical intersection.
5. Never duplicate one physical object across viewpoints.
6. Never invent hidden geometry and label it observed.
7. A generative model never overwrites stronger photographic evidence.
8. Never silently discard contradictory observations.
9. Never rebuild the whole property when a local update is enough.
10. Never change canonical world coordinates without an explicit transform.
11. Never claim accuracy without ground-truth measurement.
12. Never claim a speed-up without a benchmark.
13. Don't install every model; select on evidence, licence, hardware and measured performance.
14. Preserve working COLMAP, WorldIR, WorldStore and reconstruction capabilities unless an audited reason justifies a change.
15. Keep entity identities, evidence references, confidence values and provenance attached to the canonical WorldIR.
16. The user can always inspect and correct uncertain geometry.

## Appendix C: Final report the agent must return

For each completed step: files changed, tests run with actual output counts, measured metrics (baseline vs. after, with hardware named), what was verified in the browser, remaining limitations, and registry status changes. Anything not run is stated as **not verified**.
