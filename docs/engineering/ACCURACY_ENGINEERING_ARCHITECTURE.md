# Accuracy engineering: target architecture, roadmap, risks

Companion documents: `capability_matrix.md` (what exists, reachable, tested), `TECHNOLOGY_EVALUATION.md` (candidates, licences,
measurements), `COMPLETE_ENTITY_UNDERSTANDING_AUDIT.md` (entity coverage), `benchmarks/reality_benchmark/` (the evaluation contract).

## 1. What the audit changes about the plan

The directive assumes an orchestrator, a correction loop, capture guidance and a technology registry must be built. They
already exist as code (observed): `reconstruction/orchestrator.py` (backend selection, failure fall-through, provenance),
`reconstruction/diagnostics.py` + `correction.py` (structured findings, bounded corrections accepted only on measured
improvement), `engine/quality/autonomous_correction_loop.py`, `reconstruction/capture_quality.py` + `completeness.py`,
`engine/technologies/` and `perception/model_registry.py` + `availability.py`. What is true today:

* The live API path (`apps.api.jobs` -> `progressive` -> stage ladder) does **not** reach `engine/quality/report.py`,
  `reconstruction/completeness.py`, `reconstruction/sfm/core_pipeline.py`, the SAM backends, tracking, video frame
  extraction or the small-object pass (capability matrix, "Implemented but not reachable").
* No stage compares independent predictions, and nothing recorded accuracy between versions (until `--history` was added now).
* Three rows have failing tests today (matrix: image quality 1, SfM 1, MVS 4), all in COLMAP/path/dense areas.

So the work is **connect, gate and measure**, not rewrite. A new orchestrator that duplicates these would add the very
complexity the directive warns against.

## 2. Target architecture (built on existing parts)

```
EVIDENCE   evidence/*, capture_quality (blur, overlap, coverage), video_frames          [exists; video not on API path]
   |
PERCEPTION per image: detect (Grounding DINO, coarse) + small-object tiles + masks (SAM 2) + embeddings (DINOv2)
   |          -> per-image INVENTORY records (benchmarks/reality_benchmark/inventory.py contract)       [inventory missing]
GEOMETRY   COLMAP poses -> depth (candidate: metric depth, ONE global scale) -> fusion -> planes/rooms
   |          independent checks: reprojection, cross-view depth consistency, SfM-vs-depth residual
MULTI-VIEW association (gated, epipolar, appearance) -> canonical entity ids -> WorldIR
   |
VERIFY     diagnostics (existing) + accuracy gates (new, benchmark-driven) -> diagnosis -> targeted correction
   |
ORCHESTRATE  stage ladder (existing) + diagnosis->correction table + resource scheduler (sequential on 6 GB)
   |
COMMIT     validated WorldIR only; release report + accuracy history (added)
```

### The Accuracy Orchestrator, concretely

Not a new framework. Three additions to the existing ladder:

1. **Engine contracts**: each stage declares inputs, outputs, VRAM budget and a measurable post-condition (extends `StageSpec`).
2. **Diagnosis table** mapping measured findings to an existing, bounded correction:
   duplicate entity -> re-run association; missing small fixture -> targeted tile pass on the wall region; several "floor"
   levels -> separate the floor (lowest large level) from support surfaces (seat/table tops, steps); the depth source was tested and is
   not the cause (TECHNOLOGY_EVALUATION 3a); weak registration -> next mapper seed (exists); uncertain region -> capture request.
3. **Gates fed by the benchmark**: a candidate result is committed only when the relevant benchmark metrics do not regress
   (the existing `judge()` acceptance pattern, generalised) and the critical gates pass.

## 3. Dependency and licensing review

Detail in `TECHNOLOGY_EVALUATION.md`. Consequences for the roadmap: VGGT, DUSt3R/MASt3R, CoTracker, Depth Anything Base/Large and
Ultralytics YOLO are **excluded from a commercial path as published**; the plan therefore keeps COLMAP as the geometry
authority and treats VGGT/MASt3R as research-only comparisons, never shipped. Candidates cleared on the licences read: SAM 2,
Grounding DINO, Florence-2, Depth Anything V2 Small, DINOv2, LightGlue, Open3D. Depth Anything V2 Metric-Indoor Small has no
declared licence and needs confirmation before it ships.

## 4. Integration roadmap (each step: change, benchmark before/after, regression tests)

P0 (in order; each has a pass criterion that is measured, not asserted)

1. **Ground truth first.** Annotate the available images (boxes + cross-view instance ids); survey or otherwise obtain dimensions for at least one real room. Blocks every accuracy claim.
2. **Persist per-view observations** (box, mask, score, label, view) through promotion; `inventory_from_world`. Pass: inventory validates and round-trips a stored world.
3. **Floor vs support surfaces.** The metric-depth experiment (TECHNOLOGY_EVALUATION 3a) showed the many "floor" levels are not a depth artifact. Pass: on annotated images the floor entity is the lowest large level and furniture tops/steps are not floors. Metric depth stays an option for scale only, pending a reference measurement (its implied camera height is ~20-30% high).
4. **Small-object pass on the live path with a precision gate.** Pass: small-object precision and recall measured on annotated images, reported per entity group.
5. **Masks** (SAM 2, after install approval) from boxes. Pass: instance mIoU measured.
6. **Wire the unreachable pieces that earn it**: quality report, completeness, correction loop. Pass: the matrix row becomes reachable and a test exercises the live call.
7. **Fix the 6 failing tests** or classify each as environmental with evidence.

P1: fine-grained classes, materials on objects, active capture recommendations from completeness, human-approved corrections into a curated failure set (never auto-train on unverified predictions).
P2: multi-storey scale, distributed processing.

## 5. Risks and bottlenecks (observed unless marked)

| Risk | Evidence | Mitigation |
|---|---|---|
| No ground truth anywhere | audit; benchmark all `BLOCKED` | step 1 above; do not claim accuracy before it exists |
| Test images are AI-generated and may disagree across views | COLMAP geometry differed 995 vs 1,423 points between runs; planes at 45-60 deg in a bad run | include real captures (dataset B); treat generated scenes as perception inputs only |
| COLMAP run-to-run variance | observed | multiple seeds, accept by measured agreement; record the seed |
| 6 GB VRAM | observed (Grounding DINO spill) | batch/precision by measured peak; sequential model scheduling; probe script |
| Licences exclude VGGT/MASt3R/CoTracker | verified | research-only; keep COLMAP authority |
| Pass-by-API illusion | 9 matrix rows Partial | reachability is audited by `scripts/capability_audit.py`; verify by execution |
| Unit tests are not accuracy | all rows | matrix separates "unit tests pass" from "real-data evidence" |
| One full run takes ~13 min | observed, partly fixed (detector 3.4 s -> 0.2 s per prompt) | re-time the full run after the fix (not done) |
