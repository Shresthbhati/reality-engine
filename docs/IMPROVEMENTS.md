# Improvement catalogue

Stable IDs. One row = one distinct, checkable improvement. No padding: a row needs evidence or is marked as a hypothesis.

**Class:** OF observed failure · VN verified need · MP measured performance gap · RO research opportunity · PO product opportunity · SP speculative · DN done.
**Evidence:** [V] read in source or seen in a run this month · [M] capability matrix / entity audit (15 Nov) · [A] 9 Nov audit, recheck · [H] hypothesis.
**Lands in:** work package in `docs/BUILD_SPEC.md`, or `open` (not yet scheduled), or `later` (after December).

How this grows honestly: every ontology class × every benchmark metric becomes a row **once it is measured** (118 metrics exist, none measured yet). Annotate the 15 test images and measure six rooms, and this file gains hundreds of real rows. Until then it holds what the evidence supports.

| ID | Class | Problem → change | Evidence | Lands in |
|---|---|---|---|---|
| RE-ARCH-000001 | OF | Stages pass live objects, so a crashed job cannot resume → stages exchange content-addressed artifacts | [V] `apps/api/jobs.py::_RecordOnlyStore.lookup` returns None | B01 |
| RE-ARCH-000002 | OF | Two taxonomies (`perception/ontology.py`, `perception/entities/taxonomy.py`) → one ontology | [V] | B03 |
| RE-ARCH-000003 | VN | Detections are not persisted per photo → `ObservationTable` artifact | [M] | B02 |
| RE-ARCH-000004 | VN | Parts are not entities → parts as entities with `part_of` | [M] | B04, B08 |
| RE-ARCH-000005 | VN | Relationship kinds are free strings → closed validated set | [V] `Relationship.kind: str` | B15 |
| RE-ARCH-000006 | VN | `Visibility` has 3 states; no "not observed" → coverage enum | [V] `world_ir/observations.py` | later |
| RE-ARCH-000007 | OF | `StatementState` duplicates `Provenance` → single vocabulary | [V] | later |
| RE-ARCH-000008 | VN | `vertical_slice.py` ~2,000 lines → split along stage boundaries as stages take over | [V at 199ca3bb] | B01 |
| RE-ARCH-000009 | VN | `routes_worlds.py` ~3,500 lines → split by resource when next touched | [V at 199ca3bb] | open |
| RE-ARCH-000010 | VN | `AdaptiveInspector.tsx` 102 KB → new UI in new files only | [V] | B18 |
| RE-ARCH-000011 | OF | `UniversalSpatialPipeline` unwired second pipeline → fold or delete | [V at 199ca3bb], inventory in `docs/audits/` | open |
| RE-ARCH-000012 | VN | Unknown objects have no home → every ontology family gets an `unknown` child, kept not dropped | [H] | B03 |
| RE-ARCH-000013 | VN | Containment has two sources (query routes vs future relations) → one source | [V] `routes_query.py` `/contents` | B15 |
| RE-ARCH-000014 | VN | Artifact lineage not queryable → DAG of artifact → producing stage, invalidate by reverse BFS | [H] | B01 |
| RE-PERC-000001 | VN | 19 flat prompts for everything → whole → part → sub-part detection | [M] | B04 |
| RE-PERC-000002 | OF | Tiled small-object pass mislabels (railing as wiring, texture as drain) → parent-box and size checks | [M] 10 images | B04 |
| RE-PERC-000003 | OF | 678 unreviewed small-object drafts, no ground truth → review queue before any use | [M] | open |
| RE-PERC-000004 | OF | `small_objects.py` not reachable from the API → called from hierarchical detector | [M] | B04 |
| RE-PERC-000005 | OF | SAM 2 backend exists, package and weights absent, 0/3 reachable → install, wire, RLE store | [M] | B06 |
| RE-PERC-000006 | VN | 3D lifting uses boxes, includes background → lift from mask pixels | [M] | B06 |
| RE-PERC-000007 | OF | DINOv2 re-id code not wired into association → embeddings passed to `associate(appearance=…)` | [M], [V] param exists | B07 |
| RE-PERC-000008 | VN | Identity joining is order-dependent → global clustering, strongest evidence first | [V] sequential Hungarian | B08 |
| RE-PERC-000009 | VN | Two detections in one photo can merge → same-image cannot-link | [H] | B08 |
| RE-PERC-000010 | VN | No attributes (socket type, lock type, handle type) → margin-gated attribute classifier | [M] | B05 |
| RE-PERC-000011 | VN | Attribute votes across views treated as independent → cap correlated evidence at 3 views | [H] | B05 |
| RE-PERC-000012 | OF | Materials only for room surfaces, not objects → material as an attribute on every family | [M] | B05 |
| RE-PERC-000013 | OF | Switchboard decomposition unreachable → becomes ontology components of `switchboard` | [M] | B03, B04 |
| RE-PERC-000014 | OF | Fine-grained pipeline only on SDK path → retired in favour of B04 | [M] | B04 |
| RE-PERC-000015 | OF | Tracking modules 0/3 reachable → used for video only, after photo identity works | [M] | later |
| RE-PERC-000016 | VN | No text reading (nameplates, labels) → OCR as attribute basis | [H] | later |
| RE-PERC-000017 | OF | Closed windows found by gaps only → visual evidence path | [A] | later |
| RE-PERC-000018 | VN | Mirrors and glass create phantom objects → reflective-surface flag suppresses lifting behind planes | [H] | later |
| RE-PERC-000019 | VN | Detector scores used as confidence → calibrate once annotations exist | [M] no annotations | open |
| RE-PERC-000020 | VN | Blurry regions still produce detections → per-region sharpness gates low-score detections | [H] | open |
| RE-PERC-000021 | VN | Occlusion state not recorded → visibility per observation from mask vs box area and depth order | [M] | B06 |
| RE-PERC-000022 | VN | Instance counts unreliable → count = entities after B08, never detections | [M] | B08 |
| RE-PERC-000023 | VN | Part outside its parent accepted → reject by containment | [H] | B04 |
| RE-PERC-000024 | VN | Sizes not checked against class → `typical_dims_m` plausibility after lifting | [V] field exists | B04 |
| RE-PERC-000025 | VN | Same detector call repeated per crop per attribute → batch attribute phrases 2 per pass | [M] measured batch setting | B05 |
| RE-GEO-000001 | OF | Same photos give 995 vs 1,423 sparse points → deterministic SfM | [M] | B12 |
| RE-GEO-000002 | OF | 9 of 10 photos register → report the unregistered photo and why | [M] | B12 |
| RE-GEO-000003 | DN | Dense reconstruction silently ran sparse-only → explicit arguments win | [V] commit `04362f72` | done |
| RE-GEO-000004 | OF | Furniture tops and steps classed as floor → support-plane rule | [M] | B09 |
| RE-GEO-000005 | OF | Room footprint ragged → direction snapping within 5°, polygonize | [M] | B10 |
| RE-GEO-000006 | VN | Missing wall gets closed by invention → report `partial` with gap length | [H] | B10 |
| RE-GEO-000007 | MP | DPT per-view depth residual 0.199 vs 0.117 for metric model → switch only if licence clears and A/B improves structure | [M] one scene | open |
| RE-GEO-000008 | OF | Metric depth did not fix floor/ceiling → not a depth problem; see RE-GEO-000004 | [M] | B09 |
| RE-GEO-000009 | VN | Points from different sources averaged ad hoc → inverse-variance voxel fusion with source bits | [H] | B11 |
| RE-GEO-000010 | VN | Empty space and unseen space look the same → voxel absent = not observed | [H] | B11 |
| RE-GEO-000011 | VN | Dimensions lack provenance → fused measurement with used/rejected ids | [H] | B14 |
| RE-GEO-000012 | VN | Correlated measurements counted as independent → group down-weighting | [H] | B14 |
| RE-GEO-000013 | OF | Uncertainty shown as constants → "uncalibrated" until rooms measured | [V at 199ca3bb] | open |
| RE-GEO-000014 | OF | `reconstruction/sfm/core_pipeline.py` unreachable → wire or delete | [M] | open |
| RE-GEO-000015 | OF | `reconstruction/mvs/dense_cloud.py` unreachable → wire or delete | [M] | open |
| RE-GEO-000016 | VN | SfM fails on blank walls → feed-forward backend as second candidate | [M] | later |
| RE-GEO-000017 | VN | No device poses → ARKit/ARCore priors | [A] | later |
| RE-GEO-000018 | VN | Wall thickness is a default → measured from both faces when both seen, else `INFERRED` | [A] | later |
| RE-GEO-000019 | VN | Mesh quality unknown → watertight / manifold / components verdict | [H] | open |
| RE-GEO-000020 | VN | Openings cut by ad hoc geometry → robust booleans | [H] | later |
| RE-GEO-000021 | RO | Domes, arches, vaults unrepresentable → surface-of-revolution and arch primitives | [H] | later |
| RE-GEO-000022 | VN | Steps are not parts of stairs → stair ontology with step/riser components | [M] | B03, B09 |
| RE-GEO-000023 | VN | Second storey can be read as a table → storey = floor with its own walls | [H] | B09 |
| RE-GEO-000024 | RO | Scale from known objects (door heights) → scale hypothesis marked `INFERRED` | [H] | later |
| RE-GEO-000025 | VN | Collinear wall fragments (25 → 8 by consolidation) → merge by sweep along direction | [M] | B10 |
| RE-PERF-000001 | MP | 10-photo run ~13 min, not re-timed after detector speed-up → time it, record per stage | [M] | B20 |
| RE-PERF-000002 | DN | Grounding DINO 3.4 s → 0.2 s per prompt | [M] | done |
| RE-PERF-000003 | VN | Models reload between stages → `ModelPool` LRU | [H] | B13 |
| RE-PERF-000004 | VN | Out-of-memory can kill a job → admission check, CPU fallback or `skipped` | [H] | B13 |
| RE-PERF-000005 | VN | VRAM per model guessed → measured table persisted | [M] 4 models measured | B13 |
| RE-PERF-000006 | VN | SAM 2 image encoder would run per box → once per image | [H] | B06 |
| RE-PERF-000007 | VN | DINOv2 would run per crop → one forward per image, pool tokens by mask | [H] | B07 |
| RE-PERF-000008 | VN | Masks as images → run-length encoding | [H] | B06 |
| RE-PERF-000009 | VN | Embeddings as Python lists → float16 memory-mapped matrix | [H] | B07 |
| RE-PERF-000010 | VN | Depth maps float32 in RAM → float16 memory-mapped | [H] | B01 |
| RE-PERF-000011 | VN | Identity candidates N² → KD-tree pairs within 3σ | [H] | B08 |
| RE-PERF-000012 | VN | Unchanged stages rerun → skip on matching inputs hash | [V] | B01 |
| RE-PERF-000013 | VN | New photo rebuilds everything → only stages whose inputs changed | [H] | B01 |
| RE-PERF-000014 | VN | Video duplicates processed → counted and dropped | [V] picker keeps 6 of 18 | B17 |
| RE-PERF-000015 | MP | `pytest -k "colmap or dense"` takes 57 min → mark slow, split | [V] | open |
| RE-PERF-000016 | VN | Nine specialist tools in the Studio bundle → lazy-load | [V at 199ca3bb] | later |
| RE-PERF-000017 | VN | Refinement can loop without gain → heap by gain/cost, stop on no change | [H] | B16 |
| RE-PERF-000018 | VN | Detector calls unbounded per image → `budget_calls` | [H] | B04 |
| RE-DB-000001 | VN | Stage lookup by (job, name, inputs hash) needs an index → confirm unique index exists | [M] | B01 |
| RE-DB-000002 | VN | Dev on SQLite, production on Postgres → run core tests on both | [V] `apps/api/db.py` | open |
| RE-DB-000003 | VN | Job claim locking on Postgres unverified → two-claimer test | [A] | open |
| RE-DB-000004 | VN | Observations as DB rows would be millions → artifact files, DB holds the hash | [H] | B02 |
| RE-DB-000005 | VN | Backups never restored → one restore drill | [A] | open |
| RE-DB-000006 | VN | WorldStore and DB can disagree after a crash → test crash between the two writes | [V] D01 tests cover this area | D01 |
| RE-SEC-000001 | VN | Non-commercial model can run → licence gate called at job start | [V] gate exists, `mast3r` is CC-BY-NC | open |
| RE-SEC-000002 | VN | Metric-Indoor depth checkpoint has no declared licence → keep out of default path | [M] | open |
| RE-SEC-000003 | VN | Checkpoint hashes not enforced on load → `verify_checkpoint` before every load | [V] function exists | B13 |
| RE-SEC-000004 | SP | Weights loaded with pickle → `weights_only=True` or safetensors | [H] | open |
| RE-SEC-000005 | VN | Test backend reachable in production path → refuse outside test env | [V at 199ca3bb], recheck | open |
| RE-SEC-000006 | VN | Tenant scoping per route unproven → inventory test over all routes | [A] | later |
| RE-SEC-000007 | OF | `main` unprotected → require CI | [M] | owner |
| RE-SEC-000008 | VN | "Certified Auditor" default on a life-safety PDF → "preliminary" wording | [V at 199ca3bb], recheck | open |
| RE-SEC-000009 | VN | No secret scan in CI → add one | [H] | open |
| RE-SEC-000010 | VN | Upload limits untested → under/over tests per limit | [A] | open |
| RE-SEC-000011 | VN | Coding agents share one checkout → one branch per package, no push | [V] this session | process |
| RE-SEC-000012 | VN | Socket/lock labels could be read as safety claims → visual class only, stated in UI and export | [H] | B05, B19 |
| RE-TEST-000001 | DN | Six failing tests | [V] commit `04362f72` | done |
| RE-TEST-000002 | DN | Two crash-recovery tests fail → verified passing (1125s, 0 failures) on chore/december-release | [V] commit 04362f72 | done |
| RE-TEST-000003 | OF | Video journey test expects 18 frames, engine keeps 6 | [V] | B17 |
| RE-TEST-000004 | DN | CI registry stale on main | [V] commit `f0df2663` | done |
| RE-TEST-000005 | DN | Nightly job missing `trimesh`, `pytest-asyncio` | [V] | done, unverified on GitHub |
| RE-TEST-000006 | VN | Local Python 3.14, CI 3.10–3.12 → run 3.11 locally before release | [V] | B20 |
| RE-TEST-000007 | OF | Lint and type checks cannot fail CI | [V at 199ca3bb], recheck | open |
| RE-TEST-000008 | OF | 118 benchmark metrics, none measured | [M] | open |
| RE-TEST-000009 | OF | Inventory scorer verified on synthetic data only | [M] | open |
| RE-TEST-000010 | VN | No real-photo run in CI → nightly golden loop on a fixed photo set | [M] | B20 |
| RE-TEST-000011 | VN | RLE and voxel packing are easy to get subtly wrong → round-trip property tests | [H] | B06, B11 |
| RE-TEST-000012 | VN | Identity result may depend on input order → shuffle test | [H] | B08 |
| RE-DATA-000001 | OF | 15 test images unannotated | [M] | owner |
| RE-DATA-000002 | OF | No measured rooms | [M] | owner |
| RE-DATA-000003 | VN | No blind hold-out → rooms 5–6 never opened during development | [M] protocol | owner |
| RE-DATA-000004 | VN | Failures not collected → curated failure set, one folder per failure with photos and expected result | [H] | open |
| RE-DATA-000005 | VN | Model pre-labels could become truth → reviewed flag required before any training use | [M] | open |
| RE-UX-000001 | VN | Parts and attributes invisible → parts tree, attribute rows with evidence crop | [H] | B18 |
| RE-UX-000002 | PO | "Where is X" unanswerable → find box, camera flies to result | [H] | B15, B18 |
| RE-UX-000003 | VN | Unknown shown as blank → "Unknown — not visible in any photo" | [H] | B18 |
| RE-UX-000004 | VN | Closed container contents implied empty → "interior not observed", user may state contents | [H] | B15 |
| RE-UX-000005 | VN | 40 routes → 12 | [V at 199ca3bb] | later |
| RE-UX-000006 | VN | Specialist tools read as certification → "Preliminary" badge | [A] | later |
| RE-UX-000007 | PO | No "what to photograph next" → capture requests from unresolved attributes and unseen regions | [M] | B16 |
| RE-ASSET-000001 | VN | Exports flatten parts → IFC aggregates, glTF child nodes | [V] `ifc_bridge.py` | B19 |
| RE-ASSET-000002 | VN | Attributes lost on export → property set with basis and confidence | [H] | B19 |
| RE-ASSET-000003 | VN | Unknown family mapped to a wrong IFC class → proxy element | [H] | B19 |
| RE-ASSET-000004 | VN | Exports never opened in external tools → one tool per format, version recorded | [A] | B20 |
| RE-ASSET-000005 | RO | Not game-ready (no UVs, LODs, collision) → separate production-asset profile | [H] | later |
| RE-GIS-000001 | RO | Worlds have no geographic frame by default → site anchor and CRS on import | [A] | later |
| RE-GIS-000002 | RO | Whole world loads into the viewer → tile streaming | [A] | later |
| RE-GIS-000003 | RO | No aerial capture path → drone ingestion with GNSS | [A] | later |
| RE-MLOPS-000001 | VN | Runs not reproducible from records → store seed, commit, model hashes, artifact hashes per job | [H] | B01 |
| RE-MLOPS-000002 | VN | Technology catalogue (50 entries) omits models in use → reconcile with `model_registry.py` | [M] | open |
| RE-MLOPS-000003 | RO | No trained specialists → blocked on reviewed labels | [M] | later |
| RE-AGENT-000001 | OF | A subagent stalled for 600 s without reporting → main session checks the diff itself, never assumes | [V] this session | process |
| RE-AGENT-000002 | VN | Agents choose data structures ad hoc → fixed table in BUILD_SPEC section 2 | [V] | done |
| RE-AGENT-000003 | VN | Reports paraphrase test output → last output line verbatim | [V] brief template | done |

Row count: run `grep -c "^| RE-" docs/IMPROVEMENTS.md`.
