# Implementation log

Environment: Windows 11, Python 3.14.7, Node 26.8.1, RTX 4050 6 GB, **PyTorch 2.14.0+cpu (CUDA unavailable to torch)**,
COLMAP 4.2.0 CPU-only binary, SciPy 1.18.1. GTSAM: no wheel for Python 3.14. Open3D: not installed.

## Batch A -- stabilise
- Baseline (`pytest tests -m "not slow"`, before any fix): **6 failed, 3159 passed, 28 skipped, 5 xfailed** in 25m48s.
  - FIXED `reconstruction/merge.py:90` read `point.color` on points lacking it (2 colmap_coverage tests) -> `getattr(..., None)`.
  - FIXED `assistant` package missing from `pyproject.toml` packages.find.include (packaging smoke).
  - FLAKY `test_world_store_concurrency::test_no_lost_writes_under_thread_contention`: failed in the 25-min run,
    passed in isolation. Under-load race in WorldStore -- NOT root-caused; do not treat as resolved.
  - OPEN 2x `test_geometry_artifacts::TestPromoteRoomsWritesRealGeometry`: compiled rooms now carry an INFERRED BOX
    geometry `geom-room-plane-NNN` with no `data_uri`; tests expect a boundary point cloud / empty geometry_ids.
    Coincides with the "true room footprint" interior redesign (commit 489272b). Needs an owner decision on which
    contract is intended; not touched.
- 3D Tiles profile verified: 8 tests pass, tileset validates against the official 3d-tiles schema; also Godot profile
  regression (15 pass). CLI `export --format 3dtiles|godot` writes bundle files as bytes. Box geometry + 1 LOD only.
- `cityjson.io` named in the brief does not exist; cloned `cityjson/specs` and `cityjson/cjval` instead.

## Batch B -- SAM
- Backlog said SAM BROKEN; measured: **works** (vit_b, CPU, 1280x960 real photo, points_per_side=16: 12 regions, 29 s).
- Real defects fixed in `perception/segmentation/sam_backend.py`: (1) default vit_h would silently download ~2.5 GB
  -> now `allow_download=False` by default with an actionable error naming cached alternatives; (2) per-image failures
  were swallowed invisibly -> `backend.diagnostics` (status ok/skipped/failed, reason, seconds, device). A missing file
  previously returned `[]` with no explanation (observed). Tests: 3 added, 23 pass incl. the real-model integration test.
- Not done: Depth Anything V2 benchmark vs MiDaS; unified perception-provider interface; GPU torch.

## Batch C -- pose graph
- New `registration/pose_graph.py` (scipy backend). 7 tests (synthetic circular trajectory with known ground truth).
- Measured design failures along the way, kept as documented reasons in the module docstring:
  absolute rotation-vector parameterisation is singular past pi (did not converge in 300 evals, 28 s) -> local
  perturbations + vectorised residuals (131 s -> 14 s suite); Huber smeared an outlier onto healthy edges; Cauchy from a
  drifted start rejected a legitimate loop closure -> likelihood-ratio test per candidate constraint.
- GNSS: needs >=3 non-collinear anchors; initial guess is Kabsch-aligned to the anchors.
- CLI `pose-graph` added. NOT yet connected to registration/cross_session.py or WorldIR; no IMU/VIO.

## Batch C (cont.) -- pose graph wired into cross-session registration, WorldIR, API
- `registration/cross_session.py`: `refine_session_chain` (joint solve over ALL accepted pairwise alignments; BFS result
  is the initial guess and is preserved under `refinement.initial_transforms`; reference stays at identity),
  `align_reconstructed_sessions(all_pairs=, refine=)` (opt-in; defaults unchanged), `CrossSessionReport.refinement`
  (serialised only when set), `record_cross_session_registration(world, report)` -> `world.metadata["registration"]
  ["cross_session"]` with provenance `registration:pose_graph` | `registration:chain`. Per-session frames cannot be
  WorldIR frame-graph edges (Frame is a closed enum), hence the metadata record.
- Side fix: `align_reconstructed_sessions` used `list(results.keys())` as the chain, which only treats the reference
  correctly if it is first; the chain is now reference-first.
- `registration/pose_graph.py`: constraints that are inconsistent but NOT individually identifiable (single loop: every
  edge's removal removes the same cost) are reported in `ambiguous`, nothing rejected; cross-session then keeps the BFS
  result and records `refinement.status = "ambiguous"`. Multi-loop graphs identify the offender (tested, 4 sessions).
- API `apps/api/routes_registration.py` (mounted in `apps/api/main.py`): `POST /api/registration/pose-graph`,
  `POST /api/registration/session-alignments`. Stateless; 422 with the optimiser's reason; size caps.
- Tests: 27 pass (pose graph 8, cross-session refinement 8, API 5, existing cross-session 6).
- MEASURED, upstream: on three synthetic sessions with known frames, `align_session` ACCEPTED a C->A alignment that is
  4.29 m off (rmse 0.55 m, inlier fraction 1.0) while A/B and B/C were exact. The pose graph flagged the loop as
  inconsistent and refused to guess. The ICP acceptance gate (relative to scene scale) is the weak point, not the graph.

## Batch E (start) -- CityJSON validated against the official schema
- Oracle: `cityjson/specs` schemas (v2.0.2, draft-07) via jsonschema + referencing. `cjval` itself NOT run (no Rust
  toolchain here), so geometric validity (planarity, ring orientation, closed shells) is UNCHECKED; only structure is.
- DEFECT FOUND AND FIXED: `exporters/cityjson/exporter.py` emitted `MultiSurface.boundaries` one nesting level too
  shallow (list of rings instead of surfaces -> rings -> indices). Every exported object failed the schema; the module
  and tests claimed official-tool parseability. Fixed at both emit sites (`[ring]`).
- Two existing tests indexed `boundaries` with the old shape (they encoded the defect); updated to flatten
  surface -> ring -> index. New test `test_output_validates_against_official_cityjson_schema` (skips if the clone is
  absent). Exporter suite: 79 passed, 1 skipped.
- Gaps: exporter declares version "1.1" while the oracle is 2.0.2 (version string substituted in the test; for the types
  emitted the structure is identical, not proven for every 1.1/2.0 difference); a procedural building exported 0
  CityObjects (entities carry no transform position, so they are skipped) -- the exporter cannot yet export procedural
  worlds; CityGML/3DCityDB exporters not yet checked against an oracle.

## Batch E (cont.) -- procedural worlds now export from every applicable format
- MEASURED before: on a generated 2-storey building (35 entities), 11 of 14 exporters exported 0 entities (every
  exporter that requires `transform.position`); only ifc, godot and 3dtiles exported (30 each).
- Cause: procedural grammars author bounds in world space and attach no transform. Fix: `exporters.boxes.
  with_explicit_placement(world)` gives GENERATED, box-bounded, transform-less entities an explicit identity placement
  on a shallow copy; `sdk.reality.export` / `export_profile` apply it (so CLI and API get it too). Other provenance
  without transform is still skipped (bounds frame unknown).
- AFTER: 13 of 14 export the same 30 entities; the other 5 are containers with no geometry; SUMO exports 0 by design
  (roads only, reports "not a road"). CityJSON vertex extent equals the geometry bounds exactly.
- Tests: tests/test_procedural_export.py (17). Sweep over all export/API tests: 304 passed, 8 skipped.
- Caveat: direct calls to an individual exporter module (not via sdk.reality) still skip transform-less entities.

## Batch D -- evidence-backed change proposals over WorldStore
- `worldstore/proposals.py`: `propose_update` (candidate vs stored version -> persisted, deterministic, idempotent
  proposal with per-entity verdicts supported | needs_review | conflict), `prepare_acceptance` / `record_acceptance` /
  `accept_proposal`, `reject_proposal`, `list_proposals`, `get_proposal`. Rules are in the module docstring. Accepting
  creates a NEW version (parent = base); the base is never touched; merged world must pass `validate_world_ir`; the
  saved version is read-back verified (quarantined on failure). Conflicts need `override_conflicts` + a reason.
- FOUND BY TESTS: a geometry edited IN PLACE (same ids) leaves the entity record identical, so an entity-diff-only
  assessor produced NO review item for it. Fixed: geometry diffs are attributed to every entity referencing them.
- API `apps/api/routes_proposals.py`: `POST/GET /api/worlds/{id}/proposals`, `GET /{pid}`, `POST /{pid}/accept`,
  `POST /{pid}/reject`. Accept goes through `worldstore_service.commit_version` (DB mirror + atomic HEAD claim); a moved
  HEAD is 409 "stale" (never a silent rebase); another world's proposal is 404; rule refusals 422; already decided 409.
- Tests: 9 (store level) + 5 (API, real SQLite DB + WorldStore). Sweep over WorldStore/adoption/API/packaging: 166 passed.
- Limits: materials/surfaces of applied entities are not copied (validation refuses dangling refs); relationship edges
  from OTHER entities to an added/removed entity are not rewritten; no Studio UI yet; the candidate is supplied as a
  WorldIR document (no "compile new evidence into a candidate" endpoint wired to a proposal yet).

## Batch F (start) -- Studio UI for change proposals
- Files: `frontend/src/lib/api/proposals.ts` (typed DTOs + calls on the canonical client), `frontend/src/lib/proposals/
  proposal-model.ts` (pure selection/validation logic), `frontend/src/components/proposals/ProposalsPanel.tsx`,
  route `frontend/src/app/(desktop)/worlds/[id]/proposals/page.tsx`, entry link in `WorldNavPanel` (Versions section).
- Controls, each wired to a real endpoint: list, select, per-entity checkbox (supported pre-selected), override of a
  selected conflict with a required written reason, Accept N changes, Reject with reason, "Submit a candidate world"
  (paste/load WorldIR JSON -> POST). Server errors are shown verbatim (409 stale gets a reload hint).
- VERIFIED in the browser against a real API + SQLite + WorldStore (throwaway dir): create via UI; verdict ordering and
  reasons; conflict tick -> reason field + disabled Accept + blocker text; accept with override -> new version is HEAD,
  base kept as parent, `wall-b` taken at INFERRED, `door` removal (needs_review) NOT applied; reject recorded, no new
  version; stale proposal -> 409 text shown; 375 px: single column, no horizontal overflow; nav link href correct.
- Automated: `tsc --noEmit` clean; eslint clean on new files; contract suite 139/139 (7 new in proposal-model.test.mts).
- NOT verified: screenshots (pane could not draw while hidden; checked via DOM/text/network instead), keyboard-only
  navigation by hand, contrast by measurement, Playwright e2e (none written), the Next dev server under Turbopack
  (the worktree's symlinked node_modules crashes Turbopack; verified with the repo's webpack launch config).
- Unattributed console error `getComputedStyle ... not of type 'Element'` from anonymous injected code appears on every
  page (also the empty-state load before any proposal call); not traced to app source.
