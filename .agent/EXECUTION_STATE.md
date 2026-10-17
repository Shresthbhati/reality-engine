# Reality Engine — Execution State (current)

Rewritten 2026-10-17 from the full sprint history, which is preserved verbatim at
[`docs/archive/execution-history/EXECUTION_STATE_until_2026-10-17.md`](../docs/archive/execution-history/EXECUTION_STATE_until_2026-10-17.md)
(1700 lines; read it only for how a past decision was reached). The ledger is `.agent/TASKS.yaml`
(each task has `status` and a `reconciliation:` / `verification:` field); the roadmap is
`docs/implementation/ROADMAP.md`; the backlog is `docs/implementation/PENDING_IMPLEMENTATION.md`.
Git HEAD at rewrite: `3e452a6` (PR #145, prototype-completion), 2026-10-04.

## Ledger counts (computed from `.agent/TASKS.yaml`, 44 tasks)

| Status | Count | Ids |
|---|---|---|
| DONE | 37 | P0-01..03, P0-ROBUST-01/02, P1-01/02, P2-01/02, P3-01/03, P4-01, P5-01/02, P6-01/02/03, P7-01/02/04/06, P7-05-ADAPTIVE-SUBDIVISION, P8-01/02/03, P9-01, P10-01/02, P11-01, P12-01, P13-01/02, P15-01, P16-01, P17-01, P18-01, P19-01 |
| PARTIAL | 1 | PROD-01 |
| BLOCKED | 5 | P1-03, P3-02, P7-03, P7-05, P14-01 |
| OPTIONAL_DEFERRED | 1 | P16-02 |
| MISSING | 0 | none |

Roadmap numbering (P4.20 WorldStore, P6 Studio, ...) differs from ledger ids (P11-01, ...). The mapping is in
the ROADMAP Status column.

## Last verified results (taken from the history; nothing was re-run when this page was written)

| Date | What | Result |
|---|---|---|
| 2026-10-02/03 | Full non-slow suite (final sprint) | 2776 passed / 0 failed / 28 skipped / 74 deselected |
| 2026-10-02/03 | `tests/integration` (real COLMAP 4.2.0 CUDA + MiDaS, South Building) | 135 passed, plus 1 harness bug fixed and re-run green; 14 skipped (13+1 gitignored synthetic room fixtures not generated locally) |
| 2026-10-02/03 | 8 stage-synchronised real process kills; evidence order ABC vs BAC | pass; camera residual 0.0006 of scene extent (tolerance 0.01) |
| 2026-10-02/03 | compileall; frontend tsc / next build / contract tests / lint | 0 / 0 / 0 / 10 errors (pre-existing, files untouched) |
| 2026-10-03 | Export writers (P16-02 targets) | 23 tests green (`tests/test_box_exporters.py`) |
| 2026-10-03 | Core exports glTF/USD/IFC/CityGML/CityJSON/Blender (P16-01) | `tests/test_core_exports.py`, 17 tests, each output re-parsed with an independent reader (per ledger) |
| 2026-10-03 | Synthetic RGB-D / VIO | `tests/test_synthetic_rgbd.py` 13, `tests/test_synthetic_vio.py` 20 (ledger counts); indoor suite has strict xfails (PT-06) |
| 2026-10-03 | Clean-venv install | found and fixed missing `opencv-python-headless`, OpenEXR/imageio extras (commit dc35a79) |

No full-suite total is recorded after 2026-10-03. Commits `7687ba9`, `f698f49`, `f981edb` (2026-10-03/04)
changed code and tests; their runs are in git history and PRs #144-#145, not in this file.
Treat every count above as stale until re-run: `python -m pytest -m "not slow"`.

## BLOCKED items and the blocker

| Task | Blocker | What exists |
|---|---|---|
| P1-03 depth integration | No physical RGB-D device | 16-bit PNG/EXR, scale, invalid depth, unprojection, scale-coherence gate; synthetic RGB-D verified |
| P3-02 VIO federation | No built ORB-SLAM3/OpenVINS/Basalt backend (OpenVINS needs ROS/Linux, Docker engine not running), no EuRoC MH_01 | Adapters, selection, TUM parsing, diagnostics, clock sync, synthetic pipeline verified; nothing claimed about real backends |
| P7-03 architectural perception | No real indoor room/corridor photo dataset in the repo (`real_room_capture*`, `room_capture` are synthetic) | Geometry-only classifier, multi-storey fix, 12-case matrix, adversarial suite; real South Building yields one low-confidence (0.06) partial room |
| P7-05 detail budget | Universal calibration needs more real datasets (indoor, other cameras/scales); thresholds are empirical from ONE scene (`docs/engineering/DETAIL_CALIBRATION.md`) | Scale-aware budget (no mm GSD for non-metric worlds), robust-extent voxel, 15 tests |
| P14-01 city compiler | No 3DCityDB v5 (PostgreSQL/PostGIS) + citydb-tool; env-gated `TestRealDatabase` skipped | OSM/GeoJSON/CityJSON/CityGML/streaming/CLI; 12 fake-CLI contract tests; satellite/aerial/LiDAR ingestion unimplemented |

P16-02 (GIS, ROS/SDF, Habitat, SUMO, Unreal, Godot) is OPTIONAL_DEFERRED: writers exist and are
structure-tested; target-runtime acceptance is external.

## Known defects and limitations still open

Sources: `docs/KNOWN_LIMITATIONS.md`, `docs/implementation/PENDING_IMPLEMENTATION.md` (PT-06, CE-*, PROD.*).

- PT-06 (strict xfail, not hidden): rooms joined by doorways over one floor merge; stairs not wired into interior
  assembly; `detect_stairs` refuses unsegmented/unobserved-tread flights; phantom openings under realistic depth noise
  (sigma ~1.6 cm at 5 m); mesh validation too slow on ~50k-point clouds.
- PROD-01 PARTIAL: `Observation`/`Relationship` constructors default confidence to 1.0 (ledger "remaining"; commit
  f698f49 says defaults were pinned, not re-checked here); no real corridor/room photo set exists.
- CE-02 versions rebuilt from the union of all evidence (entity ids re-derived, cost grows with evidence; measure at
  100+ photos). CE-03 non-photo evidence does not constrain geometry. CE-04 UNKNOWN to OBSERVED transitions not
  asserted. CE-01 only camera-pose and planar-geometry conflicts implemented.
- CE-09 not covered: reconstruction-parse failure, subprocess timeout, disk-full, two parallel jobs from one HEAD,
  corrupted WorldStore artifact. CE-10: after a rollback the next run is a full rebuild.
- Studio: no uncertainty visualization; frontend lint had 10 pre-existing errors (2026-10-03).
- Environment: SAM torch-hub cache failure (1 failing perception test when included); no open3d/trimesh/skimage by
  design; gitignored synthetic room fixtures absent locally (14 integration skips).
- Carried from archived audits (added 2026-10-17): `WorldIRLegacy` removed from `world_ir/__init__`; `apps/api`
  now has bearer-token auth (`REALITY_API_TOKEN`, loopback-only when unset, fails closed on 0.0.0.0); ring tracing
  handles non-convex (L/U/T) rooms via a validated DFS trace. All three resolved this session (see
  "Session 2026-10-17" below).
- No ledger task: material perception (P2.17), advanced simulation (P8.31).

## Contradictions found while writing this page

- The history's last ledger line (2026-10-03: DONE 36 / PARTIAL 1 / BLOCKED 6) predates the prototype-completion
  sprint; `.agent/TASKS.yaml` now has P16-01 DONE and P16-02 OPTIONAL_DEFERRED. The YAML is authoritative.
- `PENDING_IMPLEMENTATION.md` P1.5 and P0.x entries and the PROD/CE tables were not re-audited on 2026-10-17.

## Session 2026-10-17 (this worktree, branch claude/project-improvement-3b3fb3)

Committed in 9 logical groups (all on this worktree; `.codex/` and `AGENTS.md` left untracked — user asked before
committing them):

1. Cleanup: removed .obsidian/.kilo/.freebuff/.serena state dirs + ad-hoc scratch files; anchored their ignores in
   .gitignore.
2. Docs: archived 8 audits under docs/archive/audits/; rewrote EXECUTION_STATE.md (1709 -> 92 lines); aligned
   IMPLEMENTATION_STATUS.md / PENDING_IMPLEMENTATION.md / ROADMAP.md statuses with code+tests.
3. Engine fixes+tests: colmap_backend (timeout labeling, typed model-parse error), colmap_session (heal committed
   model across kill-between-swap), artifact_store (atomic write + never-trust-a-corrupt-blob), worldstore +
   cli `store prune` retention; tests/test_failure_modes_*.py, tests/test_worldstore_retention.py.
4. Features: evidence/promote_rooms.py non-convex ring tracing; removed WorldIRLegacy export; tests/test_room_ring_tracing.py.
5. API auth: apps/api/auth.py (REALITY_API_TOKEN bearer, loopback-only when unset, missing peer address = local),
   apps/api/main.py middleware wiring, tests/test_api_auth.py, httpx<0.28 pin in pyproject.toml.
6. Frontend: shared lib/api/backend.ts backendFetch used by all 19 proxy routes; AdaptiveInspector uncertainty
   panel (confidence bar role=meter + "Low confidence" chip < 0.5); lint/type fixes.
7. CI: .github/workflows/ci.yml (nightly-only slow job; frontend lint/typecheck/build job).
8. Acceptance test: tests/integration/test_acceptance_p018.py (real South Building photos, real COLMAP/MiDaS,
   --no-mesh on anchored run, skip-if-not-available).
9. README: README.md + QUICKSTART.md rewritten to current state, QUICKSTART documents the API token.

Verification actually run this session (real runs, not inferred):
- tests/test_api_auth.py: 5 passed (token path incl. 200 with token / 401 without; health & preflight open; token never logged).
- Live HTTP check against uvicorn with REALITY_API_TOKEN=secret123: GET /api/health 200 (no token); protected 401 no token / wrong token / 200 correct token.
- Live HTTP check against Next.js proxy on :3000: /api/worlds 200, /api/sessions 200 (token forwarded server-side).
- tests/test_space_graph_api.py: 1 passed.
- tests/test_interior_adversarial.py::test_l_shaped_corridor: 1 passed (slow, 91.6s).
- tests/test_confidence_defaults.py, tests/test_room_ring_tracing.py, tests/test_worldstore_retention.py,
  tests/test_failure_modes_colmap.py, tests/test_failure_modes_worldstore.py: 41 passed.
- NOT run: test_interior_scene_e2e (timed out >120s even without slow marker), the full slow suite, and
  test_interior_adversarial as a whole (all 12 slow-marked). Full non-slow suite not re-recorded here.

Mesh stage (section 3): reconstruction/meshing/surface.py runs COLMAP `poisson_mesher` with a hard 1800s timeout
and raises MeshingError on timeout — it is SLOW, not hung. On South Building (~246k dense points, CPU) it had not
finished after 40+ min, so the acceptance test uses --no-mesh and the mesh criterion stays UNVERIFIED on South
Building (verified on the room capture per backlog P0.18). No decimation/GPU fix added (would change mesh quality;
the stage fails safely instead of hanging). Limit documented in the acceptance test and in KNOWN_LIMITATIONS.

Environment drift (section 4): global Python has httpx 0.27.2 (project pins httpx<0.28). skillspector requires
httpx>=0.28.0 and is broken by the global downgrade. Options: (a) move this project into a venv so httpx<0.28 is
isolated and global stays >=0.28 for skillspector, or (b) leave global as-is (project works, skillspector broken).
Pending user decision.

Ledger conflict (section 5): RESOLVED in favor of code. Commit f698f49 pinned Observation/Relationship/CausalRelation
constructor+from_dict defaults to 0.5 / Provenance.UNKNOWN (never 1.0/OBSERVED); tests/test_confidence_defaults.py
asserts this and that no producer omits confidence; PENDING_IMPLEMENTATION.md PROD.3 confirms. TASKS.yaml's PROD-01
"remaining" item was stale — updated to mark the confidence-defaults audit DONE (evidence: commit f698f49 +
test_confidence_defaults.py); the external-corridor/room-data item remains EXTERNAL DATA (UNVERIFIED).

## Exact next priorities (ledger order; none needs a new product decision)

1. PT-06 known limits: doorway room-merge, stairs into interior assembly, depth-noise phantom openings, slow mesh
   validation (each has a strict xfail to flip).
2. WorldIR consolidation (remove or migrate `world_ir/world.py`). (API authentication done this session.)
3. Re-run the full non-slow suite and `tests/integration` on HEAD and record dated counts here.
4. External-data blockers (P7-03, P7-05, P1-03, P3-02, P14-01): need real corridor/room photos, more calibration scenes,
   an RGB-D device, a built VIO backend + EuRoC MH_01, a 3DCityDB. Do not start without the data.
5. CE-02 rebuild-cost measurement at 100+ photos. (Quarantine retention DONE: `WorldStore.prune_quarantine` + `store prune`, tests/test_worldstore_retention.py; not auto-scheduled.)
6. Environment drift: venv vs global httpx — pending user decision (see "Session 2026-10-17").

Rules: never mark work complete without execution evidence; update this file and `.agent/TASKS.yaml` at the end of
every session; put sprint detail in a new file under `docs/archive/execution-history/`, not here.
