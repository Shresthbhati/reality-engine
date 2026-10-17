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
- Carried from archived audits (added 2026-10-17): two WorldIR classes still exported (`WorldIRLegacy`), `apps/api` has
  no authentication, room inference traces convex rings only. See `docs/KNOWN_LIMITATIONS.md`.
- No ledger task: material perception (P2.17), advanced simulation (P8.31).

## Contradictions found while writing this page

- The history's last ledger line (2026-10-03: DONE 36 / PARTIAL 1 / BLOCKED 6) predates the prototype-completion
  sprint; `.agent/TASKS.yaml` now has P16-01 DONE and P16-02 OPTIONAL_DEFERRED. The YAML is authoritative.
- `PENDING_IMPLEMENTATION.md` P1.5 and P0.x entries and the PROD/CE tables were not re-audited on 2026-10-17.

## Exact next priorities (ledger order; none needs a new product decision)

1. PROD-01 remainder (engineering): audit `Observation`/`Relationship` confidence defaults and pin with tests.
2. PT-06 known limits: doorway room-merge, stairs into interior assembly, depth-noise phantom openings, slow mesh
   validation (each has a strict xfail to flip).
3. WorldIR consolidation (remove or migrate `world_ir/world.py`); API authentication before any non-local deployment.
4. Re-run the full non-slow suite and `tests/integration` on HEAD and record dated counts here.
5. External-data blockers (P7-03, P7-05, P1-03, P3-02, P14-01): need real corridor/room photos, more calibration scenes,
   an RGB-D device, a built VIO backend + EuRoC MH_01, a 3DCityDB. Do not start without the data.
6. CE-02 rebuild-cost measurement at 100+ photos. (Quarantine retention DONE: `WorldStore.prune_quarantine` + `store prune`, tests/test_worldstore_retention.py; not auto-scheduled.)

Rules: never mark work complete without execution evidence; update this file and `.agent/TASKS.yaml` at the end of
every session; put sprint detail in a new file under `docs/archive/execution-history/`, not here.
