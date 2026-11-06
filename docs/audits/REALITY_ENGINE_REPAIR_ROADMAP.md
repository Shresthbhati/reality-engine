# Reality Engine — Repair Roadmap

**Updated:** 2026-11-06 · **Branch:** `claude/integration-2026-11-05` (user's local tip `2e2750b6` + `origin/main` `17eb3f40` + repairs below)
**Evidence rule:** every "resolved" row names a test or command that was actually run. "Not verified" means exactly that.

## 1. Resolved (with evidence)

| Item | Evidence |
|---|---|
| OpenAPI generation crashed (missing `Optional` import in `routes_assistant.py`) | `app.openapi()` returns 118 paths; `test_frontend_backend_contract.py` 21 passed |
| `RigidTransform` defined 3×, `NameError` in `world_ir/camera.py` | One class in `engine/core/rigid_transform.py`; `world_ir.camera` / `reconstruction.calibration.transforms` re-export (`is`-identity checked); registration, alignment, calibration, uncertainty tests pass |
| `MeshData` API regression (~30 tests) | Single `class MeshData` (`world_ir/geometry_data.py`); meshing, exporters, USD, Rule 7: 59 passed |
| Old WorldIR fixtures stopped round-tripping | `Entity.to_dict` omits an all-`None` `confidence_vector`; `test_contract_hierarchy` + 7.1 tests pass |
| Frontend type-check / lint broken | `tsc` 0 errors, ESLint 0 errors (50 pre-existing warnings), contract suite 385/385 (agent-run; re-run at end) |
| Undeclared dependencies, 4 dead tests, stale `engine/temporal/__init__` | `pyproject.toml` extras updated; `test_packaging_smoke` passes |
| Non-deterministic metric scale | Seeded RNG + determinism test |
| Hard-coded YOLOX/MaskDINO boxes (prior session, `4bd85e3a`) | Raise `DetectorUnavailable` |
| Fabricated OneFormer masks | Raises `SegmentationBackendUnavailableError`; `test_oneformer_no_fabrication` |
| XFeat/DeDoDe silently ran AKAZE/BRISK; availability probes lied | Truthful names + probes (`xfeat`, `dedode`, `midas`, `depth_anything`, `owlvit`, `sam` report precise reasons); 24 tests pass |
| `engine/technologies` benchmark fabricated VRAM/confidence/"Verified" | Simulated results can never pass/recommend; measured-or-null; 12 tests pass |
| Correction commit dropped cameras/report/sessions; coarse provenance; no image size/intrinsics in API job | `routes_worlds.py`, `jobs.py`; 28 focused tests pass (not exercised in a live COLMAP run) |
| Rate-limit bypass via `X-Tenant-Id`; invalid `X-Role` → ADMIN; unbounded buckets | `tests/test_auth_hardening.py` + existing auth tests: 24 passed |
| Correction payload rejected by backend (422); failed commit left undo entry; nav links to 404 routes | Payload accepted / old shape rejected by the real `_validate_transform`; contract tests |

## 2. Remaining — ranked

### P0 — before any feature work
1. **Get CI green and enforce it.** Push the integration branch, fix what CI reports, then branch protection. `main` has been red since 2026-10-28. Full Python suite and `next build` are **not verified** on this branch.
2. ~~**Tenant isolation:** only `routes_worlds.py` enforces it; 19 other route modules never read the auth context.~~ — **resolved 2026-11-06.** `apps/api/tenancy.py` wraps the existing `enforce_project_isolation` as router-level dependencies (path ids) plus list filters; every module that reads or writes worlds, sessions, uploads, evidence, jobs, notifications, activity or sites now answers a foreign id with 404 and the handler's own "not found" text. Sessions, worlds created by reconstruction/geo/compose, uploads, evidence and sites are stamped with the creator's `team_id` (`uploads`, `evidence`, `sites` gained the column; migration `0003`). `tests/test_tenant_isolation.py` drives every module as tenant A vs tenant B. Open: rows created before this change have no `team_id` and stay visible to every tenant (no data to infer an owner from); `registration`, `system`, `technologies` hold no tenant data and are unchanged.
3. ~~`FakeReconstructionBackend` as a production fallback~~ — **resolved 2026-11-06.** Removed from `UniversalPipeline._run_reconstruction` and the CLI `_default_backends`; test doubles are now injected only via explicit `backends=`. `tests/test_no_fake_backend_in_production.py` pins this. Also fixed: the default path passed a nonexistent `quality=` kwarg to `ColmapReconstructionBackend` and raised `TypeError` on every default run (so the fallback was unreachable anyway). Open: mapping `PipelineMode` to `dense_mvs` is a measured decision, not made here.
4. **GPU lifetime:** acceptance run peaked at 5,902 / 6,141 MiB with a CUDA OOM warning and the idle API held ~5.8 GiB. Release VRAM after each stage (see `RTX4050_APPLICATION_COMPATIBILITY.md`).
5. **Environment:** `timm`, `transformers`, `segment-anything` are declared in the `perception` extra but not installed in the venv, so depth, OWL-ViT and SAM are unavailable. Install, then re-run the acceptance journey.
6. Install `jsonschema` (declared in `dev`) — 4 schema-validation tests fail without it.

### P1 — make the reconstruction chain real
7. **Intrinsics on a first run:** current fix covers EXIF and re-runs; add a post-reconstruction hook that feeds COLMAP's own estimated intrinsics (labelled `estimated`) to depth/perception/meshing in the same job.
8. **Registration rate** (22/32 via API → dense gate refuses at <80%): try COLMAP 4.2's built-in `ALIKED_LIGHTGLUE` / `SIFT_LIGHTGLUE` and `global_mapper` (GLOMAP), benchmark against SIFT on this set.
9. **Wire or delete the unconnected modules** (≈37 of 148 added modules have no production caller): `run_incremental_ingest` (6.3), `engine/pipeline/coverage.py` → capture requests (5.3→6.1), `crop_pass` (9.1), geo anchor (9.3), texture projection (10.1), completion proposals (10.3). Decide per module; do not mark DONE without a named caller.
10. **Structure compiler infers rooms/corridors/windows from an exterior facade.** Require an evidence gate; never manufacture structure.
11. **`engine/technologies` adapters:** 37 return constants, 13 are numpy re-derivations; wrap the real upstream library or remove from the catalog.

### P2 — spatial intelligence and validation (needs data)
12. **Real interior, multi-room data + tape measurements** (Task 2.3 — *blocked on the user*); public sets (ScanNet++, ARKitScenes) need access requests.
13. Task 3.6 identity scenarios; Phase 12 acceptance on interior data.
14. Detector decision for doors/windows/fixtures (OWLv2 vs Grounding DINO) — *user decision*.

### P3 — Studio
15. Browser verification (Playwright) of the full journey; Studio defaults to Simple mode so the inspector/corrections are Pro-only.
16. Server-side undo/rollback in the UI (backend `rollback` exists, no UI); `parent_version_id` should be the viewed version.
17. Reports page has no backend binding (404 unless `NEXT_PUBLIC_SHOW_UNFINISHED=1`); ~28 backend routes have no UI caller.
18. Split `WorldNavPanel.tsx` (3,063 lines), `three-scene.ts` (2,835), `AdaptiveInspector.tsx` (2,073).

### P4 — advanced
Neural reconstruction (Gaussian splatting), BIM/GIS depth, large-world — only after P0–P2.

## 3. Process rules (to stop the drift recurring)
- No merge to `main` unless CI is green.
- "DONE" requires a named production caller **and** a recorded run (not a test-generated file) for any benchmark claim.
- One status ledger: `docs/blueprint/30_BUILD_PLAN.md` + `.agent/TASKS.yaml`; retire the duplicate roadmap/matrix documents.
- One PR per task; no 12-task squash PRs.

## 4. Blocked on the user
Real interior captures + tape measurements; detector decision; permission to push the integration branch and open a PR.
