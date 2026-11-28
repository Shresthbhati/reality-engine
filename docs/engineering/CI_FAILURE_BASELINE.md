# CI Failure Baseline

Measured 2026-11-28. Main at **ff73ea90** (`feat(perception): Phase 4 & 6 multi-view identity rekeying ...`), CI run
46124752079; reproduced locally in a torch-free CI-like environment (Python 3.13 venv, `pip install -e ".[dev]"`, no torch,
no GPU libraries) by running all 8 `full` shards with `-m "not slow"`, and on the workstation environment (Python 3.14.7,
torch 2.14+cu130, COLMAP 4.2.0, RTX 4050). Raw logs: GitHub runs 46124752079 / 46124614788; per-shard JUnit summarised in
`ci_diagnostics_baseline.json`. "Fixed" below means fixed **in the PR that adds this file**; nothing was fixed by skipping,
deleting or loosening a test unless the old assertion was demonstrably stale (named per row).

## 1. Job-level state of main at ff73ea90

| job | result | why |
|---|---|---|
| Core Tests 3.11 | fail | 1 test (gltf) |
| Core Tests 3.10 / 3.12 | **cancelled** | matrix `fail-fast` default: 3.11's failure cancelled them, so their results were never seen (fixed: `fail-fast: false`) |
| Full Suite shards 1,3,4,5,6,7 | fail | see section 2 |
| Full Suite shard 2 | cancelled | the run was superseded |
| Full Suite shard 8 | pass | |
| Lint & Type Check | fail | Ruff 4262 > baseline 4048, Mypy 807 > 770 |
| Frontend / Room Fidelity / Security | pass | |
| Slow (nightly) | skipped on push | runs on schedule only |
| Playwright E2E | pass | |

The earlier run 46124614788 (feature branch) additionally showed **4 shards interrupted by collection errors**
(`ModuleNotFoundError: No module named 'torch'` at import of `tests/test_a5_*`, `test_segmentation_adapter_mask.py`):
one error stops the whole shard, hiding every other test in it. Already fixed on main by #416 (SAM2 adapter);
`tests/test_dependency_contract.py` now guards the whole class.

## 2. Test failures: classification, evidence, state

Categories: **P** product defect, **T** test defect (stale/wrong assertion), **E** dependency/environment, **N** native runtime
crash, **F** flaky, **D** stale docs/registry, **U** unknown.

| # | test(s) | where seen | cat | root cause (observed) | state |
|---|---|---|---|---|---|
| 1 | `test_gltf_exporter::test_mesh_geometry_without_resolvable_artifact_falls_back_to_cube` | core, shard 6 | T | #391 deliberately made the exporter skip-and-name an unresolvable payload instead of drawing a fabricated cube (covered by `test_evidence_export_geometry.py`); this older test still asserted the cube | test now asserts the honest contract (no node, payload named) |
| 2 | `test_evidence_firewall::test_duplicate_upload_is_flagged_against_the_earlier_copy` | shard 1 | T | #415 made same-session identical uploads idempotent (same Evidence row); the test uploaded twice in one session | test asserts the dedup AND the cross-session flag |
| 3 | `test_artifact_integrity::test_export_download_filename_safe` | shard 7 | T | #415 requires a World + `ExportRecord` for a download; the test served bare bytes for a made-up id (`wld_ok`; ids are validated) | test seeds a real world + record, keeps the header-injection assertions, adds isolation + hostile-id assertions |
| 4 | `test_multiengine_refinement::test_engines_endpoint_...` | shard 7 | T | endpoint lists the 6-engine baseline registry; `real_models_by_name()` now has 11 | exact 6-engine set AND subset of probed |
| 5 | `test_matcher_policy` x4 | shard 4 | T + P | tests hard-coded "torch+kornia present"; **and** `benchmarked_here()` reported unavailable matchers as benchmarked, contradicting its own docstring | policy fixed (`status == "available"`); tests assert LightGlue when available and SIFT otherwise, both envs verified |
| 6 | `test_ontology_grounded_detection` x5, `test_small_object_phase3::...partial`, `test_engine_reachability_...::TestMaskRcnn*unavailable_branch` | shards 4, 6 | P | `perception/detection_adapter.py` (and `maskrcnn_adapter.py`, since refactored upstream in #424) imported torch unguarded, so importing the module failed without the optional extra | torch import guarded; all pass without torch |
| 7 | `test_engine_reachability` x5 (real Mask R-CNN / MiDaS / wired engines), `test_evidence_world_bridge_golden::test_real_rtx4050_models...`, `test_sequential_maximum_fidelity` x2, `test_a5_...::test_18` | shards 3, 4 | E | need torch/torchvision/transformers + real weights; the CI image has none | `@pytest.mark.requires(...)`: visible skip by default, **failure** under `RE_STRICT_REQUIREMENTS=1` |
| 8 | `test_december_perception_suite::test_available_probes_for_real_cached_models` | shard 5 | E | asserts "all three models cached on this workstation" | `requires` + three `hf_cache:` capabilities |
| 9 | `integration/test_real_photo_acceptance` x2 | shard 6 | E | needs `datasets/real_room_capture` (not committed); failed on `assert len(items) == 4` instead of declaring the dependency | module `requires("colmap", "dataset:real_room_capture")` |
| 10 | `test_entity_perception::test_taxonomy_covers_all_eighteen...` | shard 5 | P | `config/ontology.yaml`: `envelope.molding` / `envelope.trim` had `parent: architecture/wall` (another ontology's id) | parents point at `envelope.wall` |
| 11 | `test_a5_gpu_lifecycle...::test_13_gpu_memory_exhaustion...` | shard 4 | T | `run_stage` re-probes REAL hardware; on a CPU-only runner the simulated CUDA OOM had no CPU fallback left | probe pinned to the simulated 6 GB GPU |
| 12 | `test_ci_test_registry::test_doc_is_current` | shard 5 | D | `docs/CI_TEST_REGISTRY.md` was hand-regenerated and stale | registry generated from discovery (`docs/engineering/CI_TEST_REGISTRY.md`) |
| 13 | `test_import_boundaries::test_no_new_boundary_violations` | shard 7 / local | P | 7 forbidden imports (`apps.api -> perception` x2, `perception -> reconstruction` x3, `perception -> evidence` x2) | **NOT fixed, remains red.** Architectural refactor in perception/api code; DECISION REQUIRED, owner: perception. The test says "fix the import, do not allow-list it" and I did not |
| 14 | Lint: Ruff +214, Mypy +37 vs baseline | lint | P | debt landed on main via merged PRs | per-file ratchet baseline written at the measured state (4265 / 807); this branch adds none (4265 vs 4267 on current main). The one-time `--allow-increase` accepts what is already on main |
| 15 | `test_cross_world_undo_isolation`, `test_same_world_concurrent_completion_single_head` | branch runs before #418 | P/F | concurrent same-world commit; fixed by #418 (merged 08:31 UTC) | 0/15 failures each on current main, locally |
| 16 | `test_refine_world_api::test_a_registered_engine_is_used...`, SQLite "malformed" in `test_application_api` | local shards | F | see `FLAKY_TEST_REPORT.md` | reported, not fully root-caused |
| 17 | workstation-only: `test_real_perception_benchmark::test_hardware_telemetry_reporting` | local | E | `psutil` (declared `resources` extra) missing from `dev`; no `resource` module on Windows | `resources` added to `dev` |
| 18 | workstation-only: `test_spatial_stage::test_aligned_depth...` | local | E | plane extraction needs open3d (`geometry` extra); the stage reports the failure honestly | `requires("open3d")` |
| 19 | `tests/integration/test_release_gate.py` | workstation | N? / U | no native crash reproduced; job grades `PARTIAL`, gate asserts `succeeded`; free-RAM dependent | see `NATIVE_RUNTIME_DIAGNOSTICS.md`; reconstruction owner |

## 3. Defects in the CI machinery itself (found while reproducing)

| defect | effect | state |
|---|---|---|
| `pytest tests/ --collect-only -q 2>&1 \| tail -20` | exit status was `tail`'s, so a collection error could never fail the step | replaced; plus `ci_test_registry.py --verify-collection` |
| matrix `fail-fast` default on `core` | one red Python version hid the other two | `fail-fast: false` |
| `scripts/ci_shard_tests.py` printed `\r\n` on Windows | stray `\r` on every path; pytest "file not found" for the whole shard | LF-only manifest + test |
| `world/` and `geometry/` missing from `[tool.setuptools.packages.find]` | wheel / non-root installs cannot import `engine`, `sdk`, `apps.api` | added; contract test |
| `shapely`, `networkx`, `pyyaml` undeclared | worked only through `ifcopenshell` / `pre-commit` | declared |
| `tests/test_deployment_and_docs.py` overwrote `docs/openapi.json` and `docs/API_DOCUMENTATION.md` | every test run dirtied the tree and hid a stale catalogue | writes to `tmp_path`; separate staleness test (the committed docs must be regenerated when routes change) |
| `py -3.11` launcher points at a missing interpreter on the workstation | cannot reproduce 3.10-3.12 locally | documented, not fixed |

## 4. Counts

Local torch-free baseline (8 shards, `-m "not slow"`, before the fixes): 6972 tests, 6800 passed, 39 failed, 2 errors, 131 skipped.
After this PR every failure in the table is fixed except #13, #16 and #19; re-verification is in the PR description.
Frontend (Node 26.8.1): ESLint 0 errors / 82 warnings, `tsc --noEmit` 0 errors, contract tests pass, `next build` succeeds
(a first `tsc` run showed 6 errors only because `node_modules` was absent: environment, not code).
