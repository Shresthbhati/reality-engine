# 35 — Task Cards

Work queue for `34_MASTER_PLAN.md`. Read its section 9 (Runbook) before starting any card.

**Tier:** `[N]` Nemotron/Qwen-class · `[S]` strong model · `[H]` human.
**Every card:** `Check` is step 0 — if its output differs from what the card states, stop and report. Touch only `Files`. One commit per card.
**Default regression** (called `REG` below):
`pytest tests/test_golden_path_api_to_glb.py tests/test_durable_job_queue.py tests/test_world_adoption_api.py -q --tb=short`
**Default frontend check** (called `FE` below):
`cd frontend && npx tsc --noEmit && npm run lint && npm run test:contract`

Baseline for all line numbers: `199ca3bb`. Line numbers drift; symbols do not — locate by symbol.

---

# Phase P0 — Truth

### T001 [N] Generate the CI test registry
**Needs:** — · **Files:** create `scripts/ci_test_registry.py`, `tests/test_ci_test_registry.py`, `docs/CI_TEST_REGISTRY.md`
**Check:** `grep -c "tests/test_" .github/workflows/ci.yml` → a number ≥ 50.
- [ ] Write `build_registry(ci_yaml: Path, tests_dir: Path) -> dict[str, list[str]]`. Parse `ci.yml` with `yaml.safe_load`. Keys: `core` (paths in job `core`, step "Run core test suite"), `fidelity_gate` (paths in job `fidelity-gate`), `collect_only` (always `["tests/ (collection only, nothing executed)"]`), `unlisted` (every `tests/**/test_*.py` not in `core` or `fidelity_gate`), `advisory` (`["ruff --exit-zero", "mypy || true"]`).
- [ ] Write `render(registry) -> str` producing Markdown with one `##` heading per key and a count line `Total: N`.
- [ ] `main()` writes `docs/CI_TEST_REGISTRY.md`; `--check` exits 1 if the file on disk differs.
**Test:** `test_core_contains_golden_path` asserts `"tests/test_golden_path_api_to_glb.py" in reg["core"]`; `test_unlisted_contains_gaussian_splats` asserts `"tests/test_gaussian_splats.py" in reg["unlisted"]`; `test_no_overlap` asserts `set(core) & set(unlisted) == set()`; `test_doc_is_current` runs `--check` and asserts exit code 0.
**Run:** `pytest tests/test_ci_test_registry.py -q` · **Stop if:** `ci.yml` has no job named `core`.

### T002 [N] Time the eleven unlisted suites (report only)
**Needs:** — · **Files:** create `docs/audits/2026-11-unlisted-suites.md`
**Check:** `ls tests/test_gaussian_splats.py tests/test_no_fake_backend_in_production.py` → both exist.
- [ ] Run each file alone with `pytest <file> -q --tb=line --durations=3`: `test_gaussian_splats`, `test_temporal_delta`, `test_mep_topology`, `test_material_takeoff_boq`, `test_life_safety_compliance`, `test_thermal_envelope_ashrae`, `test_universal_spatial_pipeline_rules11_12`, `test_no_fake_backend_in_production`, `test_clash_detection_bcf`, `test_collab_telepresence`, `test_section_elevation_drawings` (all under `tests/`).
- [ ] Write a table: file · passed · failed · skipped · wall seconds · last output line verbatim.
**Done when:** eleven rows, every number copied from real output. **Stop if:** a run needs network or a GPU — record "not runnable here: <reason>".

### T003 [N] Put the fast passing suites into required CI
**Needs:** T001, T002 · **Files:** `.github/workflows/ci.yml`, `docs/CI_TEST_REGISTRY.md`
- [ ] For each T002 row with `failed = 0`, `skipped = 0` and under 60 s, add its path to the `Run core test suite` list, one per line, before `-v --tb=short`.
- [ ] Do not add rows that failed, skipped everything, or took 60 s or more. List those in the commit message.
- [ ] Regenerate the registry: `python scripts/ci_test_registry.py`.
**Run:** `pytest tests/test_ci_test_registry.py -q` then run the added files together once. **Stop if:** T002 shows any failure in `test_no_fake_backend_in_production.py` — that goes to T005 first.

### T004 [N] Remove "Certified" from life-safety defaults
**Needs:** — · **Files:** `apps/api/routes_life_safety.py`, `engine/pipeline/life_safety_compliance.py`, `tests/test_life_safety_compliance.py`
**Check:** `grep -rn "Certified Auditor" --include=*.py .` → exactly two hits (the two files above). If a test also matches, it is in scope.
- [ ] Replace both defaults with `"Automated preliminary screening (not professionally reviewed)"`.
- [ ] In the PDF builder, ensure this sentence is drawn on page 1: `Preliminary automated screening. Not a code-compliance certificate. Requires review by a qualified professional.`
**Test:** `test_default_author_is_not_certified` asserts `"Certified" not in` the default value of `architect_name` (read via `inspect.signature`); `test_pdf_carries_disclaimer` generates the PDF with defaults and asserts `b"Not a code-compliance certificate"` appears in the extracted text (use whatever text-extraction helper the existing tests in this file use; if none, assert on the list of strings passed to the PDF canvas).
**Run:** `pytest tests/test_life_safety_compliance.py -q` · **Stop if:** the word "certificate" is part of a public route path or response key — report, do not rename.

### T005 [S] Triage failing unlisted suites
**Needs:** T002 · **Files:** only the failing test file and the single module it tests.
For each failing suite decide: stale expectation (fix test, explain), real regression (fix code, add to CI), or environment-bound (mark `@pytest.mark.slow` with a reason string). One commit per suite. Never weaken an assertion without stating what behaviour changed and when.

### T006 [N] Specialist responses declare their status
**Needs:** T004 · **Files:** `apps/api/routes_takeoff.py`, `apps/api/routes_thermal_envelope.py`, `apps/api/routes_mep.py`, `apps/api/routes_life_safety.py`, create `tests/test_specialist_truth_labels.py`
**Check:** `grep -n "^@.*\.get(\|^@.*\.post(" apps/api/routes_takeoff.py apps/api/routes_thermal_envelope.py apps/api/routes_mep.py apps/api/routes_life_safety.py` → note each JSON-returning handler.
- [ ] Add, **additively**, two top-level keys to every JSON analysis response: `"validation_status": "preliminary"` and `"assumptions": [<strings>]`. Take assumption strings from the handler's own defaults (e.g. occupancy, sprinklered, climate zone, ΔT, unit-rate source). Do not rename or remove any existing key.
- [ ] MEP responses additionally carry `"basis": "inferred_from_world_entities"`.
**Test:** one test per router calling the main analysis endpoint through the same fixture pattern the router's existing test file uses, asserting `body["validation_status"] == "preliminary"` and `len(body["assumptions"]) >= 1`.
**Stop if:** a response is a Pydantic model with `extra="forbid"` consumed by a frontend type — report; needs an `[S]` contract decision.

### T007 [N] One job-status parser for the frontend
**Needs:** — · **Files:** create `frontend/src/lib/api/job-status.ts`, `frontend/tests/contract/job-status.test.mts`
**Check:** `grep -n "JOB_" apps/api/jobs.py | head -8` → shows `queued running partial succeeded failed cancelled`.
- [ ] Export `export type JobStatus = "queued" | "running" | "partial" | "succeeded" | "failed" | "cancelled" | "unknown";`
- [ ] Export `export function toJobStatus(raw: unknown): JobStatus` — lower-cases strings, returns the matching literal, returns `"unknown"` for anything else (including `null`, numbers, `"SUCCESS"`, `""`).
- [ ] Export `export function isTerminal(s: JobStatus): boolean` — true for `partial`, `succeeded`, `failed`, `cancelled`; false for `queued`, `running`, **and `unknown`**.
- [ ] Export `export function isUsableResult(s: JobStatus): boolean` — true only for `succeeded` and `partial`.
**Test:** assert `toJobStatus("Succeeded") === "succeeded"`, `toJobStatus("done") === "unknown"`, `toJobStatus(undefined) === "unknown"`, `isUsableResult("unknown") === false`, `isUsableResult("failed") === false`, `isTerminal("unknown") === false`. Copy the import/runner style of an existing file in `frontend/tests/contract/`.
**Run:** `FE`.

### T008 [N] Adopt the parser where job status is read
**Needs:** T007 · **Files:** `frontend/src/lib/api/reconstructions.ts`, `frontend/src/components/reconstruct/ReconstructionStatusBar.tsx`, `frontend/src/components/session/ReconstructPanel.tsx`, `frontend/src/lib/dashboard/attention-model.ts`, `frontend/src/lib/dashboard/dashboard-model.ts`
**Check:** `grep -n "succeeded\|partial" <each file>`.
- [ ] Replace every direct comparison of a raw backend status string with `toJobStatus(...)` and a `switch` that has an explicit `case "unknown":` rendering the text `Status unknown — refresh or check the job log` (never a success style).
- [ ] No `default:` branch may fall through to a success or ready state.
**Run:** `FE` · **Stop if:** a file derives status from a field other than `status`/`grade` — report which.

### T009 [N] Licence gate for models
**Needs:** — · **Files:** `perception/model_registry.py`, create `tests/test_model_licence_gate.py`
**Check:** `grep -n "CC-BY-NC" perception/model_registry.py` → at least one hit (`mast3r_vitlarge`).
- [ ] Add `ALLOWED_PRODUCTION_LICENCES: frozenset[str] = frozenset({"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"})`.
- [ ] Add `class LicenceNotAllowedError(RuntimeError)`.
- [ ] Add `def assert_production_licences(model_names: Iterable[str]) -> None` — looks each name up with the registry's existing lookup function; raises `LicenceNotAllowedError` naming the model and its licence if the licence is not allowed or is empty; raises `KeyError` for an unknown name.
**Test:** `assert_production_licences(["sam2_hiera_tiny"])` returns `None`; `["mast3r_vitlarge"]` raises `LicenceNotAllowedError` whose message contains `"CC-BY-NC"`; `["no_such_model"]` raises `KeyError`; `test_every_registry_entry_has_a_licence` iterates the registry and asserts no empty licence string.
**Run:** `pytest tests/test_model_licence_gate.py -q` · **Stop if:** the last test fails on existing entries — report the names; do not invent licences.

### T010 [N] Status ledger from one YAML file
**Needs:** — · **Files:** create `docs/status.yaml`, `scripts/render_status.py`, `docs/STATUS.md`, `tests/test_status_ledger.py`
- [ ] `docs/status.yaml`: a list of items `{id, capability, status, gate, evidence, tests, updated, next}`. Seed it with every row of section 5 of `34_MASTER_PLAN.md`. `status` ∈ `MISSING, FOUNDATION, PARTIAL, UNWIRED, IMPL-UNVERIFIED, CONTRACT-VERIFIED, FIELD-VALIDATED, DEFERRED`. `gate` ∈ `G0, G1, G2, G3, G4`. `updated` is `2026-11-10`.
- [ ] `render_status.py`: loads the YAML, validates the two enums and that `evidence` is non-empty for any status above `FOUNDATION`, writes a Markdown table to `docs/STATUS.md` with header line `Baseline: <git rev-parse --short HEAD>`; `--check` exits 1 on drift.
**Test:** `test_enums_valid`, `test_field_validated_requires_dataset` (any `FIELD-VALIDATED` item must have an `evidence` path starting with `benchmarks/`), `test_doc_is_current`.
**Run:** `pytest tests/test_status_ledger.py -q`.

### T011 [N] Banner and archive superseded status documents
**Needs:** T010, owner approval of item 7 in plan section 8 · **Files:** the listed Markdown files only.
- [ ] Insert as line 1 of each: `> **SUPERSEDED (2026-11-10).** Current status lives in [docs/STATUS.md](docs/STATUS.md); current plan in docs/blueprint/34_MASTER_PLAN.md. Kept for history.` (adjust the relative link per file depth).
- [ ] Files: `docs/implementation/IMPLEMENTATION_STATUS.md`, `docs/implementation/PENDING_IMPLEMENTATION.md`, `docs/implementation/CAPABILITIES.md`, `docs/REALITY_ENGINE_CURRENT_STATUS.md`, `docs/blueprint/33_TURNAROUND_PLAN.md`, `docs/blueprint/20_REUSE_MAP.md`, `UPDATED_CAPABILITY_MATRIX.md`, `UPDATED_IMPLEMENTATION_BACKLOG.md`, `IMPLEMENTATION_DEPENDENCY_GRAPH.md`, `CODING_AGENT_ASSIGNMENTS.md`.
- [ ] Do not move or delete anything in this card.
**Done when:** `grep -L "SUPERSEDED (2026-11-10)" <the ten files>` prints nothing. **Stop if:** `CLAUDE.md` or `AGENTS.md` cites one of these as authoritative — report the line; an `[S]` card updates those references.

### T012 [S] Test backend cannot run outside tests
**Needs:** — · **Files:** `apps/api/jobs.py`, `conftest.py`, `tests/test_no_fake_backend_in_production.py`
**Check:** `grep -n "REALITY_TEST_BACKEND" apps/api/jobs.py` and read `tests/test_no_fake_backend_in_production.py` to see what it already asserts.
- [ ] In `_run_reconstruct_session_inner`, when `REALITY_TEST_BACKEND` is non-empty and `os.environ.get("REALITY_ENV") != "test"`, raise `RuntimeError("REALITY_TEST_BACKEND is set outside REALITY_ENV=test; refusing to run a test backend")` before any evidence is read.
- [ ] In root `conftest.py` add `os.environ.setdefault("REALITY_ENV", "test")`.
**Test:** new test `test_test_backend_refused_outside_test_env` — monkeypatch `REALITY_ENV` to `"production"`, set `REALITY_TEST_BACKEND` to the living-room backend spec, enqueue and process one job, assert `job.status == "failed"` and `"refusing to run a test backend" in job.error`.
**Run:** the file plus `REG`.

---

# Phase P1 — One pipeline

### P101 [S] Stage contract and runner (pure, no database)
**Needs:** — · **Files:** create `engine/pipeline/stages/__init__.py`, `base.py`, `runner.py`, `tests/test_stage_runner.py`
- [ ] Implement `StageSpec`, `StageResult`, `Stage`, `RunManifest` exactly as in plan section 3.3.
- [ ] `StageContext`: `job_id: str`, `workdir: Path`, `artifacts: dict[str, str]` (logical name → sha256, grows as stages finish), `params: dict`, `state: dict` (in-memory objects passed between stages in one process), `log: Callable[[str], None]`.
- [ ] `def compute_inputs_hash(version: str, input_hashes: Mapping[str, str], params: Mapping) -> str` — sha256 over `json.dumps([version, sorted(input_hashes.items()), params], sort_keys=True, default=str)`.
- [ ] `class CheckpointStore(Protocol)`: `lookup(job_id, name, inputs_hash) -> StageResult | None`, `record(job_id, result: StageResult) -> None`. Provide `NullCheckpointStore`.
- [ ] `def run_stages(stages: Sequence[Stage], ctx: StageContext, *, checkpoints: CheckpointStore = NullCheckpointStore(), should_cancel: Callable[[], bool] = lambda: False, on_start: Callable[[StageSpec], None] | None = None, on_end: Callable[[StageResult], None] | None = None) -> RunManifest`.
  Behaviour: check `should_cancel()` before each stage and raise `StageCancelled`; an exception inside `stage.run` becomes a `failed` `StageResult` with `error=repr(exc)`; a failed **required** stage stops the run; a failed optional stage continues; a checkpoint hit is appended unchanged and `stage.run` is not called.
**Test:** `test_runs_in_order`, `test_required_failure_stops_and_grades_failed`, `test_optional_failure_continues_and_grades_partial`, `test_degraded_grades_partial`, `test_skipped_does_not_change_grade`, `test_cancel_before_second_stage_raises`, `test_checkpoint_hit_skips_run` (counter stays 0), `test_exception_becomes_failed_result`.
**Run:** `pytest tests/test_stage_runner.py -q`.

### P102 [N] Property tests for `compute_inputs_hash`
**Needs:** P101 · **Files:** `tests/test_stage_runner.py`
**Test:** same inputs in different dict order → equal hash; changing `version` → different; changing one artifact hash → different; changing one param → different; result is 64 lowercase hex characters.

### P103 [N] `JobStageRun` table
**Needs:** P101 · **Files:** `apps/api/models.py`, create `apps/api/migrations/versions/0005_job_stage_runs.py`, create `tests/test_job_stage_runs_model.py`
**Check:** `ls apps/api/migrations/versions` → newest is `0004_tenant_isolation_tables.py`; open it and copy its revision-id style. If a `0005_*` already exists, use the next number.
- [ ] Model `JobStageRun(Base)`, `__tablename__ = "job_stage_runs"`: `id` String(32) pk; `job_id` String(32) indexed, FK `jobs.id`; `team_id` String(32) nullable indexed; `name` String(64); `version` String(32); `inputs_hash` String(64); `status` String(16); `attempt` Integer default 1; `outputs` JSON nullable; `metrics` JSON nullable; `warnings` JSON nullable; `error` Text nullable; `peak_rss_mb` Float nullable; `started_at`, `ended_at` DateTime(timezone=True). Unique constraint on `(job_id, name, inputs_hash, attempt)`.
- [ ] Alembic migration creating exactly that table, `down_revision` = newest existing id.
**Test:** insert one row through the same async session fixture `tests/test_durable_job_queue.py` uses, read it back, assert every field; inserting a duplicate `(job_id, name, inputs_hash, attempt)` raises `IntegrityError`.
**Run:** the new test + `REG`.

### P104 [S] Database checkpoint store and resume
**Needs:** P101, P103 · **Files:** create `apps/api/stage_checkpoints.py`, `tests/test_stage_checkpoints.py`
`DbCheckpointStore(session_factory, team_id)` implementing `CheckpointStore`. `lookup` returns only rows with `status == "succeeded"` whose every output sha resolves through `apps.api.storage.resolve_artifact`. `record` inserts with `attempt = previous max + 1`. Tests: hit, miss on different hash, miss when an artifact file was deleted, attempt increments.

### P105 [N] `GET /api/jobs/{job_id}/stages`
**Needs:** P103 · **Files:** `apps/api/routes_jobs.py`, create `tests/test_job_stages_api.py`
**Check:** open the `GET /{job_id}/logs` handler in `routes_jobs.py`; you will copy its auth and tenant-ownership lines verbatim.
- [ ] Add handler returning `{"items": [{"name","version","status","attempt","metrics","warnings","error","started_at","ended_at","duration_s"}...]}` ordered by `started_at`. Unknown job → 404. Job of another tenant → same status code the `/logs` handler returns for that case.
**Test:** `test_stages_empty_list_for_new_job` (200, `items == []`), `test_stages_returns_rows_in_order`, `test_unknown_job_404`, `test_other_tenant_denied`.
**Run:** new test + `REG`.

### P106 [S] Job function runs through the runner (coarse stages)
**Needs:** P101, P104, T012 · **Files:** `apps/api/jobs.py`, create `engine/pipeline/stages/registry.py`, `s01_evidence.py`, `s02_quality.py`, `s04_reconstruct_legacy.py`, `s11_compile_validate.py`, `s13_changeset.py`
Cut `_run_reconstruct_session_inner` at its existing `job.stage = ...` boundaries into five stages: `evidence_resolve`, `evidence_quality`, `reconstruct_legacy` (the whole `vertical_slice()` call, required), `compile_validate`, `changeset_commit`. Objects pass through `ctx.state`. `on_start` sets `job.stage`; `on_end` records via `DbCheckpointStore`; `should_cancel` reads `job.cancel_requested`. Job outcome = `manifest.grade(specs)`; delete `_stage_facts_degraded` only when its cases are covered by stage statuses. No behaviour change otherwise: `REG`, `tests/test_edit_preservation_reprocess.py` and `tests/test_observability_job_logs.py` must pass unchanged.

### P107 [S] Split pose and scale out of `reconstruct_legacy`
**Needs:** P106 · **Files:** `engine/pipeline/vertical_slice.py`, `engine/pipeline/stages/s04_pose.py`, `s05_scale.py`
Extract the backend run + `WorldArbiter` selection into `pose`; the scale block into `scale`. `vertical_slice()` keeps its public signature and calls the same extracted functions, so the CLI path (`tests/test_cli_vertical_slice.py`) is unaffected.

### P108 [S] Split depth, dense, mesh, perception, layout; remove `reconstruct_legacy`
**Needs:** P107 · **Files:** `engine/pipeline/vertical_slice.py`, `engine/pipeline/stages/s06…s10_*.py`, `registry.py`
One commit per stage. After the last, the profile matches plan section 3.4 and `s04_reconstruct_legacy.py` is deleted. Each optional stage returns `skipped` with a reason when its backend is unavailable — never `succeeded`.

### P109 [N] Resume test: a crashed run reuses finished stages
**Needs:** P104 · **Files:** create `tests/test_stage_resume.py`
- [ ] Build three fake stages with call counters; stage 3 raises `SystemExit` on its first call only (simulate a killed worker by catching it in the test).
- [ ] Run `run_stages` twice with the same `DbCheckpointStore` and the same `ctx.job_id`.
**Test:** after run 2, counters are `[1, 1, 2]`; manifest grade is `succeeded`; stage 1 and stage 2 each have exactly one `job_stage_runs` row.

### P110 [N] `quality_report` stage: geometry validity
**Needs:** P108 · **Files:** create `engine/pipeline/stages/s12_quality_report.py`, `tests/test_quality_report_stage.py`
**Check:** `grep -n "trimesh" pyproject.toml` → present under the `meshing` extra (optional import).
- [ ] `def mesh_validity(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | str]` returning keys `watertight` (0.0/1.0), `non_manifold_edges`, `degenerate_faces`, `components`, and `verdict` ∈ `"PASS" | "WARN" | "FAIL" | "NOT_EVALUABLE"`. When `trimesh` cannot be imported return `{"verdict": "NOT_EVALUABLE", "reason": "trimesh not installed"}`.
- [ ] Verdict: `FAIL` if `degenerate_faces > 0` or `components > 1`; `WARN` if not watertight; else `PASS`. A room shell open at doorways is expected to be `WARN`, not `FAIL`.
- [ ] Stage `quality_report` (cpu, required) writes these under `metrics` with prefix `mesh_`; if there is no mesh artifact it records `mesh_verdict = "NOT_EVALUABLE"`.
**Test:** closed cube → `PASS`; cube with one face removed → `WARN`; two disjoint cubes → `FAIL`; with `trimesh` import patched to fail → `NOT_EVALUABLE`; assert the string `"PASS"` never appears when verdict is `NOT_EVALUABLE`.
**Run:** `pytest tests/test_quality_report_stage.py -q`.

### P111 [S] Architectural constraint checks in `compile_validate`
**Needs:** P108 · **Files:** `world_ir/validation.py` (or a new sibling module if that file exceeds ~600 lines), tests
First inventory what `world_ir/validation.py` and `perception/architecture/topology.py` already check; add only the missing ones: opening without a host wall, opening outside its host wall's extent, wall base not at its storey floor elevation (tolerance from config), stair not connecting two storeys. Violations are returned as data (`ConstraintViolation(code, entity_ids, message, severity)`), and `error` severity makes the stage `degraded`, not `failed`.

### P112 [S] Inventory `UniversalSpatialPipeline` (report only)
**Needs:** P108 · **Files:** create `docs/audits/2026-11-universal-pipeline-inventory.md`
For each of its seven stages: what it computes, which existing stage already does that, verdict `FOLD` (name target stage and card) or `DELETE`.

### P113 [N] Remove `UniversalSpatialPipeline`
**Needs:** P112 with every row `DELETE` or already folded · **Files:** delete `engine/pipeline/universal_spatial_pipeline.py`, `tests/test_universal_spatial_pipeline_rules11_12.py`; edit `engine/pipeline/__init__.py`
**Check:** `grep -rn "universal_spatial_pipeline\|UniversalSpatialPipeline" --include=*.py --include=*.ts --include=*.tsx --include=*.yml .` → only the three files above.
- [ ] Remove the import block and the `__all__` entries in `engine/pipeline/__init__.py`; delete the two files.
**Run:** `python -c "import engine.pipeline"` and `REG`. **Stop if:** the Check shows any other reference.

### P114 [N] Frontend client for job stages
**Needs:** P105 · **Files:** create `frontend/src/lib/api/job-stages.ts`, `frontend/tests/contract/job-stages.test.mts`
- [ ] `export type StageStatus = "succeeded" | "degraded" | "skipped" | "failed" | "unknown";`
- [ ] `export interface JobStage { name: string; status: StageStatus; attempt: number; durationS: number | null; warnings: string[]; error: string | null; }`
- [ ] `export function parseJobStages(body: unknown): JobStage[]` — throws `Error("job stages: unexpected response shape")` when `body.items` is not an array; maps unknown status strings to `"unknown"`.
- [ ] `export async function fetchJobStages(jobId: string): Promise<JobStage[]>` using the same client helper `frontend/src/lib/api/reconstructions.ts` uses; a non-2xx response throws.
**Test:** valid body → 2 stages; `{}` → throws; status `"weird"` → `"unknown"`; `{items: []}` → `[]`. **Run:** `FE`.

---

# Phase P2 — Real data and an honest evaluator

### E201 [S] Evaluator: optimal matching, missing and extra counted
**Needs:** — · **Files:** `perception/architecture/accuracy_report.py`, `tests/test_measured_room_accuracy_report.py`
- [ ] Replace `_match_points_by_distance` with one-to-one optimal assignment (`scipy.optimize.linear_sum_assignment` on the distance matrix; pairs beyond `max_dist` are rejected after assignment). Keep the gates at 1.5 m and 1.2 m — do not tune.
- [ ] Add `@dataclass ClassScore: element_type: str; n_ground_truth: int; n_detected: int; n_matched: int; precision: float; recall: float; f1: float` (0/0 is defined as 0.0).
- [ ] `RoomAccuracyReport` gains `class_scores: list[ClassScore]`, `missing: list[dict]` (ground-truth elements with no match: `{"element_type","gt_id"}`), `extra: list[dict]` (detections with no match: `{"element_type","element_id"}`).
- [ ] `meets_tolerances` is `False` when `missing` contains any `wall`, `door`, `window` or `opening`, or when no room entity was found, regardless of dimensional checks. A matched element whose measured value is 0 or absent is a **failed** check, not a skipped one.
- [ ] `to_dict()` and the text report include the three new fields.
**Test:** perfect prediction → all recall 1.0, gate passes; one wall removed → wall recall 0.75 (of 4), `missing` length 1, gate fails; extra wall added → wall precision 0.8, `extra` length 1; no room entity → gate fails.

### E202 [N] Adversarial evaluator fixtures
**Needs:** E201 · **Files:** create `tests/test_accuracy_report_adversarial.py`
**Check:** read the fixture helpers at the top of `tests/test_measured_room_accuracy_report.py`; build worlds the same way.
One test each, asserting on `class_scores`, `missing`, `extra`, `meets_tolerances`: missing opening; opening with wrong width (+10 cm → check fails, recall stays 1.0); duplicate wall entity (precision drops, `extra` length 1); door labelled as window (door recall 0, window precision < 1); whole room shifted 2 m (all walls missing **and** all extra); empty world (every recall 0.0, gate fails); ground truth with zero openings and zero detected (opening `n_ground_truth == 0`, `missing` has no openings, gate decided by walls only).

### E203 [S] Evaluate every room, not the first
**Needs:** E201 · **Files:** `perception/architecture/accuracy_report.py`, tests
`evaluate_world_accuracy(world, ground_truth) -> WorldAccuracyReport` matching rooms by ground-truth id when present, else by centroid assignment; an unmatched ground-truth room is a missing room and fails the gate. `evaluate_measured_room_accuracy` stays as the single-room entry point.

### E204 [N] The accuracy CLI never defaults to synthetic silently
**Needs:** E201 · **Files:** `scripts/measured_room_accuracy_report.py`, `tests/test_measured_room_accuracy_report.py`
**Check:** `grep -n "add_argument" scripts/measured_room_accuracy_report.py` → shows `--fixture` with default `synthetic`.
- [ ] Remove the default: the CLI requires either `--fixture synthetic` or both `--world` and `--ground-truth`; otherwise exit code 2 with message `choose --fixture synthetic or provide --world and --ground-truth`.
- [ ] With `--fixture synthetic`, print as the first output line: `SYNTHETIC FIXTURE — regression check only, not real-world accuracy evidence`, and write `"evidence_class": "synthetic"` in the JSON output; real inputs write `"evidence_class": "measured"`.
**Test:** no arguments → exit 2; synthetic → first line matches and JSON has the key; update any call site if `grep -rn measured_room_accuracy_report .github scripts tests` shows one relying on the default (in scope for this card).

### E205 [N] Ground-truth schema
**Needs:** — · **Files:** create `benchmarks/measured_rooms/ground_truth.schema.json`, `tests/test_ground_truth_schema.py`
**Check:** open `synthetic/fidelity_room/ground_truth.json` and list its keys; the schema must accept that file unchanged.
- [ ] JSON Schema (draft 2020-12): required `room{id,name,area_m2,ceiling_height_m}`, `walls[]{id,start[2],end[2],length_m}`, `openings[]{id,kind ∈ door|window|opening,center[3],width_m,height_m,sill_height_m?,host_wall_id}`, `fixtures[]?`, and a new required block `capture{room_id,date,device_model,measured_by,method ∈ laser|tape|both,tolerance_m,consent_reference,lighting,notes}`. For the synthetic file, `capture` is not required when top-level `"synthetic": true` is present. Where the synthetic file's real key names differ from the names above, the synthetic file wins — use its names.
**Test:** synthetic file validates; a copy without `walls` fails; a copy with `kind: "portal"` fails; a measured file without `capture.consent_reference` fails.

### E206 [S] Uncertainty from residuals (after E210)
Replace the constant uncertainty strings (`"+-0.015 m"`) with per-dimension values derived from the measured-room residual distribution; until E210 exists, label them `"uncalibrated"` instead of a number.

### E207 [N] Dataset protocol document
**Needs:** E205 · **Files:** create `benchmarks/measured_rooms/PROTOCOL.md`
Write these sections with exactly this content: **Rooms** — six, each ticking at least two of: plain painted walls, closed window, glass door or mirror, non-rectangular plan, sloped or stepped ceiling, heavy furniture occlusion, sockets and switches on two walls, low light. **Consent** — written permission from the occupier, stored outside the repo, referenced by id; no people, documents or screens in frame. **Capture** — one ordinary phone, default camera app, 40–80 photos per room, walk the perimeter facing the centre then the centre facing each wall, include every corner floor-to-ceiling; second independent capture of two rooms on a different day. **Ground truth** — laser distance meter: every wall length, ceiling height at two points, each opening width/height/sill, each socket and switch height above floor and offset from nearest corner; measured twice by two people where possible, record both. **Files** — `benchmarks/measured_rooms/<room_id>/{photos/, ground_truth.json, capture.json, README.md}`; photos stored with Git LFS or outside the repo with checksums listed in `capture.json`. **Splits** — rooms 1–4 development, rooms 5–6 blind hold-out; hold-out ground truth is never opened by anyone changing reconstruction code. **Retention** — delete on occupier request within 7 days.

### E208 [H] Capture and measure six rooms per E207.

### E209 [N] Runner script: photos in, scored report out, through the real API
**Needs:** E201, E204, E205 · **Files:** create `scripts/run_measured_room.py`, `tests/test_run_measured_room_cli.py`
**Check:** read `tests/test_golden_path_api_to_glb.py` for the exact sequence of API calls (create session → upload → reconstruct → poll job → fetch world).
- [ ] CLI: `--room-dir <path> --api <base url> --token <t> --out <dir>`.
- [ ] Refuse to start (exit 3, message `REALITY_TEST_BACKEND is set; this script measures the real engine only`) if that env var is non-empty.
- [ ] Validate `ground_truth.json` against the E205 schema; replay the golden-path call sequence against `--api` with `httpx`; poll the job until a terminal status; on `failed` or `cancelled` write the job error and exit 4 — never score a failed job.
- [ ] Write `<out>/<room_id>/<git short sha>.json` containing: git sha, job id, job status, stage list (P105), evaluator `to_dict()`, wall-clock seconds, sha256 of every input photo.
**Test:** env var set → exit 3; missing `ground_truth.json` → exit 2; (API interaction is covered by E210, not unit-tested here).

### E210 [H] Run E209 on all six rooms; commit results; publish the baseline table in `docs/STATUS.md`.
### E211 [H] Open the IFC export of two rooms in Bonsai (BlenderBIM) and one commercial BIM tool; record version, warnings, screenshots under `benchmarks/results/interop/`.
### E212 [H] Open the DXF/PDF plan of the same rooms in two CAD tools; check units, scale, layers, dimension text.

---

# Phase P3 — Geometry upgrade

### G301 [S] Spike: feed-forward models on the dev GPU (report only)
**Needs:** owner approval of D6 · **Files:** `docs/audits/2026-11-feedforward-spike.md`
Measure MapAnything (Apache checkpoint) and Depth Anything 3 on 8 / 16 / 32 photos of a development room: peak VRAM, seconds, registered views, scale error against ground truth. State which fits 6 GB and at what view count. Recommend one.

### G302 [S] `FeedForwardBackend`
**Needs:** G301 · **Files:** create `reconstruction/backend/feedforward_backend.py`, tests
Implements the interface in `reconstruction/backend/interface.py` and returns the same `ReconstructionResult` shape COLMAP does (cameras, poses, points with tracks where available, scale state `metric` with provenance `ESTIMATED`). Unavailable weights or insufficient VRAM → explicit unavailable result, never an empty success.

### G303 [N] Register the model with licence and checksum
**Needs:** G302, T009 · **Files:** `perception/model_registry.py`, `tests/test_model_licence_gate.py`
**Check:** read one existing registration (e.g. `sam2_hiera_tiny`) and copy its argument style.
- [ ] Add the entry chosen in G301 with `license="Apache-2.0"`, its weights URL, sha256, framework, input and output contract strings.
**Test:** `assert_production_licences([<new name>])` passes; entry has a non-empty checksum.

### G304 [S] Arbiter considers both backends
**Needs:** G302 · **Files:** `engine/pipeline/candidate_selection.py`, `engine/pipeline/stages/s04_pose.py`, tests
`pose` stage produces up to two candidates; `WorldArbiter` selects on registered-view fraction, reprojection/consistency residual and scale confidence. The losing candidate's metrics stay in the stage result. Selection is deterministic for fixed inputs.

### G305 [N] Backend comparison table
**Needs:** G304, E210 · **Files:** create `scripts/compare_backends.py`, output `benchmarks/results/backend_comparison.md`
- [ ] For each development room run E209 twice with pipeline param `pose_backend` = `colmap` then `feedforward`; tabulate registered views, wall recall, opening recall, median wall-length error, seconds. Hold-out rooms are excluded (rooms 5–6 per `PROTOCOL.md`).

### G306 [S] Closed windows: visual evidence, not only gaps
**Needs:** P108 · **Files:** `perception/architecture/windows.py`, `openings.py`, `engine/pipeline/stages/s09_perception.py`, tests
Add a second hypothesis source: open-vocabulary detection (`grounding_dino_tiny` or `owlv2_base_patch16`, both already registered) + SAM 2 mask, projected onto the host wall plane, fused with the gap-based detector across views. A window supported by visual evidence but no geometric gap is kept with provenance `OBSERVED` and a `closed` state; one supported by a gap only stays as today.

### G307 [N] Window/door detection score on labelled images
**Needs:** G306, labelled set from [H] · **Files:** create `scripts/eval_openings_2d.py`, `benchmarks/windows/README.md`
- [ ] Input: folder of images with one JSON per image `{"boxes":[{"label":"window|door","xyxy":[...]}]}`. Output: per-label precision, recall, F1 at IoU 0.5, plus list of missed boxes per image.

### G308 [S] Occlusion-aware walls
**Needs:** W402 · **Files:** `perception/architecture/wall_slabs.py`, tests
Wall spans with no supporting points because furniture blocks every view are emitted with `ObservationCoverage.OCCLUDED` and provenance `INFERRED`, not as solid observed wall and not as an opening.

### G309 [S] Monocular plane and normal priors for blank walls
**Needs:** G301 · Use MoGe-2 or Metric3D v2 normals as an additional plane hypothesis source where SfM points are sparse. Provenance `ESTIMATED`.

### G310 [N] Re-run the benchmark and diff
**Needs:** G304–G309, E210 · **Files:** `benchmarks/results/measured_rooms/`, `docs/status.yaml`
- [ ] Run E209 on development rooms at the new commit; produce a before/after table per room and class; the owner runs the hold-out.

---

# Phase P4 — Persistent World

### W401 [S] One evidence-class vocabulary
**Needs:** — · Proposal first (`docs/engineering/` note), then code. Map `StatementState.OBSERVED/INFERRED` onto `Provenance`; keep `StatementState` importable as a deprecated alias; stored worlds load unchanged (add a round-trip test over every fixture world in `tests/`).

### W402 [S] `ObservationCoverage`
**Needs:** W401 · **Files:** `world_ir/observations.py`, serialization, tests
`class ObservationCoverage(str, Enum)`: `OBSERVED_PRESENT`, `OBSERVED_ABSENT`, `OCCLUDED`, `NOT_IN_VIEW`, `INSUFFICIENT_QUALITY`, `NOT_OBSERVED`. Additive; `Visibility` untouched. Absent field deserialises as `NOT_OBSERVED`.

### W403 [S] Coverage from camera frusta
**Needs:** W402 · **Files:** create `world_ir/coverage.py`
`def coverage_for_bounds(bounds_min, bounds_max, cameras: Sequence[CameraView], *, max_range_m: float, min_pixels: int, occluders: PointSet | None) -> ObservationCoverage`.

### W404 [N] Tests for `coverage_for_bounds`
**Needs:** W403 · **Files:** create `tests/test_world_coverage.py`
Build cameras with the helper W403's own tests use. Cases: box centred 2 m in front of one camera → `OBSERVED_PRESENT`; box behind the camera → `NOT_IN_VIEW`; box 2 m ahead but 60° off-axis with a 60° FOV → `NOT_IN_VIEW`; box beyond `max_range_m` → `INSUFFICIENT_QUALITY`; box projecting to fewer than `min_pixels` → `INSUFFICIENT_QUALITY`; box in view with an occluder plane between → `OCCLUDED`; empty camera list → `NOT_OBSERVED`.

### W405 [S] Adoption outcome `not_observed`
**Needs:** W403 · **Files:** `worldstore/proposals.py`, `apps/api/routes_adoptions.py`, tests
A head entity absent from the candidate becomes `removed` (needs review) only if its coverage in the new session is `OBSERVED_ABSENT`; otherwise `not_observed`, which is never applied as a deletion. Response gains a `not_observed` list (additive).

### W406 [N] Adoption behaviour matrix
**Needs:** W405 · **Files:** create `tests/test_adoption_matrix.py`
**Check:** read `tests/test_world_adoption_api.py` and reuse its fixtures and client.
One test per case, asserting on the proposal body and on the committed version: unchanged entity keeps its id; manual edit on head survives adoption of a session that did not touch it; moved furniture → one `modified`, zero `added`+`removed` pairs; new object → `added`; removed with the area in view → `removed`, `needs_review`; same removal with the area out of view → `not_observed`, entity still present after commit; candidate from a `partial` job → proposal carries a `partial_source` flag and is not auto-committable; reject leaves head unchanged.

### W407 [S] Stale-head protection on commit
Verify `POST .../adoptions/{id}/commit` compares the proposal's base version with World head inside the same transaction that advances head; two concurrent commits → exactly one 200 and one 409. Add the test first; fix only if it fails.

### W408 [S] Idempotent commit
`Idempotency-Key` header on adoption commit and on reconstruction submit; a repeat returns the first response and creates nothing.

### W409 [N] Review panel groups
**Needs:** W405, U504 · **Files:** `frontend/src/lib/api/adoptions.ts`, the component under `frontend/src/components/adoptions/`, a contract test
- [ ] Parse the additive `not_observed` list; render groups in this order with these labels: `New`, `Changed`, `Removed — needs review`, `Not seen this visit (kept)`, `Conflicts`. The "Not seen" group has no accept/reject controls.
**Run:** `FE`.

### W410 [H] Three-room apartment, then revisit one room a different day. Record: World count, entity count before/after, edits preserved, review time.

---

# Phase PU — Clean UI (see `36_UI_SPEC.md`)

### U501 [N] Route inventory (report only)
**Needs:** — · **Files:** create `docs/audits/2026-11-route-inventory.md`
- [ ] For each of the 40 `page.tsx` files (`find frontend/src/app -name page.tsx`): route, top-level component rendered, number of inbound links (`grep -rn "href=\"<route>" frontend/src | wc -l`, also check `router.push`), API proxies it calls, and the target from the mapping table in `36_UI_SPEC.md` section 2.
**Done when:** 40 rows. **Stop if:** a route in the spec's mapping does not exist — record it, continue.

### U502 [N] Redirects for merged routes
**Needs:** U501 · **Files:** `frontend/next.config.*`, create `frontend/tests/contract/redirects.test.mts`
- [ ] Add a `redirects()` entry (`permanent: false`) for every row in spec section 2 marked "redirect". Do not delete any page file.
**Test:** import the config, call `redirects()`, assert each source maps to the spec's destination and that no source equals a destination (no loops).

### U503 [N] Five-item navigation
**Needs:** U502 · **Files:** `frontend/src/lib/shell/shell-logic.ts` (the nav model that feeds `frontend/src/components/shell/Sidebar.tsx`), its contract test
**Check:** `grep -n "href" frontend/src/lib/shell/shell-logic.ts` → shows the nav entries. If they live elsewhere, stop and report.
- [ ] Primary entries exactly: `Home /`, `Worlds /worlds`, `Capture /capture`, `Deliveries /deliveries`; secondary: `Settings /settings`. Remove others from the model (their pages remain reachable by URL/redirect).
**Test:** model has 4 primary + 1 secondary; every `href` is unique.

### U504 [N] `StudioLayout` — four zones, no behaviour change
**Needs:** — · **Files:** create `frontend/src/components/studio/StudioLayout.tsx`
- [ ] Props: `top`, `left`, `viewport`, `right`, `bottom`, `drawer` (all `ReactNode`), `leftOpen`, `rightOpen`, `bottomOpen`, `drawerOpen` (booleans).
- [ ] CSS grid: rows `auto 1fr auto`; columns `280px 1fr 340px`; closed side columns collapse to `0`; the viewport cell never unmounts; `drawer` is an overlay sheet anchored right, 420 px, above `right`.
- [ ] Landmarks: `<header>`, `<nav aria-label="World tree">`, `<main>`, `<aside aria-label="Inspector">`, `<footer aria-label="Timeline">`.
**Run:** `FE`.

### U505 [S] Move `SpatialWorkstation` into the layout
**Needs:** U504, U506 · Split the 976-line component: state to the store, each zone's content to its own file, `SpatialWorkstation` becomes composition only. No feature removed; Playwright e2e must pass unchanged.

### U506 [N] Studio store
**Needs:** — · **Files:** create `frontend/src/lib/studio/store.ts`, `frontend/tests/contract/studio-store.test.mts`
**Check:** `grep -n "zustand" frontend/package.json` → present.
- [ ] State: `selectedEntityId: string | null`, `hoveredEntityId: string | null`, `mode: "inspect" | "edit" | "review"`, `representation: "points" | "mesh" | "splat" | "plan"`, `panels: {left: boolean; right: boolean; bottom: boolean}`, `activeTool: string | null` (Analyze drawer; `null` = closed), `inspectorTab: "properties" | "evidence" | "measurements" | "issues"`.
- [ ] Actions: `select(id)`, `hover(id)`, `setMode(m)`, `setRepresentation(r)`, `togglePanel(k)`, `openTool(id)`, `closeTool()`, `setInspectorTab(t)`.
- [ ] Rules: `select(id)` with a non-null id sets `panels.right = true`; `openTool` closes any other tool (one at a time); `setMode("review")` sets `panels.bottom = true`.
**Test:** one assertion per rule; initial state is `mode "inspect"`, `representation "mesh"`, `panels {left: true, right: true, bottom: false}`.

### U507 [N] Inspector tabs
**Needs:** U506 · **Files:** `frontend/src/components/inspector/AdaptiveInspector.tsx` and one new `InspectorTabs.tsx`
- [ ] Four tabs bound to `inspectorTab`: Properties, Evidence, Measurements, Issues. Existing inspector sections are moved under the matching tab, not rewritten. Tabs use `role="tablist"/"tab"/"tabpanel"`, arrow-key navigation, and show a count badge on Issues.
- [ ] Empty selection renders: `Select something in the model or the tree.`

### U508 [N] One selection everywhere
**Needs:** U505, U506 · **Files:** `frontend/src/components/workspace/SceneTree.tsx`, `QueryPanel.tsx`, the viewport selection handler in `frontend/src/components/viewport/`
- [ ] Each reads `selectedEntityId` from the store and calls `select(id)`; remove any local selected-id state and prop-drilled callbacks for selection.
**Test (Playwright, add to the existing studio spec):** click a tree row → inspector title equals that entity's label and the row has `aria-selected="true"`.

### U509 [N] Undo/redo end-to-end check
**Needs:** — · **Files:** one new Playwright spec under `frontend/e2e/`
**Check:** read `frontend/e2e/support.ts` and an existing spec for login/seed helpers.
- [ ] Move a furniture entity → press Ctrl+Z → position equals the original → reload the page → position still original → Ctrl+Shift+Z → position equals the moved value.
**Stop if:** the app has no keyboard binding for undo — report; do not add one here.

### U510 [N] Evidence badge and legend
**Needs:** — · **Files:** create `frontend/src/components/ui/EvidenceBadge.tsx`, `EvidenceLegend.tsx`, contract test for the mapping
- [ ] `export const EVIDENCE_STYLE: Record<"OBSERVED"|"RECONSTRUCTED"|"ESTIMATED"|"INFERRED"|"GENERATED"|"UNKNOWN"|"CONFLICT", {label: string; token: string; pattern: "solid"|"dashed"|"dotted"|"hatched"}>` with the values in spec section 5.
- [ ] Badge shows label + pattern swatch; unknown input renders `UNKNOWN`, never `OBSERVED`.
**Test:** all seven keys present; `styleFor("nonsense").label === "Unknown"`.

### U511 [N] One representation switch
**Needs:** U506 · **Files:** create `frontend/src/components/studio/RepresentationSwitch.tsx`; edit the toolbar file that currently holds the points/mesh/splat toggles (find with `grep -rn "Toggle Points" frontend/src`)
- [ ] Segmented control with four options bound to `representation`; an option whose data is absent is disabled with the tooltip `Not available for this version`. Remove the separate layer toggles it replaces from the toolbar (keep their command-palette entries).

### U512 [N] Analyze drawer
**Needs:** U504, U506 · **Files:** create `frontend/src/components/studio/AnalyzeDrawer.tsx`, `frontend/src/lib/studio/tools.ts`
- [ ] `tools.ts`: array of `{id, label, group: "Drawings"|"Quantities"|"Checks"|"Services"|"Change", load: () => Promise<{default: ComponentType}>}` for: Sections & elevations, Takeoff schedule, Clash detection, Life safety & accessibility, Thermal envelope, Surface quality, MEP topology, Temporal diff, Construction progress.
- [ ] Drawer lists tools by group; choosing one lazy-loads it (`next/dynamic`, `ssr: false`) into the drawer body. Every tool header shows a `Preliminary` badge with the tooltip `Automated analysis from the current model. Not a professional certification.`
- [ ] Replace the individual toolbar/menu launchers for these tools with a single `Analyze` button.

### U513 [N] Specialist modals render inside the drawer
**Needs:** U512 · **Files:** the nine tool components named in U512, one per commit
- [ ] Remove each component's own full-screen modal chrome (backdrop, fixed positioning, close button) and accept `onClose` from the drawer. No logic changes.

### U514 [N] Photo quality reasons on upload cards
**Needs:** — · **Files:** `frontend/src/components/ui/EvidenceCard.tsx`, `frontend/src/lib/api/evidence-quality-model.ts` (+ its contract test)
**Check:** read `evidence-quality-model.ts` for the reason codes the backend already returns.
- [ ] Map each code to one short instruction (`blur` → `Blurry — hold still and retake`, `underexposed` → `Too dark — add light`, `overexposed` → `Too bright — avoid facing windows`, unknown code → show the raw code). Show the first reason on the card with a warning style; never hide a card for quality.

### U515 [N] "What to capture next" panel
**Needs:** — · **Files:** `frontend/src/components/studio/CapturePlanPanel.tsx`, `frontend/src/lib/api/capture-coverage-model.ts` (+ test)
- [ ] Render the existing capture-coverage and capture-request data as at most five imperative lines (`Photograph the north wall from the doorway`); when the API returns none, show `No gaps detected from the photos so far` — and when the API errors, show the error state, not that sentence.

### U516 [N] Timeline strip
**Needs:** P114, U504 · **Files:** create `frontend/src/components/studio/TimelineStrip.tsx`
- [ ] Horizontal list of sessions, jobs and versions for the World, newest right. A job chip shows `toJobStatus` text and, expanded, the stage list from `fetchJobStages` with duration and warnings. No percentage bar anywhere.

### U517 [N] One state block
**Needs:** — · **Files:** create `frontend/src/components/ui/StateBlock.tsx`; adopt in `HomeDashboard.tsx`, the Worlds list page and the Sessions list page
- [ ] `kind: "loading" | "empty" | "error" | "partial"`, `title`, `detail?`, `action?: {label, onClick}`. `error` uses `role="alert"`; `loading` uses `aria-busy`. Replace each page's ad-hoc versions. An error response must render `error`, never `empty`.

### U518 [N] Accessibility smoke in e2e
**Needs:** — · **Files:** one Playwright spec; `frontend/package.json` devDependency `@axe-core/playwright` (the only new dependency this plan allows in the frontend)
- [ ] Run axe on `/`, `/capture`, `/worlds/<seeded id>`; fail on `serious` or `critical` violations; write the rest to the test log.

### U519 [N] Command palette matches the new IA
**Needs:** U503, U511, U512 · **Files:** `frontend/src/components/shell/CommandPalette.tsx`
- [ ] Remove entries that point to redirected routes; add `Open Analyze`, `Switch to Points/Mesh/Splat/Plan`, `Toggle Tree`, `Toggle Inspector`, `Toggle Timeline`. Labels use sentence case and no bracketed shortcuts in the label text (render the shortcut in its own `<kbd>`).

### U520 [N] Delete redirected pages
**Needs:** U502 merged and soaked; owner approval · **Files:** only the page files U501 lists with zero inbound links
- [ ] Delete the page file; keep the redirect. `FE` and the Playwright suite must pass.

---

# Phase P5 — Capture

### C601 [S] Capture manifest
`evidence/capture_manifest.py` (Pydantic): device model and OS; per-frame `{evidence_id, timestamp_s, time_base ∈ device_monotonic|exif_wallclock, time_uncertainty_s, intrinsics?, pose_world_from_camera?, tracking_state?, depth_ref?}`; optional streams `imu`, `gnss` with their own time base; `missing_sensors: list[str]` (absence is recorded, never inferred); units and axis convention stated in the schema docstring.

### C602 [N] Importer for an open ARKit capture folder
**Needs:** C601 · **Files:** create `evidence/arkit_capture.py`, `tests/test_arkit_capture_import.py`
**Check:** confirm the layout against the capture app's README before coding (expected: `rgb.mp4`, `odometry.csv` with columns `timestamp, frame, x, y, z, qx, qy, qz, qw`, `camera_matrix.csv` 3×3, `imu.csv`, `depth/*.png` 16-bit millimetres, `confidence/*.png`). If it differs, stop and report the real layout.
- [ ] `def parse_arkit_capture(folder: Path) -> CaptureManifest` — reads the CSVs with the standard library `csv`, records `missing_sensors` for each absent file, raises `ValueError` naming the file on a malformed row. Does not decode video.
**Test:** build a tiny synthetic folder in `tmp_path` (3 odometry rows, a 3×3 matrix, no IMU file): 3 frames with poses; `"imu" in manifest.missing_sensors`; a row with 8 columns raises `ValueError` mentioning `odometry.csv`.

### C603 [N] Timestamp association
**Needs:** C601 · **Files:** create `evidence/time_association.py`, tests
- [ ] `def nearest_sample(t: float, times: Sequence[float], max_dt: float) -> int | None` (binary search; `None` when the nearest is further than `max_dt`); `def interpolate_pose(t, t0, pose0, t1, pose1)` (linear translation, slerp rotation).
**Test:** exact hit; between two samples picks the nearer; outside `max_dt` → `None`; empty list → `None`; interpolation at `t0`, `t1` and midpoint.

### C604 [S] Pose priors into the `pose` stage — feed tracked poses to COLMAP as priors; registered-view count must be ≥ the no-prior run on the same set.
### C605 [S] Metric scale from device tracking — new scale source `device_tracking`, provenance `OBSERVED`, with its own uncertainty.
### C606 [H] Physical indoor loop and a staircase: measure start/end drift and floor height against a tape.

---

# Phase P6 — SaaS operations

### S701 [N] Tenant-scope inventory with a failing-by-default test
**Needs:** — · **Files:** create `scripts/route_tenant_inventory.py`, `tests/test_route_tenant_inventory.py`, `docs/audits/2026-11-route-tenancy.md`
**Check:** read `apps/api/tenancy.py` and one handler in `routes_worlds.py` to learn the name of the tenant/role dependency.
- [ ] Walk `app.routes` from `apps.api.main`; for each route record path, methods, and whether its dependency tree includes the tenancy dependency.
- [ ] `tests/...`: every route is either tenant-scoped or listed in `PUBLIC_ALLOWLIST` (health, auth/login, signup, share-link viewer, Stripe webhook). Seed the allow-list only with those; every other unscoped route makes the test fail and is listed in the audit doc.
**Stop if:** more than 10 routes fail — commit the test as `xfail(strict=True)` with the list, and report; fixing is S704.

### S702 [N] Cross-tenant tests: worlds and sessions
**Needs:** S701 · **Files:** create `tests/test_cross_tenant_worlds_sessions.py`
**Check:** reuse the two-tenant fixture in `tests/test_security_dast.py`.
- [ ] Tenant B requests every `GET/POST/PATCH/DELETE` under `/api/worlds/{A's id}` and `/api/sessions/{A's id}` that S701 lists. Assert status is 403 or 404 and that the body contains none of A's ids or names.

### S703 [N] Cross-tenant tests: jobs, artifacts, exports, share links — same pattern as S702; plus a revoked share token returns 404/410 and an artifact sha from tenant A is not served to tenant B.
### S704 [S] Fix every finding from S701–S703, one commit per router.
### S705 [S] Claiming on Postgres — verify `claim_next_job` uses row locking that prevents double claim; add a test with two concurrent claimers against Postgres (CI service container); fix if it fails.

### S706 [N] Duplicate delivery does not double-commit
**Needs:** P106 · **Files:** create `tests/test_job_duplicate_delivery.py`
- [ ] Process the same reconstruct job twice (reset its status to `queued` after the first success). Assert the World has exactly one new version and `job_stage_runs` shows checkpoint reuse for every stage on the second pass.

### S707 [S] Resource classes — `jobs.resource_class` column; worker flag `--classes cpu,gpu`; a GPU-class job is never claimed by a CPU-only worker.
### S708 [S] Per-tenant concurrency cap — claim skips tenants already at their limit; test fairness with two tenants.

### S709 [N] Upload limit tests
**Needs:** — · **Files:** create `tests/test_upload_limits.py`
**Check:** `grep -rn "MAX_\|max_pixels\|max_bytes" apps/api evidence | head -20` → note the existing limit names.
- [ ] For each existing limit: one request just under passes, one just over is refused with 413 or 422 and a message naming the limit. If bytes, pixels or frame count has no constant, list it in the report instead of inventing one.

### S710 [H] Stripe test-mode walkthrough with real test keys: checkout, portal, webhook retry, cancel; record ids.

### S711 [N] Presence limit made explicit
**Needs:** — · **Files:** `engine/pipeline/collab_presence.py`, `apps/api/routes_collab.py`, test
- [ ] Add to the presence API response `"scope": "single_process"`. At API start-up log one warning if env `WEB_CONCURRENCY` > 1: `collaboration presence is per-process; peers on other workers will not see each other`.
**Test:** response contains the key; with the env var patched to `2`, the warning is emitted once.

### S712 [H] Protect `main` (require CI + E2E), run one backup/restore drill, write down the steps.

---

# Phase P7 — Expansion (not carded; each needs a named customer workflow first)

Capture director and route planning · fine-grained components and materials · parametric / constraint-aware editing · linked assets · LLM planner over the deterministic copilot tools · photo-based Gaussian-splat training (gsplat) and LOD streaming (3D Tiles) · portals through openings · automatic site registration · specialist hardening (exact-geometry clash, regional price data, jurisdiction profiles) · outdoor / drone / RTK.
