# 12-Day Execution Plan — Day 1 Baseline Audit

Baseline inspected: `main` @ `8975eb2e` ("merge: resolve conflicts with origin/main
(reconstruction sprint10)"), working tree clean apart from this change set.

This audit is the record of what was actually read and executed, not of what the
plan hoped to find. Every claim below names the file and the line-level fact it
rests on. Where something could not be verified, it says so.

---

## 1. Method

| Step | What was done |
|---|---|
| Repository state | `git log --oneline -20`, `git status --porcelain`, branch list |
| Inventory | directory listings of `perception/`, `apps/`, `docs/`, `tests/` (734 test files) |
| Contract reading | `perception/engine_evidence.py`, `perception/engine_orchestrator.py`, `perception/multi_engine_stage.py`, `perception/engine_adapters.py`, `perception/{detection,segmentation,depth}_adapter.py` |
| Production consumer | `apps/api/evidence_world_stage.py` (788 lines), `engine/multiengine/evidence_matrix.py`, `engine/multiengine/worldir_bridge.py`, `engine/multiengine/observation.py` |
| Baseline test runs | `tests/test_multiengine_foundation.py`, `test_multi_engine_adapters.py`, `test_multi_engine_comparator.py`, `test_multiengine_evidence_flow.py`, `test_multiengine_worldir_bridge.py` → **68 passed** |
| Runtime probe | `torch 2.14.0+cu130` importable; `perception.engine_adapters._get_vram_peak()` returns `0.0`; `list_engines()` returns `[]` before registration |

Not available in this environment and therefore not verified here: a CUDA device
was not exercised (engine gating was verified by patching the CUDA probe in
tests), real model weights for every adapter, and any CI run history (the
repository metadata for prior CI runs was not reachable from this host).

---

## 2. System architecture as found

```
upload ─► apps/api/jobs.py ─► stage runner
                              └─ evidence_world_stage.run_for_job()        [opt-in: RE_EVIDENCE_WORLD=1]
                                   ├─ run_evidence_flow(model)  ── engine/multiengine/evidence_flow.py
                                   │    ├─ per-engine adapters ─► EvidenceMatrix (append-only, content-addressed)
                                   │    ├─ association / conflicts / canonicals
                                   │    └─ BridgeResult  (canonical entities, geometry, materials)
                                   ├─ persist_world(bridge) ─► WorldIR ─► validation ─► WorldStore version ─► reload compare
                                   └─ run_multi_engine_for_job()  ── perception/multi_engine_stage.py
                                        └─ MultiEngineOrchestrator ─► UnifiedEngineOutput (SIFT, LightGlue-ALIKED,
                                           Grounding DINO, SAM 2, Depth Anything V2)
```

There are **two parallel evidence representations**:

| Representation | Module | Store | Consumers before this change |
|---|---|---|---|
| `EngineObservation` + `EvidenceContext` + `RunRecord` | `engine/multiengine/observation.py`, `evidence_matrix.py` | `EvidenceMatrix` (`observations.jsonl`) | WorldIR bridge, association, canonical entities, job report |
| `UnifiedEngineOutput` + `ObservationRecord` | `perception/engine_evidence.py` | none (list in memory) | job report only (`facts["multi_engine"]`) |

That split was the central producer-consumer gap: the second representation had no
persistence and no path into the world.

---

## 3. Findings

### F1 — P0: multi-engine observations never reached WorldIR or the Evidence Matrix

*Evidence:* `apps/api/evidence_world_stage.py` computed
`facts["world_ir"] = persist_world(...)` and only afterwards called
`run_multi_engine_for_job(...)`, storing the result at `facts["multi_engine"]`.
`perception/engine_evidence.py` has no persistence layer, and nothing in
`engine/multiengine/worldir_bridge.py` read `UnifiedEngineOutput`.

*Consequence:* the plan's Day 3 acceptance criterion ("actual engine observations
must enter WorldIR before canonical persistence") was unmet: a persisted world
carried no trace of which perception engines had run, and a fresh process could
not recover their lineage.

*Status:* **fixed in this change** (see §4).

### F2 — P0: the orchestrator executed engines whose availability probe failed

*Evidence:* `MultiEngineOrchestrator.run_wave()` built its candidate list as
`available_specs = [_ENGINES[e] for e in _ENGINES]` — every registered engine —
while `run_condition` was consulted only in `list_available_engines()` and
`get_unavailable_engines()`, neither of which `run_wave` called. An engine with a
missing dependency (e.g. kornia absent) was therefore *invoked*, failed, and was
reported as `failed` rather than `unavailable`, and never appeared in the report's
`unavailable` list (`merge_engine_outputs` hard-coded `unavailable=[]`).

*Status:* **fixed**.

### F3 — P0: a CPU-only machine deferred every GPU-declaring engine forever

*Evidence:* `_compute_vram_budget()` returned `0.0` when `torch.cuda.is_available()`
is False, and `run_wave` skipped any engine with `spec.vram_mb > budget`. All
model adapters register `vram_mb >= 1024` and all of them already fall back to
`device = "cuda" if torch.cuda.is_available() else "cpu"`. On a CPU-only machine
every one of them was recorded `vram_deferred` and never ran.

The plan's Day 2 criterion "CPU-only execution does not incorrectly defer
CPU-capable engines" was violated.

*Status:* **fixed** — VRAM is only gated when CUDA is present.

### F4 — P1: deferred engines were never actually retried

*Evidence:* the `run_wave` docstring promised engines were "retried after VRAM is
freed by a completed wave"; no retry existed, and `_deferred` was appended to
across calls without being cleared in `run_wave` (only in `run_all`).

*Status:* **fixed** — one retry after the cache is released; what still does not
fit is reported `unavailable` with the measured budget.

### F5 — P1: dependency ordering was decorative

*Evidence:* `run_wave` ran a correct topological pass, then re-ordered the result
with an insertion loop whose dependency predicate (`all(d in completed_ids ...)`)
was always true, because `completed_ids` had already been filled by the first
pass. Dependencies were never consulted to decide whether a dependent *should*
run: SAM 2 executed with an empty `bboxes` mapping when Grounding DINO had failed.

*Status:* **fixed** — Kahn ordering by priority, and a dependent whose dependency
did not reach `succeeded`/`partial` is `skipped` with `dependency_unmet`.

### F6 — P1: no execution timeouts, cancellation or retry policy

*Evidence:* engine calls were plain `fn(image_paths, **context)` inside
`try/except`; `EngineSpec` had no timeout field, no cancel hook was consulted,
and a transient `OSError` was recorded as a permanent failure.

*Status:* **fixed** — `timeout_s`, `max_attempts`, `should_cancel`, with the
Python limitation (a running call cannot be interrupted, so it is *abandoned*)
recorded in the execution contract instead of hidden.

### F7 — P0: a fabricated confidence value

*Evidence:* `perception/engine_adapters.py`, LightGlue-ALIKED adapter:

```python
confidence=float(match_count / max(1, match_count)),   # always 1.0
```

Every LightGlue match observation claimed certainty. This directly violates the
plan's "do not fabricate confidence values".

*Status:* **fixed** — confidence is LightGlue's own mean match score, or `None`
with a recorded reason when the matcher returns no scores.

### F8 — P1: position contradictions compared relative units as metres

*Evidence:* `compare_observations()` used `dist > 0.1` and formatted the result as
`f"position_mismatch: {dist:.3f} m"` regardless of
`CoordinateFrame.scale_state`, including for `relative` depth-derived and
`sfm_gauge` positions. That is a fabricated metric claim produced by the fusion
layer itself.

*Status:* **fixed** — positions are compared only inside one frame id and one
length-carrying scale state; the message states the unit actually used; a
cross-scale pair produces a `scale_state_mismatch` note, which the merger keeps
out of the contradiction count.

### F9 — P1: label contradiction detection was unreachable

*Evidence:* `merge_engine_outputs` grouped observations by
`(source_image_id, normalised label)` and then called `compare_observations`,
whose first check is `obs_a.label != obs_b.label`. Observations that reached the
comparison always had equal labels, so `label_mismatch` could never be produced.
Conversely, two engines labelling *different objects* on one image with the same
word were compared as if they were one target.

*Status:* **fixed** — grouping is by source image, and a pair is compared only
when the two bounding boxes overlap (`SAME_TARGET_MIN_IOU = 0.5`), which is what
distinguishes "two engines, one object" from "two objects".

### F10 — P2: duplicate observations counted as extra evidence

*Evidence:* `compute_observation_hash()` existed and was imported by the
orchestrator but never called; identical repeats inflated `n_observations`.

*Status:* **fixed** — content-identical observations are counted once and
reported as `duplicate_observations`.

### F11 — P2: registry shared mutable state and caller context mutation

*Evidence:* module-level `_ENGINES` / `_ENGINE_FNS` with no lock; `run_wave`
mutated the caller's `context` dict (`context["bboxes"] = ...`), so concurrent
jobs sharing a context dict could contaminate one another.

*Status:* **fixed** — a registry lock, a per-call copy of the context, and an
orchestrator-level lock. Downstream facts are keyed by observation *category*, not
by the literal engine id `grounding_dino`, so a detector registered under another
name still feeds the segmenter.

### F12 — P3: `_get_vram_peak()` could never return a value

*Evidence:* `perception/engine_adapters.py` referenced `torch` inside
`_get_vram_peak()` without importing it in that scope; the resulting `NameError`
is an `Exception`, swallowed by the surrounding handler, so VRAM telemetry was
permanently `None`.

*Status:* **fixed** (verified: returns `0.0` on this machine rather than raising).

### F13 — open, not addressed in this change set

These are real gaps that remain; none is papered over.

| # | Gap | Where |
|---|---|---|
| G1 | No metric ground truth exists anywhere: `docs/RELEASE.md` records `total_verified: 0`, `overall_status: BLOCKED` for the six measured rooms. No accuracy claim can be made. | `RELEASE.md`, `benchmarks/measured_rooms/` |
| G2 | Camera registration quality has no measured baseline; `docs/STATUS.md` S5-06 is `IMPL-UNVERIFIED` against synthetic worlds only. | `docs/STATUS.md` |
| G3 | Confidence calibration is `LEGACY_UNSPECIFIED`; `apps/api/evidence_world_stage.py` `ENABLEMENT_CRITERIA` marks `confidence_calibration` unmet. | `evidence_world_stage.py` |
| G4 | Video ingestion, SAM masks, object tracking and small-object taxonomy exist but are not on the live API path (`RELEASE.md`, "What does not ship in 1.0"). | `RELEASE.md` |
| G5 | Structural opening geometry: `perception/world_bridge.py` builds `OpeningObservation` records, and host-wall association is a *candidate* (`host_wall_candidate`), not a resolved relationship. | `perception/world_bridge.py` |
| G6 | Multi-view depth fusion / TSDF / Poisson surface reconstruction are not wired into the evidence flow as a fused surface source. | `engine/multiengine/`, `reconstruction/` |
| G7 | The 3D viewport's object-to-WorldIR mapping is not verified in this audit; the frontend was not run. | `frontend/` |
| G8 | Duplicate-frame detection and adaptive video frame sampling are not on the production path. | `perception/quality/` |
| G9 | Benchmarks (ScanNet / Matterport3D / Replica / TUM / ETH3D) are not integrated; no dataset licence review was performed. | `benchmarks/` |
| G10 | The multi-engine pass runs every available engine on every evidence-world job; there is no per-job engine budget or opt-out beyond `RE_EVIDENCE_WORLD`. | `apps/api/evidence_world_stage.py` |

---

## 4. What this change set actually implements

1. `perception/engine_orchestrator.py` — rewritten execution layer:
   availability-gated dispatch, CPU-aware VRAM gating, Kahn dependency ordering
   with `dependency_unmet` skips, per-engine timeouts, cancellation,
   transient-only retry, output validation (`invalid_output`), and an
   `EngineExecution` contract (identity, architecture, checkpoint, version,
   licence, device, attempts, runtime, VRAM, frames, units, status, error,
   provenance) for **every** engine, including the ones that did not run.
2. `perception/engine_evidence.py` — `bbox_iou`, scale/frame-aware
   `compare_observations` that never reports a relative value in metres.
3. `perception/engine_adapters.py` — fabricated LightGlue confidence removed;
   VRAM telemetry fixed.
4. `perception/engine_matrix_bridge.py` (new) — converts engine observations into
   the canonical `EngineObservation` / `EvidenceContext` / `RunRecord` store, with
   content-keyed idempotency, no calibration claim, and a reload check.
5. `perception/multi_engine_stage.py` — honest aggregate status (a run with any
   non-succeeding engine is `partial`, never `succeeded`) and the matrix import.
6. `apps/api/evidence_world_stage.py` — the multi-engine pass now runs **before**
   world persistence, imports into the run's `EvidenceMatrix`, writes
   `multi_engine_evidence.json` into the run directory, and `persist_world` puts
   that lineage into `world.metadata["multi_engine_evidence"]` — read from the run
   directory so re-persisting the same run reproduces the same content-addressed
   version instead of a new one. A cancelled run starts no engine work.

## 5. Tests added

`tests/test_unified_engine_orchestration.py` (23 tests) and
`tests/test_engine_matrix_bridge.py` (10 tests). They pin each criterion above,
including the negative cases: an unavailable engine is never called, a permanent
failure is not retried, two different objects are not a contradiction, and a
relative position is never compared with a metric one.
