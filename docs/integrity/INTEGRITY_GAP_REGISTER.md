# Integrity Gap Register — Reality Engine V1

> Consolidated register of all integrity gaps found across the reconstructed world pipeline: Raw Images → Image Metadata → Camera Calibration → Observations → Features → Detection → Segmentation → Association → Geometry → Spatial Relationships → Evidence Fusion → WorldIR → WorldStore → World Versions → Exports.
>
> Part of the KILO integrity mandate. Owner: World Reconstruction Integrity.
> Each gap: ID, description, location, impact, priority, status (FIXED / OPEN), coordinates.
>
> Mandate rule honored: no integrity gap is silently papered over; every gap is either fixed or recorded here with a resolution path.

---

## A. Manufactured confidence / unknown → plausible default (highest priority)

These violate the mandate rule "Do not manufacture confidence" and "Convert unknown values into plausible defaults." `None` must mean "not measured."

| ID | Gap | Location | Impact | Priority | Status |
|---|---|---|---|---|---|
| IG-A1 | `Observation.confidence` defaulted to `0.5` (fabricated "half-confident") | `world_ir/observations.py:75`, `:115` | Observations with no measurement read as 0.5 confident | P0 | FIXED (→ `Optional[float] = None`; `association._fuse`/`_new_entity` guard None) |
| IG-A2 | `WorldIR.global_confidence` defaulted to `0.5` | `world_ir/world_v1.py:212`, `:301`, `:364` | World with no measured confidence reads as 0.5 confident | P0 | OPEN (legacy contract — preserve 0.5 on load; new worlds use None) |
| IG-A3 | `Measurement.precision` defaulted to `0.01` | `world_ir/schema_v1.py:198` | Manufactured precision where none measured | P1 | OPEN |
| IG-A4 | `make_rigid_transform` default uncertainty `confidence=1.0` | `world_ir/coordinates.py:218` | Transform with no stated confidence reads as certain | P1 | OPEN |
| IG-A5 | `EntityTrack.confidence` default `0.0` | `perception/instances/association.py:84` | Track with no measurements reads as 0.0 (acceptable as "no confidence" initializer, but not a measurement) | P2 | OPEN (kept as initializer) |
| IG-A6 | `compose.py` `confidence=0.5` on inferred entities/relationships | `world_ir/compose.py:262,267,269` | Manufactured confidence on INFERRED provenance | P1 | OPEN |
| IG-A7 | `metric_depth_backend` `confidence=0.50` | `perception/depth/metric_depth_backend.py:178` | Scene-statistics prior presented as 0.5 confidence | P1 | OPEN |
| IG-A8 | `room_semantics.py` `confidence=0.5` | `perception/architecture/room_semantics.py:267,427` | Room semantics with manufactured confidence | P1 | OPEN |
| IG-A9 | `trajectories/worldir.py` `confidence=0.5` | `trajectories/worldir.py:72` | Trajectory with "no calibrated confidence" reads as 0.5 | P2 | OPEN |
| IG-A10 | `attributes.py` `material_confidence = 0.50` | `perception/entities/attributes.py:394` | Material confidence manufactured | P1 | OPEN |
| IG-A11 | `compose.py:430` `min(s[2].global_confidence ...)` | `world_ir/compose.py:430` | Propagates manufactured 0.5 global confidence | P1 | OPEN |

**Cross-cutting rule:** `None` = not measured, never `0.5`. `confidence_basis` required whenever `confidence` is not None.

---

## B. Competing representations (must converge, not add another parallel contract)

| ID | Gap | Location | Impact | Priority | Status |
|---|---|---|---|---|---|
| IG-B1 | Four association implementations | `perception/instances/association.py` (production gated-Hungarian), `perception/entities/association.py` (legacy union-find), `engine/multiengine/association.py` (ray/entity), `engine/multiengine/spatial_intelligence/association.py` (temporal) | Divergent identity results between production vertical_slice and evidence_world_stage | P0 | OPEN (documented; production = instances/association) |
| IG-B2 | Four taxonomy implementations | `perception/entities/taxonomy.py` (canonical, reads config/ontology.yaml), `engine/multiengine/spatial_intelligence/taxonomy.py` (Domain/TaxonomyClass), `perception/style/taxonomy.py`, `perception/attributes/taxonomy.py` | Inconsistent entity identity & geometry contracts | P0 | OPEN |
| IG-B3 | Two `Observation` classes in `world_ir/` | `world_ir/schema_v1.py` (world ledger, `Optional[float]` confidence) vs `world_ir/observations.py` (perception association, was `float=0.5`) | Conceptual collision; contract inconsistency | P0 | FIXED (observations.py aligned to `Optional[float]=None`) |
| IG-B4 | Three confidence representations | `ConfidenceVector`, `Measurement.precision/confidence`, `Uncertainty.confidence`, `Observation.confidence` | No single source of truth for "how sure" | P1 | OPEN |
| IG-B5 | Three scale representations | `reconstruction/scale.py` ScaleState, `world_ir/schema_v1.py` Measurement, `perception/spatial_inputs.py` ScaleStateInput, `perception/engine_evidence.py` CoordinateFrame.scale_state | Inconsistent scale state across the pipeline | P1 | OPEN |
| IG-B6 | Three camera/frame representations | `world_ir/camera.py` PinholeCamera, `perception/spatial_inputs.py` CameraInput, `world_ir/coordinates.py` Frame/Transform | Inconsistent camera frame contracts | P1 | OPEN |

---

## C. Competing orchestrators / pipelines

| ID | Gap | Location | Impact | Priority | Status |
|---|---|---|---|---|---|
| IG-C1 | `vertical_slice.py` (monolithic capture-to-world) | `engine/pipeline/vertical_slice.py` | Core path but stages invoked outside stage runner | P0 | OPEN (refactor plan: thin stage composition) |
| IG-C2 | `progressive.py` (progressive ladder) | `engine/pipeline/progressive.py` | Overlaps vertical_slice; fine stages run both inside ladder and standalone | P0 | OPEN |
| IG-C3 | `universal_pipeline.py` (SECOND full implementation, UNWIRED) | `engine/orchestrator/universal_pipeline.py` | Dead parallel implementation; not called by any production job (STATUS.md S5-05) | P1 | OPEN |
| IG-C4 | `registry.build_stages` (coarse runner) | `engine/pipeline/stages/registry.py` | Two hosts (jobs.py + registry) drive overlapping logic | P0 | OPEN |

---

## D. Stage execution defects

| ID | Gap | Location | Impact | Priority | Status |
|---|---|---|---|---|---|
| IG-D1 | Cache keys hash all accumulated `ctx.artifacts` instead of declared `spec.inputs` | `engine/pipeline/stages/runner.py:76` | Unrelated artifact change invalidates all downstream cache entries | P0 | FIXED (hashes declared `spec.inputs` only) |
| IG-D2 | Timed-out stage threads abandoned; may mutate shared state | `engine/pipeline/stages/runner.py:_run_with_deadline` | Straggler thread mutates `ctx.state`/after timeout | P1 | OPEN (documented abandon contract) |
| IG-D3 | `StageSpec.inputs=None` means "assume all earlier ones (lineage)" — ambiguous | `world_ir/stages/base.py:23` | Ambiguous cache inputs | P2 | OPEN |

---

## E. Evidence-world & spatial intelligence off default path

| ID | Gap | Location | Impact | Priority | Status |
|---|---|---|---|---|---|
| IG-E1 | Evidence-world integration opt-in (`RE_EVIDENCE_WORLD=1`) | `apps/api/evidence_world_stage.py:enabled()` | Gemini's detection/segmentation/spatial lift/association bridge not on canonical path | P0 | OPEN |
| IG-E2 | Spatial intelligence (`s14_spatial_global`) not in default profile | `engine/pipeline/stages/registry.py` | Only in non-default profile; selected only when `RE_SPATIAL_GLOBAL` | P1 | OPEN |
| IG-E3 | `REALITY_DENSE_AUTO` on but `dense_mvs_enabled=False` | `engine/pipeline/dense_gate.py` | Level 3 dense off default product path | P2 | OPEN |
| IG-E4 | `REALITY_OPEN_VOCAB` off by default | — | Open-vocab furniture not on live API path | P2 | OPEN |

---

## F. WorldIR vs evidence-world divergence

| ID | Gap | Location | Impact | Priority | Status |
|---|---|---|---|---|---|
| IG-F1 | Two WorldIRs per job (canonical WorldStore version + per-run evidence world) | `worldstore/store.py` vs `apps/api/evidence_world_stage.py` | Representations diverge; connection only via opt-in propose/accept (frame/unit gate) | P0 | OPEN (DESIGN_DECISIONS.md: MERGE BLOCKED) |
| IG-F2 | WorldIR / evidence-world connection not physically validated beyond run 14 | `docs/engineering/DESIGN_DECISIONS.md` section 8 | Unverified merge path | P1 | OPEN |

---

## G. Pipeline path divergence (API / CLI / benchmark)

| ID | Gap | Location | Impact | Priority | Status |
|---|---|---|---|---|---|
| IG-G1 | API, CLI, benchmark execute different pipeline subsets | `apps/api/jobs.py` vs `apps/cli/main.py` vs `tests/integration/test_release_gate.py` | No single production path; benchmark golden path most complete | P1 | OPEN |

---

## H. Data loss / missing consumer / missing producer

| ID | Gap | Location | Impact | Priority | Status |
|---|---|---|---|---|---|
| IG-H1 | `s12_quality_report` mesh validity metrics never fail the job | `engine/pipeline/stages/s12_quality_report.py` | Degraded mesh never downgrades job grade | P2 | OPEN |
| IG-H2 | `vertical_slice.run_finalize` correction_plan / reconstruction_diagnostics reported but not consumed | `engine/pipeline/vertical_slice.py` | Recommendations never applied | P2 | OPEN |

---

## I. Release-functionality disabled by feature flag

| ID | Flag | Default | Effect | Priority | Status |
|---|---|---|---|---|---|
| IG-I1 | `RE_EVIDENCE_WORLD` | OFF | evidence_world_stage disabled | P0 | OPEN (IG-E1) |
| IG-I2 | `RE_SPATIAL_GLOBAL` | OFF | s14_spatial_global omitted from default profile | P1 | OPEN (IG-E2) |
| IG-I3 | `REALITY_DENSE_AUTO` | ON | but `dense_mvs_enabled=False`; dense off default path | P2 | OPEN (IG-E3) |
| IG-I4 | `REALITY_TEST_BACKEND` | unset | refused unless `REALITY_ENV=test` (jobs.py:789) — enforced | — | OK (test seam) |

---

## J. Jobs / persistence integrity (verified sound)

| ID | Area | Location | Status |
|---|---|---|---|
| IG-J1 | Immutable, content-addressed versions | `worldstore/store.py` | SOUND — verified: atomic writes, hash verification on load, quarantine, lineage, path-safety |
| IG-J2 | Job lifecycle + lease fencing | `apps/api/jobs.py` | SOUND — verified: `ConcurrentModificationError` retryable, cancelled job never erases adopted version, `_verify_outcome` |
| IG-J3 | `_assert_reconstruction_traced` prevents untraceable state | `apps/api/jobs.py` | SOUND — prevents permanently untraceable RECONSTRUCTED entities |

---

## Resolution convention

- **FIXED:** code change made + tests pass (IG-A1, IG-B3, IG-D1).
- **OPEN:** recorded here; resolution requires either a demonstrated-defect refactor with migration plan (per mandate "No architecture rewrite without a demonstrated need") or coordination with the owning subsystem (Sonnet reconstruction, FreeBuff perception, Cline frontend, OpenCode storage/security).
- No gap is silently papered over; every OPEN gap has a location, impact, and priority.
