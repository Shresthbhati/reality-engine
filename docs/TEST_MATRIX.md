# Test Matrix

Maps each test file to the requirement(s) it verifies and its test
category, per the directive's testing taxonomy (§40: UNIT / INTEGRATION /
SYSTEM / END-TO-END / PROPERTY / FUZZ / DETERMINISM / PERFORMANCE /
REGRESSION / SERIALIZATION / SECURITY).

**Honest state: every test file below is UNIT-level (in-process,
assertion-based, no external I/O) except the one row marked
INTEGRATION+DETERMINISM.** PROPERTY, FUZZ, PERFORMANCE, and SECURITY
categories have zero coverage — no test in the suite generates random
inputs, measures throughput/latency, or probes for injection/auth
issues. This is a real gap, not an oversight to gloss over; see
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

| Test file | Requirement(s) | Category | Count |
|---|---|---|---|
| `test_clock.py` | REQ-006 | UNIT | — |
| `test_config.py` | REQ-016 | UNIT | 17 |
| `test_coordinates.py` | REQ-005 | UNIT | — |
| `test_debris_system.py` | REQ-027 | UNIT | 22 |
| `test_dependency_graph.py` | REQ-022, REQ-023 | UNIT | 41 |
| `test_entity.py` | REQ-003 | UNIT | — |
| `test_errors.py` | REQ-013, REQ-017 | UNIT | 18 |
| `test_events.py` | REQ-014 | UNIT | — |
| `test_fracture_system.py` | REQ-025, REQ-009 (determinism assertion) | UNIT | 18 |
| `test_glass_physics.py` | REQ-026 | UNIT | 20 |
| `test_inspector.py` | REQ-030, REQ-002 (v1 schema) | UNIT | 28 |
| `test_integration_determinism.py` | REQ-006, REQ-007, REQ-009, REQ-002 | INTEGRATION + DETERMINISM | — |
| `test_jobs.py` | REQ-007 | UNIT | — |
| `test_logging.py` | REQ-015 | UNIT | 20 |
| `test_physics_backend_golden.py` | REQ-010, REQ-011, REQ-012 | UNIT (golden/regression style) | — |
| `test_physics_debugger.py` | REQ-031, REQ-010, REQ-011, REQ-013 | UNIT | 10 |
| `test_physics_collision.py` | REQ-011 | UNIT | — |
| `test_physics_events.py` | REQ-011, REQ-014 | UNIT | — |
| `test_physics_gyroscopic.py` | REQ-010 | UNIT | — |
| `test_physics_material.py` | REQ-012 | UNIT | — |
| `test_physics_math3.py` | (math primitives, no REQ) | UNIT | — |
| `test_physics_numerics.py` | REQ-013 | UNIT | — |
| `test_physics_rigid_body.py` | REQ-010 | UNIT | — |
| `test_provenance.py` | (provenance model, no REQ) | UNIT | — |
| `test_replay_system.py` | REQ-028 | UNIT | 29 |
| `test_rng.py` | REQ-009 | UNIT | — |
| `test_runtime_foundation.py` | REQ-003, REQ-018–021, REQ-024 | UNIT | — |
| `test_serialization.py` | REQ-004 | UNIT | — |
| `test_units.py` | REQ-008 | UNIT | — |
| `test_viewport.py` | REQ-029 | UNIT | 12 |
| `test_world_ir.py` | REQ-002 | UNIT | — |
| `test_world_ir_v1.py` | REQ-002, REQ-004 | UNIT + REGRESSION (3 of 32 tests guard the 2026-09-07 timestamp-determinism fix, DECISIONS.md #16) | 32 |
| `test_world_runtime.py` | REQ-005, REQ-024 | UNIT + REGRESSION (6 of 8 tests guard the 2026-09-07 V1-schema entity-iteration fix DECISIONS.md #14 and the transform-resolution fix DECISIONS.md #15) | 8 |

**Totals** (per `python -m pytest tests/ -q`): 438 tests, 100% passing,
34 test files. Counts left as "—" above were not individually re-verified
in this pass; re-run `pytest --collect-only -q` per file to refresh.

## Coverage gaps against §40

| Category | Status |
|---|---|
| UNIT | Covered — every subsystem has direct tests |
| INTEGRATION | One test (`test_integration_determinism.py`); no cross-subsystem integration beyond it (e.g. no test drives fracture → debris → replay → viewport together) |
| SYSTEM | None |
| END-TO-END | None — this is the gap the "Reality Engine end-to-end validation scenario" work (see conversation history) is meant to close, blocked on Steps 17-33 |
| PROPERTY | None |
| FUZZ | None |
| DETERMINISM | One explicit test (`test_integration_determinism.py`); several unit tests assert same-seed reproducibility inline (fracture, debris, replay) but there is no dedicated determinism suite across all solvers |
| PERFORMANCE | None — `benchmarks/` directory is empty. See [BENCHMARKS.md](BENCHMARKS.md) |
| REGRESSION | `test_physics_backend_golden.py` serves this role for physics | 
| SERIALIZATION | Covered for WorldIR (`test_world_ir_v1.py`) and per-subsystem `serialize()/deserialize()` methods (fracture, glass, debris, replay) |
| SECURITY | None — no auth/injection/input-validation tests exist anywhere in the suite |
