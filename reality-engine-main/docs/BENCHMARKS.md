# Benchmarks

**Honest state: zero benchmarks exist.** `benchmarks/` is an empty
directory (no files at all). Every BUILD_LEDGER entry to date lists
`**Benchmark**: None`. This file exists so that claim is visible and
tracked, not hidden — per the directive's §41 ("no fake completion") and
§39 ("never optimize without measuring").

## Why none yet

Every subsystem built so far (Steps 1-16, REQ-001 through REQ-029)
operates at test-scale entity counts (single digits to low tens) inside
the pytest suite. No subsystem has been exercised at a scale where its
performance characteristics would be measurable or meaningful, and no
optimization has been claimed that would need justifying.

## Known algorithmic ceilings worth benchmarking first

These are the spots in [DECISIONS.md](DECISIONS.md) and
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) already flagged as
"acceptable until proven otherwise" — the first candidates once real
benchmarks exist:

| Subsystem | Ceiling | File |
|---|---|---|
| Collision broadphase | O(n²) pair checks | `engine/physics/collision/broadphase.py` |
| Contact solver | Single-pass, no iteration | `engine/physics/constraints/contact_solver.py` |
| Debris | No spatial partitioning for fragment queries | `engine/physics/destruction/debris.py` |
| Caching | Substring-match pattern invalidation | `engine/world/caching.py` |

## What §39 asks for (not yet built)

reconstruction accuracy, metric scale error, semantic accuracy, material
classification accuracy, entity association accuracy, physics stability,
collision performance, simulation throughput, memory consumption, GPU
utilization, CPU utilization, world serialization time, branch creation
time, query latency, AI command latency.

None of these have a harness yet. Do not report a number for any of
them until one exists and is checked into `benchmarks/` with a
reproducible script — a guessed or extrapolated figure here would be
exactly the "fake completion" the directive prohibits.
