# benchmarks

Real, stdlib-only timing benchmarks (`harness.py` + `physics_bench.py` +
`quality_bench.py`, run via `python -m benchmarks.run_all`). See
`docs/CAPABILITY_MATRIX.md` for scope: covers rigid-body stepping (2
scales), command-pipeline throughput, and quality-report scaling only --
not reconstruction, export, or semantic accuracy, none of which have
enough real implementation yet to benchmark.
