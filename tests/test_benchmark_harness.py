import pytest

from benchmarks.harness import run_benchmark
from benchmarks.physics_bench import bench_command_pipeline_throughput, bench_rigid_body_step
from benchmarks.quality_bench import bench_quality_report
from benchmarks.run_all import run_all


def test_run_benchmark_basic():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1

    result = run_benchmark("trivial", iterations=5, fn=fn)
    assert result.iterations == 5
    assert counter["n"] == 5
    assert result.total_seconds >= 0
    assert result.seconds_per_iteration == result.total_seconds / result.iterations


def test_run_benchmark_propagates_exception():
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        run_benchmark("boom", iterations=3, fn=boom)


def test_rigid_body_step_benchmark_smoke():
    # Smoke test at small scale -- correctness, not performance.
    result = bench_rigid_body_step(body_count=3, n_ticks=2, iterations=2)
    assert result.iterations == 2
    assert result.seconds_per_iteration >= 0


def test_command_pipeline_benchmark_smoke():
    result = bench_command_pipeline_throughput(command_count=5, iterations=2)
    assert result.iterations == 2
    assert result.seconds_per_iteration >= 0


def test_quality_report_benchmark_smoke():
    result = bench_quality_report(entity_count=10, iterations=2)
    assert result.iterations == 2
    assert result.seconds_per_iteration >= 0


def test_run_all_returns_results():
    results = run_all()
    assert len(results) > 0
    for r in results:
        assert r.seconds_per_iteration >= 0
