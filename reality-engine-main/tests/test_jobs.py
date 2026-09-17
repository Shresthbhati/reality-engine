import pytest

from engine.core.jobs import (
    DuplicateJobError,
    JobCycleError,
    JobSystem,
    UnknownDependencyError,
)


def test_independent_jobs_run_in_id_order():
    js = JobSystem()
    order = []
    js.add_job("b", lambda: order.append("b"))
    js.add_job("a", lambda: order.append("a"))
    js.run()
    assert order == ["a", "b"]


def test_dependencies_respected():
    js = JobSystem()
    order = []
    js.add_job("build", lambda: order.append("build"), depends_on=["compile"])
    js.add_job("compile", lambda: order.append("compile"))
    js.run()
    assert order == ["compile", "build"]


def test_deterministic_topological_order():
    js1 = JobSystem()
    js2 = JobSystem()
    for js in (js1, js2):
        js.add_job("d", lambda: None, depends_on=["b", "c"])
        js.add_job("c", lambda: None, depends_on=["a"])
        js.add_job("b", lambda: None, depends_on=["a"])
        js.add_job("a", lambda: None)
    assert js1.topological_order() == js2.topological_order()
    assert js1.topological_order() == ["a", "b", "c", "d"]


def test_duplicate_job_raises():
    js = JobSystem()
    js.add_job("a", lambda: None)
    with pytest.raises(DuplicateJobError):
        js.add_job("a", lambda: None)


def test_unknown_dependency_raises():
    js = JobSystem()
    js.add_job("a", lambda: None, depends_on=["ghost"])
    with pytest.raises(UnknownDependencyError):
        js.run()


def test_cycle_raises():
    js = JobSystem()
    js.add_job("a", lambda: None, depends_on=["b"])
    js.add_job("b", lambda: None, depends_on=["a"])
    with pytest.raises(JobCycleError):
        js.run()


def test_run_returns_results_by_id():
    js = JobSystem()
    js.add_job("x", lambda: 42)
    js.add_job("y", lambda: "hi")
    assert js.run() == {"x": 42, "y": "hi"}
