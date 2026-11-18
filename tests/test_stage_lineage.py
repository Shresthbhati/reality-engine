"""RE-ARCH-000014: artifact -> producing stage, and what a changed artifact invalidates."""
from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.lineage import LineageGraph
from engine.pipeline.stages.runner import run_stages


def test_invalidated_by_follows_the_chain_not_siblings():
    g = LineageGraph()
    g.add("a", [], ["pts"])
    g.add("b", ["pts"], ["mesh"])
    g.add("c", ["mesh"], ["report"])
    g.add("d", [], ["other"])           # unrelated branch
    assert g.producer["mesh"] == "b"
    assert g.invalidated_by("pts") == ["b", "c"]
    assert g.invalidated_by("report") == []


class _S:
    def __init__(self, name, inputs, out):
        self.spec = StageSpec(name, "1", "cpu", True, inputs=inputs)
        self._out = out

    def run(self, ctx):
        now = datetime.now(timezone.utc)
        return StageResult(self.spec.name, "succeeded", "", self._out, {}, [], None, now, now, None)


def test_runner_records_lineage(tmp_path):
    ctx = StageContext("j", tmp_path, {}, {})
    m = run_stages([_S("a", (), {"x": "h1"}), _S("b", ("x",), {"y": "h2"}), _S("c", None, {})], ctx)
    assert m.lineage.producer == {"x": "a", "y": "b"}
    assert m.lineage.invalidated_by("x") == ["b", "c"]   # c declared nothing: assumed to read x and y
