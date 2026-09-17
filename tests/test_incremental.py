"""Tests for engine/incremental.py (P13 incremental compilation):
the dependency/invalidation graph that lets new evidence update only
the AFFECTED world -- not a full rebuild.

Model (directive P13):
- nodes are compile stages keyed by (stage, input-artifact digests);
- a stage's outputs are content-addressed artifacts;
- recompiling computes each stage's input digest set: unchanged
  digests -> cached artifact reused (no recompute), changed digests
  -> stage re-runs, and its change INVALIDATES every downstream
  dependent stage;
- the result records what was recomputed vs reused -- the audit trail
  for "new evidence updates only affected world".
"""

import pytest


class FakeStage:
    """A deterministic stage: output bytes depend only on input bytes."""

    def __init__(self, name, deps, cost=1):
        self.name = name
        self.deps = deps  # stage names this one consumes
        self.cost = cost
        self.runs = 0

    def compute(self, inputs: dict) -> bytes:
        """A real stage TRANSFORMS its input: the output depends on
        the actual input bytes (a stage that ignored its inputs would
        invalidate nothing, which is exactly the bug this fake must
        not have)."""
        self.runs += 1
        import hashlib

        h = hashlib.sha256()
        h.update(self.name.encode())
        for dep in sorted(self.deps):
            h.update(inputs.get(dep, b""))
        return h.hexdigest().encode()


class TestIncrementalGraph:
    def _pipeline(self):
        capture = FakeStage("capture", ("raw",))
        dense = FakeStage("dense", ("capture",))
        mesh = FakeStage("mesh", ("dense",))
        world = FakeStage("world", ("mesh",))
        return capture, dense, mesh, world

    def test_first_compile_runs_everything(self):
        from engine.incremental import IncrementalCompiler, InMemoryCache

        capture, dense, mesh, world = self._pipeline()
        compiler = IncrementalCompiler(cache=InMemoryCache())
        result = compiler.compile(
            stages=[capture, dense, mesh, world],
            sources={"raw": b"raw-bytes-v1"},
        )
        assert all(s.runs == 1 for s in (capture, dense, mesh, world))
        assert result.recomputed == ["capture", "dense", "mesh", "world"]
        assert result.reused == []

    def test_unchanged_input_reuses_everything(self):
        from engine.incremental import IncrementalCompiler, InMemoryCache

        capture, dense, mesh, world = self._pipeline()
        compiler = IncrementalCompiler(cache=InMemoryCache())
        compiler.compile(stages=[capture, dense, mesh, world],
                         sources={"raw": b"raw-bytes-v1"})
        result = compiler.compile(stages=[capture, dense, mesh, world],
                                  sources={"raw": b"raw-bytes-v1"})
        assert result.recomputed == []
        assert result.reused == ["capture", "dense", "mesh", "world"]
        assert all(s.runs == 1 for s in (capture, dense, mesh, world))

    def test_new_evidence_invalidates_only_downstream(self):
        from engine.incremental import IncrementalCompiler, InMemoryCache

        capture, dense, mesh, world = self._pipeline()
        compiler = IncrementalCompiler(cache=InMemoryCache())
        compiler.compile(stages=[capture, dense, mesh, world],
                         sources={"raw": b"raw-bytes-v1"})
        # New evidence arrives: capture re-runs, and everything
        # downstream (dense -> mesh -> world) invalidates with it.
        result = compiler.compile(stages=[capture, dense, mesh, world],
                                  sources={"raw": b"raw-bytes-v2"})
        assert result.recomputed == ["capture", "dense", "mesh", "world"]
        assert capture.runs == 2 and dense.runs == 2

    def test_independent_new_stage_does_not_invalidate(self):
        # A NEW independent source branch (e.g. a second capture)
        # must not recompute the first branch's stages.
        from engine.incremental import IncrementalCompiler, InMemoryCache

        capture, dense, mesh, world = self._pipeline()
        second = FakeStage("second_capture", ())
        fused = FakeStage("fusion", ("mesh", "second_capture"))
        compiler = IncrementalCompiler(cache=InMemoryCache())
        compiler.compile(
            stages=[capture, dense, mesh, world],
            sources={"raw": b"raw-v1", "second_capture": b"raw-b1"},
        )
        second.runs = 0  # not part of the first compile
        result = compiler.compile(
            stages=[capture, dense, mesh, world, second, fused],
            sources={"raw": b"raw-v1", "second_capture": b"raw-b1"},
        )
        assert "capture" not in result.recomputed
        assert "dense" not in result.recomputed
        assert "fusion" in result.recomputed  # new stage computes
        assert second.runs == 1

    def test_cache_survives_new_compiler_instance(self):
        # The cache must be content-addressed and externally persistent
        # (the restart property): a fresh compiler over the same cache
        # root reuses, not recomputes.
        import tempfile
        from pathlib import Path

        from engine.incremental import FileCache, IncrementalCompiler

        with tempfile.TemporaryDirectory() as td:
            capture, dense, mesh, world = self._pipeline()
            root = Path(td)
            c1 = IncrementalCompiler(cache=FileCache(root))
            c1.compile(stages=[capture, dense, mesh, world],
                       sources={"raw": b"raw-v1"})
            runs_after_first = sum(s.runs for s in (capture, dense, mesh, world))

            c2 = IncrementalCompiler(cache=FileCache(root))
            result = c2.compile(stages=[capture, dense, mesh, world],
                                sources={"raw": b"raw-v1"})
            assert result.recomputed == []
            runs_after_second = sum(s.runs for s in (capture, dense, mesh, world))
            assert runs_after_second == runs_after_first

    def test_changed_stage_output_changes_downstream_digests(self):
        from engine.incremental import IncrementalCompiler, InMemoryCache

        capture, dense, mesh, world = self._pipeline()
        compiler = IncrementalCompiler(cache=InMemoryCache())
        r1 = compiler.compile(stages=[capture, dense, mesh, world],
                              sources={"raw": b"v1"})
        r2 = compiler.compile(stages=[capture, dense, mesh, world],
                              sources={"raw": b"v2"})
        # World's output digest differs between the two inputs.
        assert r1.outputs["world"] != r2.outputs["world"]
