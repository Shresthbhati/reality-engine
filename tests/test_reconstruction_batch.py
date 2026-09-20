"""Production batch reconstruction path (REAL_RECONSTRUCTION_PERCEPTION_
CITY_READY increment A): a heterogeneous multi-session image set becomes
canonical ReconstructionContract results through the EXISTING canonical
chain — admission → per-session orchestration → cross-session
registration — with bounded submission and no new contracts.

Everything is composed, not reinvented:

  evidence.importers.import_folder(on_error="record")
    -> reconstruction.robustness_admission.admit_for_reconstruction
    -> ReconstructionOrchestrator.run                      (per session group)
    -> registration.cross_session.align_reconstructed_sessions
    -> reconstruction.batch summary report

Test fixtures: synthetic JPEG photos generated with PIL at test time
(clearly test fixtures — the REAL-data batch gate lives in
test_bad_evidence_real.py / the south-building run record). The backend
is a labeled TEST DOUBLE (ring_test_double) producing honest geometry.

Batch rules under test:
  - Session grouping is DECLARED per source path component, never
    guessed; unmatched items are recorded under "<unassigned>".
  - A group below the orchestrator's minimum image count fails ALONE
    with a machine-readable reason; the batch continues (degraded).
  - Bounded memory: the report is summary-sized, not O(frames).
  - Determinism: same input -> identical JSON report.
  - Unavailable backends surface their probe detail verbatim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402  (project dependency: evidence importers use it)

from evidence.importers import import_folder  # noqa: E402
from evidence.packages import DeterministicPackageBuilder  # noqa: E402
from reconstruction.backend.interface import (  # noqa: E402
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)


# ---------------------------------------------------------------- helpers

def _write_photo(path: Path, seed: int) -> None:
    """A decodable, sharp, well-exposed synthetic photo (test fixture)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(48, 64, 3), dtype=np.int16)
    ys, xs = np.mgrid[0:48, 0:64]
    grad = ((xs * 2 + ys + seed * 7) % 90).astype(np.int16)
    img = np.clip(base + grad[:, :, None], 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(path, format="JPEG", quality=92)


def _session_folder(base: Path, name: str, n: int, seed: int = 0) -> Path:
    d = base / name
    d.mkdir(parents=True)
    for i in range(n):
        _write_photo(d / f"frame_{i:04d}.jpg", seed=seed + i)
        if i % 5 == 4:  # exact duplicate every 5th frame
            (d / f"frame_{i:04d}_dup.jpg").write_bytes(
                (d / f"frame_{i:04d}.jpg").read_bytes()
            )
    return d


def _ring_points(n: int):
    for i in range(n):
        a = 2 * np.pi * i / max(1, n)
        yield (float(np.cos(a)), float(np.sin(a)), float(i * 0.01))


def _ring_backend():
    """Labeled TEST DOUBLE backend producing honest ring geometry.

    Both sessions reconstruct the SAME ring (the same structure seen
    by two capture runs) in overlapping volumes, so the batch can
    genuinely register session_b into session_a's frame. Non-trivial
    transform recovery is covered by registration/cross_session's own
    suite; this test pins the WIRING.
    """

    class _RingBackend:
        backend_name = "ring_test_double"

        def reconstruct(self, evidence):
            if len(evidence) < 2:
                raise ValueError("insufficient evidence (test double)")
            pts = [
                ReconstructedPoint(
                    position=(x, y, z),
                    track_id=f"t{i}",
                    source_evidence_ids=[e.id for e in evidence[:2]],
                )
                for i, (x, y, z) in enumerate(_ring_points(400))
            ]
            poses = [
                ReconstructedCameraPose(
                    evidence_id=evidence[0].id,
                    position=(0.0, 0.0, 0.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                )
            ]
            return ReconstructionResult(
                points=pts, camera_poses=poses, registration_status="success"
            )

    return _RingBackend()


def _unavailable_backend():
    class _Unavailable:
        backend_name = "unavailable_backend"

        @staticmethod
        def availability_probe():
            return False, "backend unavailable: COLMAP not installed"

        def reconstruct(self, evidence):
            raise RuntimeError("should not be called")

    return _Unavailable()


def _capture(tmp_path):
    _session_folder(tmp_path, "session_a", 6, seed=100)
    _session_folder(tmp_path, "session_b", 6, seed=500)
    builder = DeterministicPackageBuilder(seed="batch-test")
    import_folder(builder, str(tmp_path), on_error="record")
    return builder.build().to_evidence_items()


# ---------------------------------------------------------------- tests

class TestBatchGrouping:
    def test_groups_follow_declared_source_prefixes(self, tmp_path):
        from reconstruction.batch import group_by_session

        items = _capture(tmp_path)
        groups = group_by_session(items, prefixes=("session_a", "session_b"))
        assert sorted(groups) == ["session_a", "session_b"]
        assert all(
            "session_a" in i.source_uri.replace("\\", "/").split("/")
            for i in groups["session_a"]
        )
        assert all(
            "session_b" in i.source_uri.replace("\\", "/").split("/")
            for i in groups["session_b"]
        )

    def test_unassigned_items_are_recorded_not_silently_dropped(self, tmp_path):
        from reconstruction.batch import group_by_session

        items = _capture(tmp_path)
        groups = group_by_session(items, prefixes=("session_a",))
        assert "session_b" not in groups
        assert groups["<unassigned>"]

    def test_prefix_matches_path_components_not_substrings(self, tmp_path):
        from reconstruction.batch import group_by_session

        # session_a_backup must NOT be captured by the session_a prefix.
        _session_folder(tmp_path, "session_a", 6, seed=100)
        _session_folder(tmp_path, "session_a_backup", 2, seed=900)
        builder = DeterministicPackageBuilder(seed="batch-test")
        import_folder(builder, str(tmp_path), on_error="record")
        items = builder.build().to_evidence_items()

        groups = group_by_session(items, prefixes=("session_a",))
        assigned = groups["session_a"]
        assert all(
            "session_a_backup" not in i.source_uri.replace("\\", "/").split("/")
            for i in assigned
        )
        backup_items = groups["<unassigned>"]
        assert backup_items, "backup-session items must be recorded as unassigned"

    def test_grouping_is_deterministic(self, tmp_path):
        from reconstruction.batch import group_by_session

        items = _capture(tmp_path)
        g1 = group_by_session(items, prefixes=("session_b", "session_a"))
        g2 = group_by_session(items, prefixes=("session_a", "session_b"))
        assert list(g1.keys()) == list(g2.keys()) == ["session_a", "session_b"]
        assert [i.id for i in g1["session_a"]] == [i.id for i in g2["session_a"]]


class TestBatchRun:
    def test_two_sessions_reconstruct_and_register(self, tmp_path):
        from reconstruction.batch import run_batch

        report = run_batch(
            _capture(tmp_path),
            prefixes=("session_a", "session_b"),
            backends=[_ring_backend()],
            reference_session="session_a",
        )
        assert set(report["sessions"]) == {"session_a", "session_b"}
        for name, s in report["sessions"].items():
            assert s["status"] == "success", s
            assert s["frame_count"] > 0
        assert report["registration"]["resolved"] == ["session_b"]
        assert report["registration"]["unresolved"] == []
        assert report["outcome"] == "complete"

    def test_undersized_group_fails_alone(self, tmp_path):
        from reconstruction.batch import run_batch

        _session_folder(tmp_path, "session_a", 6, seed=100)
        _session_folder(tmp_path, "session_b", 1, seed=500)
        builder = DeterministicPackageBuilder(seed="batch-test")
        import_folder(builder, str(tmp_path), on_error="record")
        items = builder.build().to_evidence_items()

        report = run_batch(
            items,
            prefixes=("session_a", "session_b"),
            backends=[_ring_backend()],
            reference_session="session_a",
        )
        assert report["sessions"]["session_a"]["status"] == "success"
        assert report["sessions"]["session_b"]["status"] == "failed"
        assert "reason" in report["sessions"]["session_b"]
        assert report["sessions"]["session_b"]["reason"]
        assert report["outcome"] == "degraded"

    def test_report_is_json_serializable_and_deterministic(self, tmp_path):
        from reconstruction.batch import run_batch

        items = _capture(tmp_path)
        r1 = run_batch(
            items,
            prefixes=("session_a", "session_b"),
            backends=[_ring_backend()],
            reference_session="session_a",
        )
        r2 = run_batch(
            items,
            prefixes=("session_a", "session_b"),
            backends=[_ring_backend()],
            reference_session="session_a",
        )
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_bounded_memory_summaries(self, tmp_path):
        """Batch report carries counts, not O(frames) materialized data."""
        from reconstruction.batch import run_batch

        items = _capture(tmp_path)
        report = run_batch(
            items,
            prefixes=("session_a", "session_b"),
            backends=[_ring_backend()],
            reference_session="session_a",
        )
        blob = json.dumps(report)
        assert len(blob) < 20_000, "batch report must stay summary-sized"
        assert report["totals"]["points"] == 800
        # Import deduped the 2 duplicate frames before admission, so all
        # remaining items are submitted.
        assert report["totals"]["frames_submitted"] == len(items)

    def test_partial_session_makes_batch_degraded(self, tmp_path):
        """Honest vocabulary: a PARTIAL reconstruction is usable but the
        batch is degraded, never 'complete'."""
        from reconstruction.batch import run_batch

        backend = _ring_backend()
        items = _capture(tmp_path)

        class _PartialRing(type(backend)):
            backend_name = "ring_test_double_partial"

            def reconstruct(self, evidence):
                result = backend.reconstruct(evidence)
                return ReconstructionResult(
                    points=result.points,
                    camera_poses=result.camera_poses,
                    registration_status="partial",
                )

        report = run_batch(
            items,
            prefixes=("session_a",),
            backends=[_PartialRing()],
            reference_session="session_a",
        )
        assert report["sessions"]["session_a"]["status"] == "partial"
        assert report["outcome"] == "degraded"
        assert report["sessions"]["session_a"]["points"] > 0

    def test_unavailable_backend_is_explicit(self, tmp_path):
        from reconstruction.batch import run_batch

        report = run_batch(
            _capture(tmp_path),
            prefixes=("session_a",),
            backends=[_unavailable_backend()],
            reference_session="session_a",
        )
        s = report["sessions"]["session_a"]
        assert s["status"] == "failed"
        assert "unavailable" in s["reason"].lower()
        assert report["outcome"] == "degraded"

    def test_reference_fallback_recorded_when_reference_fails(self, tmp_path):
        from reconstruction.batch import run_batch

        # session_b has 1 frame -> fails; reference requested as session_b.
        _session_folder(tmp_path, "session_a", 6, seed=100)
        _session_folder(tmp_path, "session_b", 1, seed=500)
        builder = DeterministicPackageBuilder(seed="batch-test")
        import_folder(builder, str(tmp_path), on_error="record")
        items = builder.build().to_evidence_items()

        report = run_batch(
            items,
            prefixes=("session_a", "session_b"),
            backends=[_ring_backend()],
            reference_session="session_b",
        )
        assert report["registration"]["reference_session"] == "session_a"
        assert "did not reconstruct" in report["registration"]["reasons"]["<reference>"]

    def test_memory_budget_defers_overrun_frames(self, tmp_path):
        from reconstruction.batch import MemoryBudget, run_batch

        items = _capture(tmp_path)
        report = run_batch(
            items,
            prefixes=("session_a", "session_b"),
            backends=[_ring_backend()],
            reference_session="session_a",
            memory_budget=MemoryBudget(max_frames_per_group=3),
        )
        a = report["sessions"]["session_a"]
        # 6 unique frames (5 originals + 1 non-duplicate), 3 submitted:
        # 3 deferred as a recorded fact.
        assert a["deferred_capacity"] == 3
        assert a["frame_count"] == 3
        assert report["totals"]["deferred_capacity"] > 0
        assert report["outcome"] == "degraded"
