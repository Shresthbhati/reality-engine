"""Tests for the reconstruction orchestrator
(reconstruction/orchestrator.py): evidence validation, availability
probing, preference-ordered selection with distinct gates, fallback on
decline/failure/registration-failed, provenance stamping, and full
attempt-log diagnostics.

Most fixtures are hand-rolled stub backends so every selection outcome is
engineered, not incidental; the end-to-end chain test reuses the room
suite's synthetic scene so expectations stay hand-computable.
"""

from __future__ import annotations

import pytest

from evidence.session import EvidenceItem, EvidenceKind
from provenance import Uncertainty
from reconstruction.backend.interface import (
    IReconstructionBackend,
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.orchestrator import (
    EvidenceValidationError,
    ReconstructionOrchestrationError,
    ReconstructionOrchestrator,
    validate_evidence,
)
from tests.test_room_inference import _UP  # noqa: F401  (room-suite up vector)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _items(n=3):
    return [
        EvidenceItem(id=f"photo-{i}", kind=EvidenceKind.PHOTO,
                     source_uri=f"file://capture/photo-{i}.jpg")
        for i in range(n)
    ]


class _StubBackend(IReconstructionBackend):
    """Scriptable stub: availability, acceptance, and result are settable."""

    backend_name = "stub"

    def __init__(self, result=None, available=True, available_detail="",
                 accepts=None, raises=None):
        self.result = result
        self._available = available
        self._available_detail = available_detail
        self.accepts = accepts  # orchestrator probes backend.accepts
        self._raises = raises
        self.calls = 0

    def reconstruct(self, evidence):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self.result


def _ok_result(n_poses=3, n_points=4, status="success"):
    return ReconstructionResult(
        points=[
            ReconstructedPoint(
                position=(float(i), 0.5, 0.25),
                track_id=f"track-{i}",
                source_evidence_ids=["photo-0", "photo-1"],
                uncertainty=Uncertainty(confidence=0.8, note="track"),
            )
            for i in range(n_points)
        ],
        camera_poses=[
            ReconstructedCameraPose(
                evidence_id=f"photo-{i}",
                position=(1.0, 1.0, 1.0),
                rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(confidence=0.9),
            )
            for i in range(n_poses)
        ],
        registration_status=status,
    )


# ---------------------------------------------------------------------------
# Evidence validation
# ---------------------------------------------------------------------------


class TestEvidenceValidation:
    def test_empty_evidence_refused_before_any_backend_runs(self):
        with pytest.raises(EvidenceValidationError) as exc:
            validate_evidence([])
        assert "no evidence items" in exc.value.issues[0]

    def test_fewer_than_two_images_refused(self):
        with pytest.raises(EvidenceValidationError) as exc:
            validate_evidence(_items(1))
        assert any("insufficient image evidence" in i for i in exc.value.issues)

    def test_duplicate_evidence_ids_refused(self):
        dup = EvidenceItem(id="photo-0", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg")
        with pytest.raises(EvidenceValidationError) as exc:
            validate_evidence([dup, _items(2)[0], _items(2)[1]])
        assert any("duplicate evidence ids" in i for i in exc.value.issues)

    def test_validation_summary_counts_kinds(self):
        summary = validate_evidence(_items(3))
        assert summary["total"] == 3
        assert summary["image"] == 3
        assert summary["kinds"] == ["photo"]


# ---------------------------------------------------------------------------
# Selection: availability and acceptance are distinct gates
# ---------------------------------------------------------------------------


class TestSelection:
    def test_first_available_accepting_backend_wins_in_order(self):
        b1 = _StubBackend(result=_ok_result())
        b2 = _StubBackend(result=_ok_result())
        orch = ReconstructionOrchestrator([b1, b2])
        run = orch.run(_items(3))
        # First backend should be selected (b1), display name "stub"
        assert run.diagnostics.backend_name == "stub"
        assert b1.calls == 1 and b2.calls == 0
        assert run.diagnostics.attempts[-1].outcome == "succeeded"

    def test_unavailable_backend_declined_and_fallback_runs(self):
        probe = lambda: (False, "colmap binary not on PATH")  # noqa: E731
        b1 = _StubBackend(available=False)
        b1.availability_probe = probe
        b2 = _StubBackend(result=_ok_result())
        orch = ReconstructionOrchestrator([b1, b2])
        run = orch.run(_items(3))
        assert run.diagnostics.backend_name == "stub#2"
        assert b1.calls == 0
        assert run.diagnostics.attempts[0].outcome == "declined"
        assert "colmap binary not on PATH" in run.diagnostics.attempts[0].detail

    def test_backend_that_declines_evidence_is_skipped_not_failed(self):
        b1 = _StubBackend(accepts=lambda ev: (False, "needs >= 5 images"))
        b2 = _StubBackend(result=_ok_result())
        orch = ReconstructionOrchestrator([b1, b2])
        run = orch.run(_items(3))
        # b1 declines, b2 succeeds - b2 gets "stub#2" as display name
        assert run.diagnostics.backend_name == "stub#2"
        assert run.diagnostics.attempts[0].outcome == "declined"
        assert "needs >= 5 images" in run.diagnostics.attempts[0].detail
        assert b1.calls == 0

    def test_raising_backend_falls_through_with_error_captured(self):
        b1 = _StubBackend(raises=RuntimeError("colmap crashed"))
        b2 = _StubBackend(result=_ok_result())
        orch = ReconstructionOrchestrator([b1, b2])
        run = orch.run(_items(3))
        assert run.diagnostics.backend_name == "stub#2"
        failed = [a for a in run.diagnostics.attempts if a.outcome == "failed"]
        assert len(failed) == 1
        assert "colmap crashed" in failed[0].error
        assert b2.calls == 1

    def test_registration_failed_result_is_a_failure_not_a_success(self):
        b1 = _StubBackend(result=_ok_result(status="failed"))
        b2 = _StubBackend(result=_ok_result())
        orch = ReconstructionOrchestrator([b1, b2])
        run = orch.run(_items(3))
        assert run.diagnostics.backend_name == "stub#2"
        assert run.diagnostics.final_status == "success"
        assert any(a.outcome == "failed" and "registration" in a.detail
                   for a in run.diagnostics.attempts)

    def test_all_backends_exhausted_raises_with_full_attempt_log(self):
        b1 = _StubBackend(available=False)
        b1.availability_probe = lambda: (False, "no binary")
        b2 = _StubBackend(raises=RuntimeError("boom"))
        b3 = _StubBackend(result=_ok_result(status="failed"))
        orch = ReconstructionOrchestrator([b1, b2, b3])
        with pytest.raises(ReconstructionOrchestrationError) as exc:
            orch.run(_items(3))
        assert len(exc.value.attempts) == 3
        outcomes = [a.outcome for a in exc.value.attempts]
        assert outcomes == ["declined", "failed", "failed"]
    def test_duplicate_backend_instances_get_distinct_display_names(self):
        """Two same-class instances (e.g. two COLMAP quality presets) are a
        legitimate fallback chain: attempt logs disambiguate them as
        stub / stub#2 instead of refusing them."""
        b1 = _StubBackend(raises=RuntimeError("preset A crashed"))
        b2 = _StubBackend(result=_ok_result())
        orch = ReconstructionOrchestrator([b1, b2])
        run = orch.run(_items(3))
        assert run.diagnostics.backend_name == "stub#2"
        # First attempt is b1 (stub), second is b2 (stub#2)
        assert run.diagnostics.attempts[0].backend_name == "stub"
        assert run.diagnostics.attempts[1].backend_name == "stub#2"

    def test_partial_registration_reported_honestly(self):
        orch = ReconstructionOrchestrator([_StubBackend(result=_ok_result(status="partial"))])
        run = orch.run(_items(3))
        assert run.diagnostics.final_status == "partial"
        assert run.diagnostics.attempts[-1].outcome == "partial"


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------


class TestDetection:
    def test_detect_reports_every_candidate_without_running(self):
        b1 = _StubBackend()
        b1.availability_probe = lambda: (True, "colmap 3.9 found")
        b2 = _StubBackend()
        b2.availability_probe = lambda: (False, "torch missing")
        orch = ReconstructionOrchestrator([b1, b2])
        report = orch.detect_available_backends()
        assert report[0]["available"] is True
        assert "colmap 3.9" in report[0]["detail"]
        assert report[1]["available"] is False
        assert b1.calls == 0 and b2.calls == 0

    def test_backend_without_probe_assumed_available(self):
        orch = ReconstructionOrchestrator([_StubBackend()])
        report = orch.detect_available_backends()
        assert report[0]["available"] is True
        assert "no availability probe" in report[0]["detail"]

    def test_raising_probe_reported_not_raised(self):
        b = _StubBackend()
        b.availability_probe = lambda: (_ for _ in ()).throw(OSError("disk error"))
        orch = ReconstructionOrchestrator([b])
        report = orch.detect_available_backends()
        assert report[0]["available"] is False
        assert "OSError" in report[0]["detail"]

    def test_probe_list_first_success_wins(self):
        b = _StubBackend()
        b.availability_probe = [
            lambda: (False, "no pycolmap"),
            lambda: (True, "colmap CLI found"),
        ]
        orch = ReconstructionOrchestrator([b])
        report = orch.detect_available_backends()
        assert report[0]["available"] is True
        assert "colmap CLI" in report[0]["detail"]


# ---------------------------------------------------------------------------
# Provenance stamping + diagnostics
# ---------------------------------------------------------------------------


class TestProvenanceAndDiagnostics:
    def test_points_and_poses_stamped_with_backend_identity(self):
        orch = ReconstructionOrchestrator([_StubBackend(result=_ok_result())])
        run = orch.run(_items(3))
        for p in run.result.points:
            assert p.uncertainty.note is not None
            assert "backend=stub" in p.uncertainty.note
            assert "track" in p.uncertainty.note  # backend note preserved
        for pose in run.result.camera_poses:
            assert "backend=stub" in pose.uncertainty.note

    def test_backend_confidence_preserved_verbatim(self):
        orch = ReconstructionOrchestrator([_StubBackend(result=_ok_result())])
        run = orch.run(_items(3))
        assert all(p.uncertainty.confidence == 0.8 for p in run.result.points)
        assert all(p.uncertainty.confidence == 0.9 for p in run.result.camera_poses)

    def test_diagnostics_account_for_every_input_and_attempt(self):
        b1 = _StubBackend(available=False)
        b1.availability_probe = lambda: (False, "no binary")
        b2 = _StubBackend(result=_ok_result())
        orch = ReconstructionOrchestrator([b1, b2])
        run = orch.run(_items(3))
        d = run.diagnostics
        assert d.evidence_count == 3
        assert d.image_evidence_count == 3
        assert len(d.attempts) == 2
        assert d.final_status == "success"

    def test_diagnostics_serialize_and_roundtrip(self):
        orch = ReconstructionOrchestrator([_StubBackend(result=_ok_result())])
        run = orch.run(_items(3))
        d = run.diagnostics.to_dict()
        assert d["final_status"] == "success"
        assert d["attempts"][0]["backend_name"] == "stub"
        assert isinstance(d["duration_s"], float)

    def test_orchestrator_is_deterministic_in_attempt_order(self):
        b1 = _StubBackend(raises=RuntimeError("x"))
        b2 = _StubBackend(result=_ok_result())
        orch = ReconstructionOrchestrator([b1, b2])
        r1 = orch.run(_items(3))
        r2 = orch.run(_items(3))
        # Attempt ORDER and every decision field are deterministic; duration_s
        # is wall-clock telemetry, deliberately excluded from the comparison.
        def strip(run):
            return [
                {k: v for k, v in a.to_dict().items() if k != "duration_s"}
                for a in run.diagnostics.attempts
            ]
        assert strip(r1) == strip(r2)
        assert [a.backend_name for a in r1.diagnostics.attempts] == \
               [a.backend_name for a in r2.diagnostics.attempts]
        assert r1.diagnostics.backend_name == r2.diagnostics.backend_name


# ---------------------------------------------------------------------------
# COLMAP backend orchestrator integration (probe + acceptance gates)
# ---------------------------------------------------------------------------


class TestColmapBackendIntegration:
    """The one production backend now exposes availability_probe and
    accepts for the orchestrator. Shape assertions are machine-independent;
    whether COLMAP itself is installed only changes the probe's boolean,
    never the contract.

    """

    def _backend(self):
        from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
        return ColmapReconstructionBackend()

    def test_availability_probe_returns_bool_with_detail(self):
        backend = self._backend()
        ok, detail = backend.availability_probe()
        assert isinstance(ok, bool)
        assert isinstance(detail, str) and detail
        # Detail names the binary either way (found path or not-found reason).
        assert "colmap" in detail.lower()

    def test_accepts_declines_below_two_view_floor(self):
        backend = self._backend()
        one = [EvidenceItem(id="p1", kind=EvidenceKind.PHOTO, source_uri="file://x.jpg")]
        ok, why = backend.accepts(one)
        assert ok is False
        assert ">= 2" in why

    def test_orchestrator_uses_colmap_gates_in_a_chain(self):
        """Machine-independent chain: the COLMAP backend first in line
        declines a 2-photo batch only when its PROBE fails (COLMAP not
        installed); when COLMAP is present the batch passes validation and
        the real binary would run -- so this test pins the gate wiring, not
        the environment: for a 2-photo batch the COLMAP attempt must be
        declined-by-availability on a COLMAP-less machine and never a
        decline on grounds of count (its accepts gate passes 2 images).
        The stub fallback must still be reached when COLMAP cannot run."""
        from reconstruction.backend.colmap_backend import ColmapReconstructionBackend

        colmap = ColmapReconstructionBackend()
        ok, _ = colmap.availability_probe()
        fake = _StubBackend(result=_ok_result())
        orch = ReconstructionOrchestrator([colmap, fake])
        run = orch.run(_items(2))
        colmap_attempt = run.diagnostics.attempts[0]
        if ok:
            # COLMAP present: it accepts 2 images, so the real binary runs
            # (or its execution fails) -- the fallback is NOT reached first.
            assert colmap_attempt.outcome in ("succeeded", "partial", "failed")
        else:
            # COLMAP absent: honest availability decline, fallback runs.
            assert colmap_attempt.outcome == "declined"
            assert colmap_attempt.available is False
            assert run.diagnostics.backend_name == "stub"
            assert fake.calls == 1


# ---------------------------------------------------------------------------
# End-to-end: orchestrator -> world compiler -> validated WorldIR
# ---------------------------------------------------------------------------


def _room_scene_with_cameras():
    """The hand-computable 2.5x2.25x2 m room scene + the three inside
    cameras the classifier needs (same construction as the compiler
    suite's fixture, so expectations stay closed-form)."""
    from tests.test_room_inference import _CAMS, _two_room_scene
    from reconstruction.backend.interface import ReconstructedCameraPose as _Pose

    result = _two_room_scene()
    result.camera_poses.extend(
        _Pose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
        for i, p in enumerate(_CAMS)
    )
    return result


class TestEndToEnd:
    def test_orchestrated_result_drives_full_room_pipeline(self):
        """The orchestrator's stamped output must be drop-in for the
        geometric pipeline: result -> detect_planes -> classify ->
        plane summaries -> rooms, all hand-computed expectations inherited
        from the room suite's fixture (one 2.5x2.25x2 m room, area
        5.625 m^2)."""
        from perception.geometry.planes import detect_planes
        from perception.geometry.orientation import classify_planes
        from evidence.promote_planes import positions_by_plane
        from evidence.promote_rooms import detect_rooms, plane_summary_from

        result = _room_scene_with_cameras()
        first = _StubBackend(available=False)
        first.availability_probe = lambda: (False, "unavailable")
        orch = ReconstructionOrchestrator([first, _StubBackend(result=result)])

        evidence = [
            EvidenceItem(id=p.evidence_id, kind=EvidenceKind.PHOTO,
                         source_uri=f"file://capture/{p.evidence_id}.jpg")
            for p in result.camera_poses
        ]
        run = orch.run(evidence)
        assert run.diagnostics.final_status == "success"
        assert run.diagnostics.backend_name == "stub#2"
        assert len(run.result.camera_poses) == len(result.camera_poses)

        camera_positions = [p.position for p in run.result.camera_poses]
        detection = detect_planes(run.result, seed=7)
        oriented = classify_planes(detection.planes, camera_positions, _UP)
        positions = positions_by_plane(run.result, oriented)
        summaries = [
            plane_summary_from(plane, positions[plane.plane.plane_id])
            for plane in oriented
            if plane.role != "unknown"
        ]
        rooms = detect_rooms(summaries, _UP)
        # Candidates include honest refusals (the interior slab comes back
        # NO_CLOSED_RING); exactly one candidate is a detected room.
        detected = [r for r in rooms if r.status == "detected"]
        assert len(detected) == 1, "orchestrated result must yield the one real room"
        assert detected[0].ring is not None
        # The ring's vertices span the hand-computable 2.5x2.25 m footprint
        # (in floor-plane-local 2D coordinates, whose axes may be mirrored
        # relative to world); shoelace area over the closed ring is exact:
        # 2.5 * 2.25 = 5.625.
        assert detected[0].ring.shoelace_area() == pytest.approx(5.625)
        xs = [v[0] for v in detected[0].ring.vertices]
        ys = [v[1] for v in detected[0].ring.vertices]
        assert round(max(xs) - min(xs), 6) == 2.5
        assert round(max(ys) - min(ys), 6) == 2.25
        assert any(r.status == "NO_CLOSED_RING" for r in rooms), \
            "the interior slab must be refused honestly, not silently dropped"

    def test_compiler_consumes_orchestrated_run_end_to_end(self):
        """Orchestrator -> world compiler -> validated WorldIR: the real
        integration seam the campaign asks for, with evidence ids flowing
        through to the compiled world."""
        from engine.compiler.world_compiler import compile_reconstruction_to_world
        from engine.compiler import CompileOptions

        result = _room_scene_with_cameras()
        orch = ReconstructionOrchestrator([_StubBackend(result=result)])
        evidence = [
            EvidenceItem(id=p.evidence_id, kind=EvidenceKind.PHOTO,
                         source_uri=f"file://capture/{p.evidence_id}.jpg")
            for p in result.camera_poses
        ]
        run = orch.run(evidence)

        world, diagnostics = compile_reconstruction_to_world(
            run.result, CompileOptions(seed=7)
        )
        assert diagnostics.validation_issues == []  # gate passed
        entity_types = {e.type.name for e in world.entities.values()}
        assert "WALL" in entity_types and "FLOOR" in entity_types
        assert diagnostics.rooms_detected == 1
