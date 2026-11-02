"""Large-model behaviour of the post-mapper stages.

Found on the 128-photo South Building run: plane detection (a per-point Python loop inside every RANSAC iteration)
took about a minute per call on 61k points and ran several times per build, and the compile stage then rejected the
world because a corridor listed boundary planes that were never promoted -- which dropped the whole multi-view build
into a 128x single-image depth fallback. These tests pin the fixes: identical plane output, a scale guard on a
realistically sized cloud, and a corridor whose boundary list agrees with the world.
"""

from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace

import numpy as np
import pytest

from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult


def _scene(per_plane: int = 2500, clutter: int = 1500, seed: int = 7) -> ReconstructionResult:
    """Deterministic room-like cloud (metres): floor, ceiling, four walls, a door-sized panel and clutter."""
    rng = np.random.default_rng(seed)

    def sheet(n, axis, value, lo, hi):
        pts = rng.uniform(lo, hi, size=(n, 3))
        pts[:, axis] = value + rng.normal(0, 0.01, n)
        return pts

    chunks = [
        sheet(per_plane, 1, 0.0, (0, 0, 0), (8, 0, 6)), sheet(per_plane, 1, 2.6, (0, 0, 0), (8, 0, 6)),
        sheet(per_plane, 0, 0.0, (0, 0, 0), (0, 2.6, 6)), sheet(per_plane, 0, 8.0, (0, 0, 0), (0, 2.6, 6)),
        sheet(per_plane, 2, 0.0, (0, 0, 0), (8, 2.6, 0)), sheet(per_plane, 2, 6.0, (0, 0, 0), (8, 2.6, 0)),
        sheet(per_plane // 2, 0, 4.0, (0, 0, 0), (0, 2.1, 6)),
        rng.uniform((0, 0, 0), (8, 2.6, 6), size=(clutter, 3)),
    ]
    allp = np.concatenate(chunks)
    return ReconstructionResult(
        points=[ReconstructedPoint(position=tuple(map(float, p)), track_id=f"p{i:06d}", source_evidence_ids=["a", "b"])
                for i, p in enumerate(allp)],
        camera_poses=[], registration_status="success")


def _signature(detection):
    ids = "|".join(",".join(p.inlier_ids) for p in detection.planes)
    return (len(detection.planes), [p.inlier_count for p in detection.planes],
            hashlib.sha256(ids.encode()).hexdigest()[:16], len(detection.unassigned_point_ids))


# ---------------------------------------------------------------------------------------------------------------------
# plane detection
# ---------------------------------------------------------------------------------------------------------------------

def test_inlier_mask_equals_the_scalar_expression_including_points_exactly_on_the_tolerance():
    from perception.geometry.planes import inlier_mask

    rng = np.random.default_rng(3)
    xs, ys, zs = (rng.uniform(-5, 5, 20000) for _ in range(3))
    tol = 0.08
    for _ in range(25):
        n = rng.normal(size=3)
        n = tuple(float(v) for v in n / np.linalg.norm(n))
        d = float(rng.uniform(-3, 3))
        got = inlier_mask(xs, ys, zs, n, d, tol)
        want = np.array([abs(n[0] * x + n[1] * y + n[2] * z + d) <= tol for x, y, z in zip(xs, ys, zs)])
        assert np.array_equal(got, want)

    # points ON the boundary (and one ulp either side) must make the same decision as the scalar form
    edge = [0.0, tol, -tol, np.nextafter(tol, 1.0), np.nextafter(tol, 0.0), np.nextafter(-tol, -1.0)]
    ys_e = np.array(edge)
    zeros = np.zeros_like(ys_e)
    got = inlier_mask(zeros, ys_e, zeros, (0.0, 1.0, 0.0), 0.0, tol)
    want = np.array([abs(0.0 * 0.0 + 1.0 * float(y) + 0.0 * 0.0 + 0.0) <= tol for y in ys_e])
    assert np.array_equal(got, want) and got.tolist() == [True, True, True, False, True, False]


def test_detect_planes_output_is_identical_to_the_original_implementation():
    # Golden values produced by the ORIGINAL per-point implementation (git HEAD before the fix) on this exact scene.
    from perception.geometry.planes import detect_planes

    result = _scene()
    assert len(result.points) == 17750
    n_planes, counts, digest, unassigned = _signature(detect_planes(result, 42))
    assert n_planes == 34
    assert counts[:7] == [2910, 2867, 2467, 2462, 2270, 2263, 1203]      # floor, ceiling, 4 walls, panel
    assert digest == "e0d548b1fb1e1462" and unassigned == 29


def test_detect_planes_scales_to_a_128_photo_sized_cloud():
    """The real 128-photo model has ~61k sparse points; the original implementation needed ~60 s for ONE call
    (measured 59.4 s; 0.89 s after). The bound is generous so the guard catches a regression to per-point Python,
    not machine noise."""
    from perception.geometry.planes import detect_planes

    result = _scene(per_plane=8500, clutter=4000)
    assert len(result.points) > 55000
    t0 = time.monotonic()
    detection = detect_planes(result, 42)
    elapsed = time.monotonic() - t0
    assert elapsed < 25.0, f"detect_planes took {elapsed:.1f}s on {len(result.points)} points"
    assert detection.planes[0].inlier_count > 8000


# ---------------------------------------------------------------------------------------------------------------------
# corridor boundary list agrees with the world
# ---------------------------------------------------------------------------------------------------------------------

class _Corridor:
    def __init__(self, ids):
        self.boundary_element_ids = tuple(ids)

    def to_dict(self):
        return {"corridor_id": "corridor-room-003", "boundary_element_ids": list(self.boundary_element_ids)}


def test_corridor_properties_list_only_promoted_parts_and_record_the_rest():
    from engine.compiler.world_compiler import corridor_properties

    world = SimpleNamespace(entities={"struct-plane-001": object(), "struct-plane-002": object()})
    props = corridor_properties(_Corridor(["struct-plane-001", "struct-plane-036", "struct-plane-002", "struct-plane-245"]), world)
    assert props["boundary_element_ids"] == ["struct-plane-001", "struct-plane-002"]
    assert props["boundary_elements_not_promoted"] == ["struct-plane-036", "struct-plane-245"]


def test_corridor_properties_are_untouched_when_every_part_exists():
    from engine.compiler.world_compiler import corridor_properties

    world = SimpleNamespace(entities={"w1": object(), "w2": object()})
    props = corridor_properties(_Corridor(["w1", "w2"]), world)
    assert props["boundary_element_ids"] == ["w1", "w2"] and "boundary_elements_not_promoted" not in props


def test_validation_gate_still_rejects_a_dangling_boundary_reference():
    """The fix makes the data honest; it must NOT weaken the gate. A corridor that lists a missing part is still an
    error, and the helper's output for the same corridor is not."""
    from engine.compiler.world_compiler import corridor_properties
    from provenance import Provenance
    from world_ir import Entity, EntityType, WorldIR
    from world_ir.validation import validate_world_ir

    def world_with(props):
        w = WorldIR(id="w", name="w", main_branch_id="branch-main-w")
        w.entities["wall-1"] = Entity(id="wall-1", type=EntityType.WALL, name="wall", provenance=Provenance.INFERRED)
        w.entities["corridor-1"] = Entity(id="corridor-1", type=EntityType.CORRIDOR, name="corridor",
                                          custom_properties=props, provenance=Provenance.INFERRED)
        return w

    def codes(w):
        return [i.code for i in validate_world_ir(w).issues]

    corridor = _Corridor(["wall-1", "struct-plane-036"])
    assert "interior_boundary_reference_missing" in codes(world_with(corridor.to_dict()))     # raw data: rejected
    honest = corridor_properties(corridor, world_with({}))
    assert "interior_boundary_reference_missing" not in codes(world_with(honest))             # honest data: accepted


# ---------------------------------------------------------------------------------------------------------------------
# stage timings
# ---------------------------------------------------------------------------------------------------------------------

def test_vertical_slice_reports_per_stage_timings_on_a_realistic_model(tmp_path):
    from evidence.session import EvidenceItem, EvidenceKind
    from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice
    from reconstruction.backend.interface import IReconstructionBackend
    from world_ir.artifact_store import FileArtifactStore

    scene = _scene(per_plane=3000, clutter=1000)
    ids = [f"ev_{i}" for i in range(4)]
    poses = [ReconstructedCameraPose(evidence_id=e, position=(2.0 + i, 1.4, 1.5 + i), rotation=(1.0, 0.0, 0.0, 0.0))
             for i, e in enumerate(ids)]
    points = [ReconstructedPoint(position=p.position, track_id=p.track_id, source_evidence_ids=list(ids[:2]))
              for p in scene.points]

    class Stub(IReconstructionBackend):
        def reconstruct(self, evidence):
            return ReconstructionResult(points=points, camera_poses=poses, registration_status="success")

    items = [EvidenceItem(id=e, kind=EvidenceKind.PHOTO, source_uri=(tmp_path / f"{e}.jpg").as_uri(), sha256="0" * 64)
             for e in ids]
    vs = vertical_slice(items, VerticalSliceOptions(artifact_store=FileArtifactStore(tmp_path / "store"),
                                                    reconstruction_backend=Stub()))
    timings = vs.stage_facts["stage_timings_s"]
    assert {"reconstruction", "scale", "frame", "depth", "compile", "perception", "mesh", "dense",
            "fusion_and_detail", "diagnostics_and_correction_plan"} <= set(timings)
    assert all(isinstance(v, float) and v >= 0.0 for v in timings.values())
    assert sum(timings.values()) < 120.0          # a 17k-point model must not take minutes after the fix


# ---------------------------------------------------------------------------------------------------------------------
# honest user-facing guidance when OUR check fails, not the photos
# ---------------------------------------------------------------------------------------------------------------------

def _engine_notes(detail):
    from engine.pipeline.guidance import build_guidance

    guidance = build_guidance(
        level=1, entity_roles=[], names={}, input_ids=[], registered_ids=[], camera_poses=[],
        contribution_summary=None, bootstrap_facts=None,
        attempts=[{"level": 2, "name": "sparse photogrammetric reconstruction", "outcome": "failed", "detail": detail}])
    return [g for g in guidance if g["kind"] == "engine_note"]


def test_internal_compile_failure_does_not_blame_the_photos():
    notes = _engine_notes("VerticalSliceError: world compile stage failed: compiled world failed validation with 2 issue(s)")
    assert len(notes) == 1 and "not a problem with your photos" in notes[0]["message"]
    assert "share too little" not in notes[0]["message"]
    assert "compiled world failed validation" in notes[0]["basis"]          # the real reason stays on record


def test_registration_failure_still_gets_the_overlap_advice():
    notes = _engine_notes("VerticalSliceError: reconstruction produced nothing usable (status='failed', 0 points) -- not compiling")
    assert len(notes) == 1 and "share too little of the same scene" in notes[0]["message"]
