"""Platformization campaign, Phase 80 (External Consumer Test): a tiny
application that consumes Reality Engine as a client of `sdk.reality`
ONLY -- never importing `evidence.*`, `engine.compiler.*`, or
`exporters.*` directly. This is the test that proves the public SDK
boundary in sdk/reality.py is real and sufficient, not merely present.

The reconstruction fixture itself has to come from somewhere internal
(a real ReconstructionResult over synthetic points, same fixture shape
used by tests/test_geometric_reasoning.py) -- that's the "evidence"
input any real caller would supply, not a boundary violation.
"""

from __future__ import annotations

import pytest

from sdk import reality
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)


def _point(x: float, y: float, z: float, counter=[0]) -> ReconstructedPoint:
    counter[0] += 1
    return ReconstructedPoint(position=(x, y, z), track_id=f"pt-{counter[0]:05d}", source_evidence_ids=["ev-1"])


def _room_evidence() -> ReconstructionResult:
    counter = [0]
    points = []
    for i in range(12):
        for j in range(12):
            points.append(_point(i * 0.25, 0.0, j * 0.25, counter))
            points.append(_point(i * 0.25, 2.5, j * 0.25, counter))
    for i in range(10):
        for j in range(10):
            points.append(_point(0.0, i * 0.25, j * 0.25, counter))
            points.append(_point(4.0, i * 0.25, j * 0.25, counter))
    for i in range(30):
        points.append(_point(2.0 + (i % 5) * 0.7, 1.25 + (i % 3) * 0.4, 1.5 + (i % 7) * 0.3, counter))
    camera = ReconstructedCameraPose(evidence_id="ev-1", position=(2.0, 1.25, 1.5), rotation=(1.0, 0.0, 0.0, 0.0))
    return ReconstructionResult(points=points, camera_poses=[camera], registration_status="success")


def test_external_app_full_flow_through_sdk_only():
    """ingest -> compile -> validate -> physics -> export -> diff, calling
    nothing but sdk.reality."""
    evidence = _room_evidence()

    world, diagnostics = reality.compile_world_from_reconstruction(evidence)
    assert len(world.entities) > 0
    assert diagnostics is not None

    report = reality.validate(world)
    assert report.is_valid()

    physics = reality.compile_physics(world)
    assert len(physics.compiled) > 0

    gltf_content, gltf_report = reality.export(world, "gltf")
    assert gltf_report.entities_exported
    assert isinstance(gltf_content, dict)

    usda_content, usda_report = reality.export(world, "usda")
    assert usda_content.startswith("#usda 1.0")

    blender_script, blender_report = reality.export(world, "blender")
    assert "primitive_cube_add" in blender_script

    # Same evidence recompiled must be diffably identical (determinism
    # visible through the SDK's own diff() function).
    world_again, _ = reality.compile_world_from_reconstruction(evidence)
    world_diff = reality.diff(world, world_again)
    assert world_diff.is_empty()


def test_sdk_export_rejects_unknown_format_explicitly():
    evidence = _room_evidence()
    world, _ = reality.compile_world_from_reconstruction(evidence)

    with pytest.raises(reality.UnsupportedExportFormatError):
        reality.export(world, "fbx")


def test_sdk_compile_raises_typed_error_on_empty_evidence():
    empty = ReconstructionResult(points=[], camera_poses=[], registration_status="success")
    with pytest.raises(reality.CompileInputError):
        reality.compile_world_from_reconstruction(empty)
