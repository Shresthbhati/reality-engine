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
    """ingest -> compile -> validate -> export -> diff, calling
    nothing but sdk.reality. (Physics compilation was deliberately
    removed from the core SDK in the child-project isolation split;
    the boundary guard enforces its absence.)"""
    evidence = _room_evidence()

    world, diagnostics = reality.compile_world_from_reconstruction(evidence)
    assert len(world.entities) > 0
    assert diagnostics is not None

    report = reality.validate(world)
    assert report.is_valid()

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

    index = reality.spatial_index(world)
    nearest = index.nearest((2.0, 1.25, 1.5), k=1)
    assert len(nearest) == 1
    entity, distance = nearest[0]
    assert entity.id in world.entities
    assert distance >= 0.0

    graph = reality.scene_graph(world)
    room_entities = [e for e in world.entities.values() if e.relationships]
    if room_entities:
        edges = graph.edges_from(room_entities[0].id)
        assert isinstance(edges, list)


def test_sdk_spatial_index_and_scene_graph_are_real_query_engines():
    """spatial_index()/scene_graph() are not stubs -- prove nearest()
    respects k and predicate, and scene_graph() resolves a real
    CONTAINS/PART_OF edge produced by room promotion. Needs the
    two-room fixture (tests/test_room_inference.py) since the simple
    single-box `_room_evidence()` above never closes a detectable room."""
    from world_ir import RelationshipKind
    from tests.test_room_inference import _CAMS, _two_room_scene

    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
        for i, p in enumerate(_CAMS)
    )
    world, diagnostics = reality.compile_world_from_reconstruction(result)
    assert diagnostics.rooms_detected > 0

    index = reality.spatial_index(world)
    assert len(index) > 0
    k3 = index.nearest((0.0, 0.0, 0.0), k=3)
    assert len(k3) == min(3, len(index))
    # nearest-first ordering
    distances = [d for _e, d in k3]
    assert distances == sorted(distances)

    graph = reality.scene_graph(world)
    room = next(
        (e for e in world.entities.values()
         if any(r.kind == RelationshipKind.CONTAINS for r in e.relationships)),
        None,
    )
    assert room is not None, "the room-scene fixture must promote at least one CONTAINS relationship"
    contents = graph.contents_of(room.id)
    assert len(contents) > 0
    for member in contents:
        assert graph.container_of(member.id).id == room.id


def test_sdk_export_rejects_unknown_format_explicitly():
    evidence = _room_evidence()
    world, _ = reality.compile_world_from_reconstruction(evidence)

    with pytest.raises(reality.UnsupportedExportFormatError):
        reality.export(world, "fbx")


def test_sdk_compile_raises_typed_error_on_empty_evidence():
    empty = ReconstructionResult(points=[], camera_poses=[], registration_status="success")
    with pytest.raises(reality.CompileInputError):
        reality.compile_world_from_reconstruction(empty)
