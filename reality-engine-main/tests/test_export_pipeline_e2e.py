"""End-to-end test: real reconstruction -> plane detection -> orientation
classification -> WorldIR promotion -> all three exporters.

This exercises the actual gap discovered and fixed in this session: before
`evidence/promote_planes.py` set `Entity.transform`, every promoted
wall/floor/ceiling entity had `transform=None` and every exporter (gltf,
usd, blender) silently skipped it -- three working exporters with a real
compiler pipeline that produced nothing they could see. This test proves
the pipeline is now actually connected, not just that each half works in
isolation.
"""

from __future__ import annotations

from evidence.promote_planes import promote_plane_to_entity
from perception.geometry.orientation import classify_planes
from perception.geometry.planes import detect_planes
from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult
from world_ir import EntityType, WorldIR

from exporters.gltf.exporter import export_to_gltf
from exporters.usd.exporter import export_to_usda
from exporters.blender.exporter import export_to_blender_script

_UP = (0.0, 1.0, 0.0)
_CAMS = [(2.0, 1.25, 1.5)]  # inside the room, used to disambiguate floor/ceiling


def _point(x: float, y: float, z: float, counter=[0]) -> ReconstructedPoint:
    counter[0] += 1
    return ReconstructedPoint(
        position=(x, y, z), track_id=f"pt-{counter[0]:05d}", source_evidence_ids=["ev-1"]
    )


def _room_result() -> ReconstructionResult:
    """Same synthetic room fixture as tests/test_geometric_reasoning.py:
    floor y=0, ceiling y=2.5, walls x=0 and x=4."""
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
    return ReconstructionResult(points=points, camera_poses=[], registration_status="success")


def _promoted_world() -> WorldIR:
    result = _room_result()
    det = detect_planes(result, seed=42)
    oriented = classify_planes(det.planes, _CAMS, up=_UP)

    world = WorldIR()
    for n, o in enumerate(oriented):
        if o.role in ("wall", "floor", "ceiling"):
            promote_plane_to_entity(o, result, world, f"struct-{n}")
    return world


def test_promoted_planes_carry_a_real_transform():
    world = _promoted_world()
    structural = [e for e in world.entities.values() if e.type in (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING)]
    assert len(structural) >= 3  # floor + ceiling + at least one wall pair
    for entity in structural:
        assert entity.transform is not None
        assert "position" in entity.transform
        pos = entity.transform["position"]
        # centroid must land inside the room's real extent (x in [0,4], y in [0,2.5], z in [0,2.75])
        assert -0.01 <= pos["x"] <= 4.01
        assert -0.01 <= pos["y"] <= 2.51
        assert -0.01 <= pos["z"] <= 2.76


def test_gltf_export_now_sees_promoted_planes():
    world = _promoted_world()
    gltf = export_to_gltf(world)
    assert len(gltf["nodes"]) >= 3
    assert len(gltf["scenes"][0]["nodes"]) == len(gltf["nodes"])


def test_usd_export_now_sees_promoted_planes():
    world = _promoted_world()
    usda = export_to_usda(world)
    assert usda.count("def Cube") >= 3


def test_blender_export_now_sees_promoted_planes():
    world = _promoted_world()
    script = export_to_blender_script(world)
    assert script.count("primitive_cube_add") >= 3
    # dimensions must come from the real per-plane AABB, never the (2,2,2)
    # fallback, since promote_plane_to_entity always sets bounds_min/max
    assert "obj.dimensions = (2.0, 2.0, 2.0)" not in script
