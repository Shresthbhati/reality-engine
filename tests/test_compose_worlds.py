"""world_ir.compose: merge placed worlds into one, rigidly transformed, ids namespaced, nothing invented."""

from __future__ import annotations

import copy
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import Provenance
from world_ir.compose import ComposeError, compose_worlds
from world_ir.connectors import is_generated, make_connector
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR

ID_KEY = re.compile(r"(^|_)ids?$")


@pytest.fixture(scope="module")
def room() -> WorldIR:
    from room_fidelity_report import build_and_compile
    return build_and_compile(colors="albedo", voxel_m=0.12)[2]


def y_room(w=4.0, d=3.0, h=2.5) -> WorldIR:
    """Hand-made Y-up room (x east, z south, y up): footprint 0..w x 0..d, ceiling at y=h, one wall, one table."""
    poly = [[0, 0, 0], [w, 0, 0], [w, 0, d], [0, 0, d]]
    room = Entity(id="room-1", type=EntityType.ROOM, provenance=Provenance.INFERRED, confidence=0.9,
                  custom_properties={
                      "footprint_polygon_xyz_m": poly, "footprint_polygon_m": [[p[0], p[2]] for p in poly],
                      "footprint_basis": {"e1": [1, 0, 0], "e2": [0, 0, 1], "up": [0, 1, 0]},
                      "wall_slab_ids": ["wall-1"], "ceiling_height_m": h})
    wall = Entity(id="wall-1", type=EntityType.WALL, provenance=Provenance.INFERRED, confidence=0.9,
                  transform={"position": {"x": w / 2, "y": h / 2, "z": 0.0}}, geometry_ids=["geom-wall-1"],
                  custom_properties={"room_id": "room-1", "wall_slab": {
                      "start_xyz_m": [0, 0, 0], "end_xyz_m": [w, 0, 0], "direction": [1, 0, 0],
                      "outward": [0, 0, -1], "up": [0, 1, 0], "length_m": w, "height_m": h, "thickness_m": 0.1}})
    table = Entity(id="table-1", type=EntityType.UNKNOWN, provenance=Provenance.INFERRED, confidence=0.6,
                   custom_properties={"room_id": "room-1", "bounds_min": [1, 0, 1], "bounds_max": [2, 0.7, 2]})
    from world_ir.schema_v1 import Relationship, RelationshipKind
    room.relationships.append(Relationship(RelationshipKind.CONTAINS, "wall-1", 0.9, Provenance.INFERRED))
    wall.relationships.append(Relationship(RelationshipKind.PART_OF, "room-1", 0.9, Provenance.INFERRED))
    g = Geometry(id="geom-wall-1", type=GeometryType.PLANE, bounds_min=Vector3(0, 0, -0.05),
                 bounds_max=Vector3(w, h, 0.05), provenance=Provenance.INFERRED, confidence=0.9)
    wd = WorldIR(id="y-room", name="Y room", entities={e.id: e for e in (room, wall, table)}, geometries={g.id: g})
    wd.metadata = {"compiled_from": {"up": [0.0, 1.0, 0.0]}}
    return wd


def check_integrity(world: WorldIR, prefixes: tuple[str, ...]) -> None:
    """Every id is namespaced, every reference resolves, no un-namespaced id leaks through custom_properties."""
    assert world.validate() == []
    for coll in (world.entities, world.geometries):
        for k, v in coll.items():
            assert k == v.id
            assert k.startswith(prefixes), k
    for e in world.entities.values():
        for r in e.relationships:
            assert r.target_id in world.entities, (e.id, r.target_id)

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if ID_KEY.search(k) and k != "source_world_id":
                        for s in (v if isinstance(v, list) else [v]):
                            if isinstance(s, str) and s:
                                assert s.startswith(prefixes), (e.id, k, s)
                    walk(v)
            elif isinstance(o, list):
                for i in o:
                    walk(i)

        walk(e.custom_properties)
        if e.custom_properties.get("room_id"):
            assert e.custom_properties["room_id"] in world.entities


def footprint_pts(world: WorldIR, wid: str) -> np.ndarray:
    return np.array([p for e in world.entities.values()
                     if e.type is EntityType.ROOM and e.custom_properties.get("source_world_id") == wid
                     for p in e.custom_properties["footprint_polygon_xyz_m"]])


def centre_xy(pts: np.ndarray, h=(0, 1)) -> tuple[float, float]:
    return tuple(float((pts[:, i].min() + pts[:, i].max()) / 2) for i in h)


# ---- two compiled (Z-up) rooms -----------------------------------------------------------------------------------

def test_compose_two_compiled_rooms_lands_where_placed(room):
    before = copy.deepcopy(room.to_dict())
    out = compose_worlds([("wA", "A", room, 10.0, 20.0, 0.0, 1.0, "v1"), ("wB", "B", room, 40.0, 20.0, 90.0, 1.0, "v2")],
                         name="Pair", kind="house")
    assert room.to_dict() == before  # sources untouched
    check_integrity(out, ("wA/", "wB/", "compose/"))
    ca, cb = centre_xy(footprint_pts(out, "wA")), centre_xy(footprint_pts(out, "wB"))
    assert ca == pytest.approx((10.0, 20.0), abs=1e-6) and cb == pytest.approx((40.0, 20.0), abs=1e-6)
    src = footprint_pts(room, "x") if False else np.array(
        [p for e in room.entities.values() if e.type is EntityType.ROOM
         for p in e.custom_properties["footprint_polygon_xyz_m"]])
    sw, sd = np.ptp(src[:, 0]), np.ptp(src[:, 1])
    pa, pb = footprint_pts(out, "wA"), footprint_pts(out, "wB")
    assert (np.ptp(pa[:, 0]), np.ptp(pa[:, 1])) == pytest.approx((sw, sd), abs=1e-6)
    assert (np.ptp(pb[:, 0]), np.ptp(pb[:, 1])) == pytest.approx((sd, sw), abs=1e-6)  # yaw 90 swaps extents
    # yaw 0 is a pure translation: a known source position moves by exactly (placement - source footprint centre)
    c0 = centre_xy(src)
    s = room.entities["struct-plane-000"].transform["position"]
    d = out.entities["wA/struct-plane-000"].transform["position"]
    assert (d["x"] - s["x"], d["y"] - s["y"], d["z"] - s["z"]) == pytest.approx((10.0 - c0[0], 20.0 - c0[1], 0.0), abs=1e-9)
    assert out.metadata["composed_from"][1] == {"world_id": "wB", "version_id": "v2", "name": "B",
                                                "translate": [40.0, 20.0], "yaw": 90.0, "scale": 1.0,
                                                "up_converted": False}
    assert all(e.custom_properties["source_world_id"] in ("wA", "wB") for e in out.entities.values()
               if not e.custom_properties.get("composed_by"))
    assert out.entities["wA/struct-plane-000"].provenance == room.entities["struct-plane-000"].provenance


def test_viewer_fields_stay_self_consistent_after_yaw(room):
    out = compose_worlds([("wA", "A", room, 10.0, 20.0, 37.0, 1.0)], name="Rot", kind="house")
    walls = [e for e in out.entities.values() if "wall_slab" in e.custom_properties]
    assert walls
    for e in walls:
        ws, src = e.custom_properties["wall_slab"], room.entities[e.id.split("/", 1)[1]].custom_properties["wall_slab"]
        a, b = np.array(ws["start_xyz_m"]), np.array(ws["end_xyz_m"])
        assert np.linalg.norm(b - a) == pytest.approx(src["length_m"], abs=1e-3)
        assert (b - a) / np.linalg.norm(b - a) == pytest.approx(ws["direction"], abs=1e-3)
        assert float(np.dot(ws["direction"], ws["outward"])) == pytest.approx(0.0, abs=1e-9)
        assert ws["up"] == pytest.approx([0, 0, 1])
        assert [o["entity_id"].split("/")[0] for o in ws["openings"]] == ["wA"] * len(ws["openings"])
    r = next(e for e in out.entities.values() if e.type is EntityType.ROOM)
    cp = r.custom_properties
    e1, e2 = np.array(cp["footprint_basis"]["e1"]), np.array(cp["footprint_basis"]["e2"])
    for p3, p2 in zip(cp["footprint_polygon_xyz_m"], cp["footprint_polygon_m"]):
        assert (float(np.dot(p3, e1)), float(np.dot(p3, e2))) == pytest.approx(tuple(p2), abs=1e-6)
    door = next(e for e in out.entities.values() if e.type is EntityType.DOOR).custom_properties
    mid = (np.array(door["bounds_min"]) + np.array(door["bounds_max"])) / 2
    assert mid[:2] == pytest.approx(np.array(door["position"])[:2], abs=0.06)  # AABB of a rotated box: loose, same centre
    # furniture plan centre is still consistent with its 3D bounds centre
    obj = next(e for e in out.entities.values() if e.id.endswith("object-001")).custom_properties
    ctr = (np.array(obj["bounds_min"]) + np.array(obj["bounds_max"])) / 2
    assert (float(np.dot(ctr, e1)), float(np.dot(ctr, e2))) == pytest.approx(tuple(obj["plan_obb_centre_m"]), abs=0.05)


def test_containers_house_and_neighbourhood(room):
    srcs = [("wA", "A", room, 0, 0, 0, 1.0), ("wB", "B", room, 30, 0, 0, 1.0)]
    house = compose_worlds(srcs, name="H", kind="house")
    t = lambda w, ty: [e for e in w.entities.values() if e.type is ty]  # noqa: E731
    assert len(t(house, EntityType.PLOT)) == 1 and len(t(house, EntityType.BUILDING)) == 2
    assert not t(house, EntityType.BLOCK)
    nb = compose_worlds(srcs, name="N", kind="neighbourhood")
    assert len(t(nb, EntityType.PLOT)) == 2 and len(t(nb, EntityType.BLOCK)) == 1 and len(t(nb, EntityType.LOCALITY)) == 1
    assert not any(e.geometry_ids for e in nb.entities.values() if e.custom_properties.get("composed_by"))
    check_integrity(nb, ("wA/", "wB/", "compose/"))
    room_ent = nb.entities["wA/room-plane-000"]
    parents = [r.target_id for r in room_ent.relationships if r.kind.value == "part_of"]
    assert "compose/building/wA" in parents


# ---- up axis ----------------------------------------------------------------------------------------------------

def test_mixed_up_axes_are_converted_exactly(room):
    y = y_room()
    # Y-up first: the Z-up compiled room is converted into Y-up
    out = compose_worlds([("wY", "Y", y, 5.0, 6.0, 0.0, 1.0), ("wZ", "Z", room, 30.0, 6.0, 0.0, 1.0)],
                         name="M", kind="house")
    assert out.metadata["compiled_from"]["up"] == [0.0, 1.0, 0.0]
    assert out.metadata["composed_from"][1]["up_converted"] is True
    check_integrity(out, ("wY/", "wZ/", "compose/"))
    pz = footprint_pts(out, "wZ")
    assert centre_xy(pz, (0, 2)) == pytest.approx((30.0, 6.0), abs=1e-6)
    assert np.ptp(pz[:, 1]) == pytest.approx(0.0, abs=1e-9)  # floor is flat along the new vertical
    ceil = next(e for e in out.entities.values() if e.id == "wZ/ceiling-slab-room-plane-000-01").custom_properties
    ys = [p[1] for p in ceil["slab"]["polygon_xyz_m"]]
    assert ys == pytest.approx([2.61] * len(ys), abs=0.05) and ceil["slab"]["up"] == pytest.approx([0, 1, 0])
    # and a Z-up first compose converts the Y-up room the other way: ceiling stays above the floor
    out2 = compose_worlds([("wZ", "Z", room, 0.0, 0.0, 0.0, 1.0), ("wY", "Y", y, 20.0, 5.0, 0.0, 1.0)],
                          name="M2", kind="house")
    assert out2.metadata["compiled_from"]["up"] == [0.0, 0.0, 1.0]
    wall = out2.entities["wY/wall-1"].custom_properties["wall_slab"]
    assert wall["up"] == pytest.approx([0, 0, 1])
    assert out2.entities["wY/wall-1"].transform["position"]["z"] == pytest.approx(1.25)  # height above floor, +up


def test_refuses_what_it_cannot_convert_exactly(room):
    bad = y_room()
    bad.entities["wall-1"].custom_properties["wall_slab"]["up"] = [0, 0, 1]  # Y-up world, Z-up entity
    with pytest.raises(ComposeError, match="mixed up axes"):
        compose_worlds([("a", "A", bad, 0, 0, 0, 1.0)], name="x", kind="house")
    tilted = y_room()
    tilted.metadata["compiled_from"]["up"] = [0.0, 0.7, 0.7]
    with pytest.raises(ComposeError, match="neither"):
        compose_worlds([("a", "A", tilted, 0, 0, 0, 1.0)], name="x", kind="house")
    with pytest.raises(ComposeError, match="scale"):
        compose_worlds([("a", "A", y_room(), 0, 0, 0, 2.0)], name="x", kind="house")
    with pytest.raises(ComposeError, match="only once"):
        compose_worlds([("a", "A", y_room(), 0, 0, 0, 1.0), ("a", "A", y_room(), 9, 0, 0, 1.0)], name="x", kind="house")
    with pytest.raises(ComposeError):
        compose_worlds([], name="x", kind="house")


# ---- connectors: captured > generated ---------------------------------------------------------------------------

def test_generated_connector_dropped_when_it_overlaps_a_real_entity():
    y = y_room(4, 3)
    a, b = ("wA", "A", y, 0.0, 0.0, 0.0, 1.0), ("wB", "B", y_room(4, 3), 20.0, 0.0, 0.0, 1.0)
    through = make_connector("driveway", (0.0, 0.0), (20.0, 0.0), 2.0, connector_id="gen/through")  # crosses both rooms
    clear = make_connector("driveway", (0.0, 8.0), (20.0, 8.0), 2.0, connector_id="gen/clear")  # 8 m south of both
    out = compose_worlds([a, b], name="N", kind="neighbourhood", connectors=[through, clear])
    assert "gen/through" not in out.entities and "geom-gen/through" not in out.geometries
    assert "gen/clear" in out.entities
    assert any("dropped generated 'gen/through'" in w for w in out.metadata["compose_warnings"])
    assert [e.id for e in out.entities.values() if is_generated(e)] == ["gen/clear"]
    g = out.entities["gen/clear"]
    assert g.custom_properties["asserted"] is False and g.confidence <= 0.3 and g.provenance is Provenance.GENERATED


def test_connector_touching_a_room_edge_is_not_an_overlap():
    y = y_room(4, 3)  # footprint centred on (0,0) -> spans x -2..2, z -1.5..1.5
    touching = make_connector("corridor", (2.0, -1.0), (12.0, -1.0), 1.0, connector_id="gen/touch")  # starts at x=2.0
    out = compose_worlds([("wA", "A", y, 0.0, 0.0, 0.0, 1.0)], name="T", kind="house", connectors=[touching])
    assert "gen/touch" in out.entities


def test_deterministic_recompose_is_byte_identical(room):
    s = [("wA", "A", room, 1.0, 2.0, 15.0, 1.0)]
    assert compose_worlds(s, name="D", kind="house").to_json() == compose_worlds(s, name="D", kind="house").to_json()
