"""S3/S4 on the synthetic L-room (deterministic point cloud, cameras looking along the room): solid wall / floor /
ceiling slabs in WorldIR, evidence planes kept and linked, glTF carries the wall meshes, openings are carved into their
host wall, and a gap no camera viewed is NOT carved."""

import base64
import math
import struct

import numpy as np
import pytest

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from exporters.gltf.exporter import export_to_gltf
from perception.architecture.wall_slabs import promote_room_layout
from provenance import Uncertainty
from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructionResult
from reconstruction.meshing.mesh import MeshData
from tests.test_room_semantics import build_points, result_of
from world_ir.artifact_store import MemoryArtifactStore
from world_ir.validation import ValidationSeverity, validate_world_ir


def _quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4
    q[1 + i] = 0.25 * s
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def _pose(eid, pos, forward):
    z = np.array(forward, float)
    xr = np.cross(z, [0, 0, 1]); xr /= np.linalg.norm(xr)
    return ReconstructedCameraPose(evidence_id=eid, position=pos, rotation=_quat(np.stack([xr, np.cross(z, xr), z], axis=1)),
                                   uncertainty=Uncertainty(confidence=0.95))


def poses():
    spots = ((1.5, 1.2), (3.0, 1.5), (4.5, 1.3), (1.5, 3.8), (2.2, 2.5))
    return [_pose(f"ev-{si}-{k}", (x, y, 1.5), (math.cos(k * math.pi / 4), math.sin(k * math.pi / 4), 0.0))
            for si, (x, y) in enumerate(spots) for k in range(8)]


def cameras_facing_north():
    """Cameras mid-room all looking at +y: away from the y=0 window wall."""
    return [_pose(f"n{i}", (x, 1.5, 1.5), (0.0, 1.0, 0.0)) for i, x in enumerate((1.5, 2.5, 3.5, 4.5))]


def _vol(m):
    v = np.array(m.vertices)
    return sum(float(np.dot(v[a], np.cross(v[b], v[c]))) for a, b, c in m.faces) / 6.0


@pytest.fixture(scope="module")
def built():
    r = result_of(build_points())
    r = ReconstructionResult(points=r.points, camera_poses=poses(), registration_status="success")
    store = MemoryArtifactStore()
    world, diag = compile_reconstruction_to_world(r, CompileOptions(up=(0, 0, 1), artifact_store=store))
    return world, diag, store, r


def _slabs(world):
    return sorted((e for e in world.entities.values() if "wall_slab" in e.semantic_labels), key=lambda e: e.id)


def _mesh(world, store, ent):
    return MeshData.from_bytes(store.get(world.geometries[ent.geometry_ids[0]].data_uri))


def test_closed_loop_of_six_solid_wall_entities_with_true_L_dimensions(built):
    world, diag, store, _ = built
    walls = _slabs(world)
    assert len(walls) == 6
    assert sorted(round(w.custom_properties["length_m"]) for w in walls) == [2, 3, 3, 3, 5, 6]
    assert all(world.geometries[w.geometry_ids[0]].type.value == "mesh" for w in walls)
    floor = world.entities["floor-slab-room-001"]
    area = _vol(_mesh(world, store, floor)) / floor.custom_properties["thickness_m"]
    assert area == pytest.approx(24.0, rel=0.02)
    assert not [i for i in validate_world_ir(world).issues if i.severity is ValidationSeverity.ERROR]
    assert not [w for w in diag.interior_warnings if "room layout" in w]
    rep = world.metadata["room_layout"]["rooms"][0]
    assert rep["closed"] and rep["inferred_walls"] == []


def test_labelled_defaults_and_measured_heights(built):
    for w in _slabs(built[0]):
        cp = w.custom_properties
        assert cp["thickness_basis"] == "default_unmeasured" and cp["height_basis"] == "measured"
        assert cp["height_m"] == pytest.approx(2.5, abs=0.06) or cp["height_m"] == pytest.approx(2.8, abs=0.06)


def test_plane_entities_stay_as_evidence_linked_to_the_slab(built):
    world = built[0]
    ids = {s.id for s in _slabs(world)}
    w = _slabs(world)[0]
    assert w.custom_properties["evidence_plane_ids"]
    for pid in w.custom_properties["evidence_plane_ids"]:
        assert world.entities[pid].custom_properties["slab_wall_id"] in ids
        assert any(r.target_id == pid and r.kind.value == "attached_to" for r in w.relationships)


def test_gltf_export_contains_real_wall_meshes(built):
    world, _, store, _ = built
    g = export_to_gltf(world, store)
    buf = base64.b64decode(g["buffers"][0]["uri"].split(",", 1)[1])
    by_name = {n["name"]: n for n in g["nodes"]}
    for w in _slabs(world):
        node = by_name[w.name]
        assert node["mesh"] != 0, "not the shared placeholder cube"
        prim = g["meshes"][node["mesh"]]["primitives"][0]
        acc = g["accessors"][prim["attributes"]["POSITION"]]
        m = _mesh(world, store, w)
        assert acc["count"] == len(m.vertices) and prim["mode"] == 4
        bv = g["bufferViews"][acc["bufferView"]]
        xyz = np.array(struct.unpack_from(f"<{acc['count'] * 3}f", buf, bv["byteOffset"])).reshape(-1, 3)
        assert np.allclose(xyz + np.array(node["translation"]), np.array(m.vertices), atol=1e-3)
    assert "floor-slab-room-001" in by_name


def test_window_is_carved_into_its_host_wall_with_the_measured_rectangle(built):
    world, _, store, _ = built
    win = world.entities["window-001"]
    iw = win.custom_properties["in_wall"]
    assert iw["carved"] and iw["observation_state"] == "observed"
    host = world.entities[iw["wall_id"]]
    assert (iw["width_m"], iw["height_m"], iw["sill_m"]) == pytest.approx((2.0, 1.2, 0.9), abs=0.1)
    assert any(r.target_id == host.id and r.kind.value == "part_of" for r in win.relationships)
    assert any(r.target_id == win.id and r.kind.value == "contains" for r in host.relationships)
    cp = host.custom_properties
    t = cp["thickness_m"]
    assert _vol(_mesh(world, store, host)) == pytest.approx(
        cp["length_m"] * cp["height_m"] * t - iw["width_m"] * iw["height_m"] * t, rel=0.01)
    assert cp["wall_slab"]["openings"][0]["entity_id"] == win.id


def test_gap_in_a_region_no_camera_viewed_is_not_an_opening(built):
    world, _, store, r = built
    away = ReconstructionResult(points=r.points, camera_poses=cameras_facing_north(), registration_status="success")
    rep = promote_room_layout(world, away, (0, 0, 1), artifact_store=store)
    win = world.entities["window-001"]
    iw = win.custom_properties["in_wall"]
    assert iw["observation_state"] == "unobserved_gap" and iw["carved"] is False and iw["viewed_fraction"] < 0.5
    cp = world.entities[iw["wall_id"]].custom_properties
    assert _vol(_mesh(world, store, world.entities[iw["wall_id"]])) == pytest.approx(
        cp["length_m"] * cp["height_m"] * cp["thickness_m"], rel=0.01)
    assert cp["wall_slab"]["openings"] == [] and cp["wall_slab"]["unobserved_gaps"][0]["entity_id"] == win.id
    assert rep["rooms"][0]["unobserved_gaps"][0]["id"] == win.id


def test_rerun_is_idempotent(built):
    world, _, store, r = built
    promote_room_layout(world, r, (0, 0, 1), artifact_store=store)
    assert len(_slabs(world)) == 6
    assert len([e for e in world.entities.values() if e.id.startswith("floor-slab")]) == 1
