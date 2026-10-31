"""Room-fidelity fixture (plan T0.1) + room dedupe (plan S1).

SYNTHETIC: a 5 x 4 x 2.6 m box room with one door, one window, two furniture boxes and distinctly coloured surfaces,
rendered through the RGB-D sensor model and compiled by the REAL canonical compiler. Ground truth comes from the
generator, so every assertion is an error against a known answer. Nothing here says anything about a real room
(that needs the user's photos and tape measurements: datasets/README.md).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from scripts.room_fidelity_report import delta_e76
from synthetic.indoor import fidelity_ground_truth, fidelity_room, observe
from synthetic.scene import nearest_quad
from world_ir import EntityType

COMMITTED = Path(__file__).resolve().parents[1] / "synthetic" / "fidelity_room" / "ground_truth.json"
UP = (0.0, 0.0, 1.0)


@lru_cache(maxsize=None)
def _compiled(colors, promote_building: bool):
    obs = observe(fidelity_room(), colors=colors, voxel_m=0.08)
    world, diag = compile_reconstruction_to_world(
        obs.result, CompileOptions(seed=42, up=UP, promote_building=promote_building))
    return obs, world, diag


def _of(world, t):
    return [e for e in world.entities.values() if e.type == t]


def test_committed_ground_truth_matches_the_generator():
    assert json.loads(COMMITTED.read_text(encoding="utf-8")) == json.loads(
        json.dumps(fidelity_ground_truth(fidelity_room())))


def test_ground_truth_has_the_documented_content():
    gt = fidelity_ground_truth(fidelity_room())
    assert gt["schema"] == "room_fidelity_ground_truth/1" and gt["units"] == "m"
    r = gt["room"]
    assert (r["length_m"], r["width_m"], r["height_m"], r["area_m2"]) == (5.0, 4.0, pytest.approx(2.6), 20.0)
    assert sorted(o["kind"] for o in gt["openings"]) == ["door", "window"]
    assert len(gt["furniture"]) == 2 and len(gt["walls"]) == 4
    cols = [w["color_rgb"] for w in gt["walls"]] + [gt["floor"]["color_rgb"], gt["ceiling"]["color_rgb"]]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            assert delta_e76(cols[i], cols[j]) > 20, "surface colours must be well separated"


def test_observe_colour_modes():
    ind = fidelity_room()
    plain = observe(ind, voxel_m=0.2)
    assert all(p.color is None for p in plain.result.points), "default observe must not invent colour"
    alb = observe(ind, colors="albedo", voxel_m=0.2)
    pts = np.array([p.position for p in alb.result.points])
    dist, qi = nearest_quad(ind.scene, pts)
    same = np.array([tuple(p.color) == tuple(int(v) for v in ind.scene.quads[i].color)
                     for p, i in zip(alb.result.points, qi)])
    # misses are points on a seam between two quads (nearest quad ambiguous to within the sensor noise), nothing else
    assert same.mean() > 0.99 and dist[~same].max() < 0.005, (same.mean(), dist[~same].max())
    assert [p.position for p in alb.result.points] == [p.position for p in plain.result.points]
    with pytest.raises(ValueError):
        observe(ind, colors="bogus")


@pytest.mark.parametrize("promote_building", [False, True])
def test_S1_one_physical_room_compiles_to_exactly_one_room_entity(promote_building):
    """Plan S1 claimed topology.py lacked room dedupe. It has `_find_existing_room_entity`; this pins the outcome:
    the fixture's single room yields exactly one ROOM entity with a unique id, with or without building promotion."""
    _obs, world, diag = _compiled("albedo", promote_building)
    gt_rooms = len({fidelity_ground_truth(fidelity_room())["room"]["id"]})  # the ground truth has one room
    rooms = _of(world, EntityType.ROOM)
    assert diag.rooms_detected == gt_rooms == 1
    assert len(rooms) == 1
    assert len({r.id for r in rooms}) == len(rooms)
    # the one room owns the structure: every wall/floor/ceiling is contained by it exactly once
    contained = [x.target_id for x in rooms[0].relationships if x.kind.value == "contains"]
    struct = {e.id for t in (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING) for e in _of(world, t)}
    assert struct <= set(contained) and len(contained) == len(set(contained))


def test_structure_counts_and_dimensions_match_ground_truth():
    _obs, world, _diag = _compiled("albedo", False)
    # S3: one solid slab per wall is the primary representation; the promoted planes stay as linked evidence
    walls = _of(world, EntityType.WALL)
    slabs = [e for e in walls if e.custom_properties.get("wall_slab")]
    assert (len(slabs), len(_of(world, EntityType.FLOOR)), len(_of(world, EntityType.CEILING))) == (4, 2, 2)
    assert len(walls) == 8, "4 solid wall slabs + 4 plane evidence entities"
    for slab in slabs:
        assert slab.custom_properties["wall_slab"]["openings"] is not None
        assert slab.custom_properties.get("evidence_plane_ids"), slab.id
    cp = _of(world, EntityType.ROOM)[0].custom_properties
    assert sorted((cp["floor_dimension_u_m"], cp["floor_dimension_v_m"])) == pytest.approx([4.0, 5.0], abs=0.03)
    assert cp["height_m"] == pytest.approx(2.6, abs=0.03)


def test_measured_colours_reach_the_surfaces_and_are_absent_without_point_colour():
    gt = fidelity_ground_truth(fidelity_room())
    _obs, world, _ = _compiled("albedo", False)
    ents = [e for e in _of(world, EntityType.FLOOR) + _of(world, EntityType.CEILING) + _of(world, EntityType.WALL)
            if "polygon_slab" in e.semantic_labels or e.custom_properties.get("wall_slab")]
    assert len(ents) == 6, "the six primary surfaces (4 wall slabs, floor slab, ceiling slab)"
    got = sorted(tuple(e.custom_properties["mean_color_rgb"]) for e in ents)
    want = sorted(tuple(c) for c in [gt["floor"]["color_rgb"], gt["ceiling"]["color_rgb"]]
                  + [w["color_rgb"] for w in gt["walls"]])
    assert all(delta_e76(g, w) < 3.0 for g, w in zip(got, want)), (got, want)
    # the plan's colour hypothesis: no per-point colour in -> no colour out (never an invented one)
    _o, bare, _d = _compiled(None, False)
    assert not any(e.custom_properties.get("mean_color_rgb") for e in bare.entities.values())


def test_building_promotion_succeeds_on_the_fidelity_room():
    """D2 fix: a RoomGraph that lists refused planes (furniture tops) must not lose the whole building. Those parts are
    excluded and recorded; the room, storey and building still promote."""
    _obs, world, diag = _compiled("albedo", True)
    assert not [w for w in diag.interior_warnings if "building topology promotion" in w], diag.interior_warnings
    assert len(_of(world, EntityType.BUILDING)) == 1 and len(_of(world, EntityType.STOREY)) == 1
