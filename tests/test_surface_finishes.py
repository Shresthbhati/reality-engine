"""Tests for finishes and per-surface material labels (T-3.2).

Verifies:
1. Canonical architectural material classification (paint, tile, wood, concrete).
2. Per-surface material labels (finish, material_label, material_name) attached to walls, floors, and ceilings.
3. Room-level finish aggregation and propagation to schedule exports (build_room_report, report_to_pdf).
4. Label accuracy evaluation reporting on measured rooms against ground-truth specifications.
"""

from __future__ import annotations

from exporters.floorplan import build_room_report, report_to_pdf
from perception.materials.classifier import MaterialEvidence, classify_material
from perception.materials.finish_eval import evaluate_surface_finishes
from perception.materials.material_hypotheses import attach_material_hypotheses
from perception.materials.room_surfaces import propagate_room_finishes
from world_ir import Entity, EntityType, WorldIR


def test_material_classification_canonical_categories():
    """Verify classification of canonical architectural finishes."""
    test_cases = [
        # (name, rgb, kind, rough, sheen, expected_category)
        ("wood_floor", (150, 120, 90), "floor", 0.50, 0.20, "wood"),
        ("painted_wall", (235, 235, 230), "wall", 0.70, 0.05, "plaster"),  # maps to paint
        ("concrete_structure", (145, 145, 145), "concrete", 0.45, 0.10, "concrete"),
        ("tile_floor", (220, 220, 225), "floor", 0.15, 0.70, "tile"),
    ]

    for name, rgb, kind, rough, sheen, expected in test_cases:
        ev = MaterialEvidence(color_rgb=rgb, surface_kind=kind, texture_roughness=rough, specular_sheen=sheen)
        hyps = classify_material(ev)
        assert len(hyps) > 0, f"No hypotheses for {name}"
        top = hyps[0]
        assert top.category.value == expected, f"{name}: expected {expected}, got {top.category.value}"


def test_per_surface_material_label_attachment():
    """Verify material_label, material_name, and finish are set on entities."""
    wall = Entity(id="wall-01", type=EntityType.WALL, name="North Wall")
    wall.custom_properties = {
        "mean_color_rgb": [240, 240, 235],
        "color_source": "image_projected",
        "appearance": {"texture_roughness": 0.65, "specular_sheen": 0.05},
    }

    hyps = attach_material_hypotheses(wall)
    assert hyps is not None
    assert wall.custom_properties["finish"] == "paint"
    assert wall.custom_properties["material_label"] == "paint"
    assert "material_hypotheses" in wall.custom_properties

    floor = Entity(id="floor-01", type=EntityType.FLOOR, name="Main Floor")
    floor.custom_properties = {
        "mean_color_rgb": [155, 125, 95],
        "color_source": "image_projected",
        "appearance": {"texture_roughness": 0.45, "specular_sheen": 0.20},
    }

    attach_material_hypotheses(floor)
    assert floor.custom_properties["finish"] == "wood"
    assert floor.custom_properties["material_label"] == "wood"


def test_propagate_room_finishes_to_schedule():
    """Verify per-surface finishes propagate to room and schedule exports."""
    world = WorldIR(id="w-finishes", name="Test World")

    room = Entity(id="room-101", type=EntityType.ROOM, name="Master Bedroom")
    room.custom_properties = {
        "floor_area_m2": 20.0,
        "footprint_polygon_m": [[0, 0], [5, 0], [5, 4], [0, 4]],
        "height_m": 2.6,
    }
    world.entities[room.id] = room

    # Add 4 walls (3 painted, 1 wood accent)
    coords = [
        ([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
        ([5.0, 0.0, 0.0], [5.0, 4.0, 0.0]),
        ([5.0, 4.0, 0.0], [0.0, 4.0, 0.0]),
        ([0.0, 4.0, 0.0], [0.0, 0.0, 0.0]),
    ]
    for i in range(4):
        wid = f"w-{i+1}"
        finish = "wood" if i == 0 else "paint"
        w = Entity(id=wid, type=EntityType.WALL, name=f"Wall {i+1}")
        w.custom_properties = {
            "room_id": "room-101",
            "finish": finish,
            "length_m": 5.0 if i % 2 == 0 else 4.0,
            "height_m": 2.6,
            "wall_slab": {
                "start_xyz_m": coords[i][0],
                "end_xyz_m": coords[i][1],
                "openings": [],
            },
        }
        world.entities[wid] = w

    # Add floor (wood) and ceiling (paint)
    fl = Entity(id="fl-101", type=EntityType.FLOOR, name="Floor 101")
    fl.custom_properties = {"room_id": "room-101", "finish": "wood"}
    world.entities[fl.id] = fl

    cl = Entity(id="cl-101", type=EntityType.CEILING, name="Ceiling 101")
    cl.custom_properties = {"room_id": "room-101", "finish": "paint"}
    world.entities[cl.id] = cl

    # Propagate to room
    propagate_room_finishes(world)

    assert room.custom_properties["wall_finish"] == "paint"  # Dominant wall finish
    assert room.custom_properties["floor_finish"] == "wood"
    assert room.custom_properties["ceiling_finish"] == "paint"

    # Export room report
    report = build_room_report(world)
    r0 = report["rooms"][0]
    assert r0["floor_finish"] == "wood"
    assert r0["wall_finish"] == "paint"
    assert r0["ceiling_finish"] == "paint"

    # Export PDF
    pdf_bytes = report_to_pdf(report)
    assert len(pdf_bytes) > 200
    assert b"Finishes" in pdf_bytes


def test_evaluate_surface_finishes_on_measured_rooms():
    """Verify evaluation harness computes label accuracy against ground truth."""
    world = WorldIR(id="w-eval", name="Multi-Room Survey")

    # Room 1: Living room (wood floor, painted walls, painted ceiling)
    r1 = Entity(id="r1", type=EntityType.ROOM, name="Living Room")
    r1.custom_properties = {"floor_finish": "wood", "wall_finish": "paint", "ceiling_finish": "paint"}
    world.entities["r1"] = r1

    for i in range(4):
        w = Entity(id=f"r1-w{i}", type=EntityType.WALL, name=f"W{i}")
        w.custom_properties = {"room_id": "r1", "finish": "paint"}
        world.entities[w.id] = w

    fl1 = Entity(id="r1-fl", type=EntityType.FLOOR, name="FL1")
    fl1.custom_properties = {"room_id": "r1", "finish": "wood"}
    world.entities[fl1.id] = fl1

    cl1 = Entity(id="r1-cl", type=EntityType.CEILING, name="CL1")
    cl1.custom_properties = {"room_id": "r1", "finish": "paint"}
    world.entities[cl1.id] = cl1

    # Room 2: Bathroom (tile floor, tile walls, painted ceiling)
    r2 = Entity(id="r2", type=EntityType.ROOM, name="Bathroom")
    r2.custom_properties = {"floor_finish": "tile", "wall_finish": "tile", "ceiling_finish": "paint"}
    world.entities["r2"] = r2

    for i in range(4):
        w = Entity(id=f"r2-w{i}", type=EntityType.WALL, name=f"W{i}")
        w.custom_properties = {"room_id": "r2", "finish": "tile"}
        world.entities[w.id] = w

    fl2 = Entity(id="r2-fl", type=EntityType.FLOOR, name="FL2")
    fl2.custom_properties = {"room_id": "r2", "finish": "tile"}
    world.entities[fl2.id] = fl2

    cl2 = Entity(id="r2-cl", type=EntityType.CEILING, name="CL2")
    cl2.custom_properties = {"room_id": "r2", "finish": "paint"}
    world.entities[cl2.id] = cl2

    # Ground truth
    gt = {
        "r1": {"floor": "wood", "wall": "paint", "ceiling": "paint"},
        "r2": {"floor": "tile", "wall": "tile", "ceiling": "paint"},
    }

    eval_result = evaluate_surface_finishes(world, gt)
    assert eval_result["meets_accuracy_threshold"] is True
    assert eval_result["surface_accuracy_pct"] == 100.0
    assert eval_result["room_accuracy_pct"] == 100.0
    assert eval_result["total_surfaces"] == 12  # 8 walls, 2 floors, 2 ceilings
    assert eval_result["correct_surface_finishes"] == 12
