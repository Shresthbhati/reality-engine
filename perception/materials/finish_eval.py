"""Per-surface architectural material finish evaluator (Plan T-3.2).

Evaluates the accuracy of per-surface material finish labels (paint, tile, wood, concrete)
against measured ground-truth specifications for rooms, walls, floors, and ceilings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from world_ir import EntityType


def evaluate_surface_finishes(
    world,
    ground_truth: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """Compare measured surface and room finish labels against ground truth.

    ``ground_truth`` is keyed by room_id, containing:
    {
        "floor": "wood" | "tile" | "concrete" | ...,
        "wall": "paint" | "concrete" | "brick" | "tile" | ...,
        "ceiling": "paint" | "wood" | "concrete" | ...,
    }
    """
    results: Dict[str, Any] = {
        "total_rooms": 0,
        "correct_room_finishes": 0,
        "total_surfaces": 0,
        "correct_surface_finishes": 0,
        "surface_accuracy_pct": 0.0,
        "room_accuracy_pct": 0.0,
        "meets_accuracy_threshold": True,
        "rooms": [],
        "surfaces": [],
        "summary": [],
    }

    rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
    for room in rooms:
        if room.id not in ground_truth:
            continue
        gt = ground_truth[room.id]
        cp = room.custom_properties
        results["total_rooms"] += 1

        pred_floor = cp.get("floor_finish", "unspecified")
        pred_wall = cp.get("wall_finish", "unspecified")
        pred_ceil = cp.get("ceiling_finish", "unspecified")

        gt_floor = gt.get("floor")
        gt_wall = gt.get("wall")
        gt_ceil = gt.get("ceiling")

        floor_ok = (pred_floor == gt_floor) if gt_floor else True
        wall_ok = (pred_wall == gt_wall) if gt_wall else True
        ceil_ok = (pred_ceil == gt_ceil) if gt_ceil else True

        room_all_ok = floor_ok and wall_ok and ceil_ok
        if room_all_ok:
            results["correct_room_finishes"] += 1

        results["rooms"].append({
            "room_id": room.id,
            "expected": {"floor": gt_floor, "wall": gt_wall, "ceiling": gt_ceil},
            "predicted": {"floor": pred_floor, "wall": pred_wall, "ceiling": pred_ceil},
            "pass": room_all_ok,
        })

        # Evaluate individual surfaces in this room
        walls = [e for e in world.entities.values() if e.type == EntityType.WALL
                 and e.custom_properties.get("room_id") == room.id]
        floors = [e for e in world.entities.values() if e.type == EntityType.FLOOR
                  and e.custom_properties.get("room_id") == room.id]
        ceilings = [e for e in world.entities.values() if e.type == EntityType.CEILING
                    and e.custom_properties.get("room_id") == room.id]

        for w in walls:
            if gt_wall:
                results["total_surfaces"] += 1
                w_finish = w.custom_properties.get("finish") or pred_wall
                ok = w_finish == gt_wall
                if ok:
                    results["correct_surface_finishes"] += 1
                results["surfaces"].append({
                    "entity_id": w.id,
                    "kind": "wall",
                    "expected": gt_wall,
                    "predicted": w_finish,
                    "pass": ok,
                })

        for fl in floors:
            if gt_floor:
                results["total_surfaces"] += 1
                fl_finish = fl.custom_properties.get("finish") or pred_floor
                ok = fl_finish == gt_floor
                if ok:
                    results["correct_surface_finishes"] += 1
                results["surfaces"].append({
                    "entity_id": fl.id,
                    "kind": "floor",
                    "expected": gt_floor,
                    "predicted": fl_finish,
                    "pass": ok,
                })

        for cl in ceilings:
            if gt_ceil:
                results["total_surfaces"] += 1
                cl_finish = cl.custom_properties.get("finish") or pred_ceil
                ok = cl_finish == gt_ceil
                if ok:
                    results["correct_surface_finishes"] += 1
                results["surfaces"].append({
                    "entity_id": cl.id,
                    "kind": "ceiling",
                    "expected": gt_ceil,
                    "predicted": cl_finish,
                    "pass": ok,
                })

    if results["total_surfaces"] > 0:
        results["surface_accuracy_pct"] = round(
            (results["correct_surface_finishes"] / results["total_surfaces"]) * 100.0, 2
        )
    if results["total_rooms"] > 0:
        results["room_accuracy_pct"] = round(
            (results["correct_room_finishes"] / results["total_rooms"]) * 100.0, 2
        )

    # Threshold: >= 80% surface accuracy on measured rooms
    if results["surface_accuracy_pct"] < 80.0:
        results["meets_accuracy_threshold"] = False
        results["summary"].append(
            f"Surface finish accuracy {results['surface_accuracy_pct']}% below 80.0% threshold"
        )
    else:
        results["summary"].append(
            f"Surface finish accuracy {results['surface_accuracy_pct']}% meets target (>= 80%)"
        )

    return results
