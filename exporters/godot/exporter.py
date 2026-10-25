"""Godot 4 export: a text scene (.tscn, format 3).

One StaticBody3D per box entity, with a MeshInstance3D (BoxMesh) and a CollisionShape3D (BoxShape3D) of the real
extents. Godot is Y-up, right-handed: WorldIR (x, y, z) -> Godot (x, z, -y), extents (dx, dy, dz) -> (dx, dz, dy).
No materials, scripts or lights are invented. Identical sizes share one sub-resource. Not opened in Godot here (no
Godot on the build machine): structure is validated by tests, loading is UNVERIFIED.
"""

from __future__ import annotations

import re

from exporters.boxes import collect_boxes, fmt, make_report


def _node_name(raw: str, used: set) -> str:
    base = re.sub(r'[./:@%"\s]', "_", raw) or "entity"
    name, n = base, 2
    while name in used:
        name, n = f"{base}_{n}", n + 1
    used.add(name)
    return name


def export_to_tscn_with_report(world, artifact_store=None):
    boxes, skipped, reasons = collect_boxes(world)
    sizes: dict = {}                       # godot-frame size tuple -> index (first-use order: deterministic)
    for b in boxes:
        sizes.setdefault((b.size[0], b.size[2], b.size[1]), len(sizes) + 1)
    out = [f"[gd_scene load_steps={1 + 2 * len(sizes)} format=3]", ""]
    for size, i in sizes.items():
        v = f"Vector3({fmt(size[0])}, {fmt(size[1])}, {fmt(size[2])})"
        out += [f'[sub_resource type="BoxMesh" id="BoxMesh_{i}"]', f"size = {v}", "",
                f'[sub_resource type="BoxShape3D" id="BoxShape3D_{i}"]', f"size = {v}", ""]
    out += [f'[node name="{_node_name(world.id, set())}" type="Node3D"]', ""]
    used: set = set()
    for b in boxes:
        name = _node_name(b.id, used)
        i = sizes[(b.size[0], b.size[2], b.size[1])]
        x, y, z = b.center
        out += [f'[node name="{name}" type="StaticBody3D" parent="."]',
                f"transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {fmt(x)}, {fmt(z)}, {fmt(-y)})",
                f'metadata/worldir_type = "{b.type}"', f'metadata/worldir_provenance = "{b.provenance}"', "",
                f'[node name="Mesh" type="MeshInstance3D" parent="{name}"]', f'mesh = SubResource("BoxMesh_{i}")', "",
                f'[node name="Collision" type="CollisionShape3D" parent="{name}"]',
                f'shape = SubResource("BoxShape3D_{i}")', ""]
    content = "\n".join(out)
    return content, make_report("tscn", world, content, boxes, skipped, reasons)
