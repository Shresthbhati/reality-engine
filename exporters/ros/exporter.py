"""ROS export: a Gazebo SDF 1.9 WORLD (the world format ROS 2 simulation consumes via ros_gz / gazebo_ros).

One static model per box entity: a <pose> at the box centre (SDF is Z-up, right-handed, metres -- the same frame as
WorldIR, so no conversion) and a <box> collision + visual of the real extents. No materials, no physics properties
beyond `static`, no sensors, no URDF/TF: WorldIR carries none of that and none is invented. Entities without real
bounds are skipped with a reason. Not run through Gazebo here (no Gazebo on the build machine): structure is
validated by tests, loading is UNVERIFIED.
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape, quoteattr

from exporters.boxes import collect_boxes, fmt, make_report

SDF_VERSION = "1.9"


def _ident(raw: str, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", raw) or "entity"
    if base[0].isdigit():
        base = "e_" + base
    name, n = base, 2
    while name in used:
        name, n = f"{base}_{n}", n + 1
    used.add(name)
    return name


def export_to_sdf_with_report(world, artifact_store=None):
    boxes, skipped, reasons = collect_boxes(world)
    used: set = set()
    lines = ['<?xml version="1.0"?>', f'<sdf version="{SDF_VERSION}">',
             f"  <world name={quoteattr(_ident(world.id, set()))}>"]
    for b in boxes:
        size = " ".join(fmt(v) for v in b.size)
        pose = " ".join(fmt(v) for v in b.center) + " 0 0 0"
        lines += [f"    <model name={quoteattr(_ident(b.id, used))}>",
                  "      <static>true</static>",
                  f"      <!-- worldir: type={escape(b.type)} provenance={escape(b.provenance)} confidence={fmt(b.confidence)} -->",
                  f"      <pose>{pose}</pose>",
                  '      <link name="link">',
                  f'        <collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>',
                  f'        <visual name="visual"><geometry><box><size>{size}</size></box></geometry></visual>',
                  "      </link>", "    </model>"]
    lines += ["  </world>", "</sdf>", ""]
    content = "\n".join(lines)
    return content, make_report("sdf", world, content, boxes, skipped, reasons)
