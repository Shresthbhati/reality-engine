"""SUMO export: the ROAD NETWORK as netconvert plain-XML inputs (``network.nod.xml`` + ``network.edg.xml``).

Run ``netconvert --node-files network.nod.xml --edge-files network.edg.xml -o network.net.xml`` to build the network.

Only ROAD and STREET entities take part. Each is a box footprint; its edge runs along the footprint's long horizontal
axis from end to end, and the short side is its measured width. Edge endpoints that fall on the same 0.5 m grid cell
share one node, so roads that actually meet are connected. SUMO is x = east, y = north, Z-up metres: the WorldIR frame
is used as is (the world's +x/+y are assumed to be east/north; WorldIR has no georeference).

Not invented: speed limits, lane counts, priorities, traffic lights, demand. netconvert's own documented defaults
apply; the measured width is carried as an edge <param>. A footprint with no dominant direction (a square) has no
usable edge and is skipped with a reason, as is every non-road entity. Not run through netconvert here (no SUMO on the
build machine): structure is validated by tests, acceptance by netconvert is UNVERIFIED.

``content`` is a JSON object {filename: text}; ``write_sumo_files`` writes the two files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from xml.sax.saxutils import quoteattr

from exporters.boxes import collect_boxes, fmt, make_report

_ROAD_TYPES = ("road", "street")
_GRID_M = 0.5
_MIN_ASPECT = 1.2          # long side / short side below this: no dominant direction


def _id(raw: str, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", raw) or "edge"
    name, n = base, 2
    while name in used:
        name, n = f"{base}_{n}", n + 1
    used.add(name)
    return name


def export_to_sumo_with_report(world, artifact_store=None):
    all_boxes, skipped, reasons = collect_boxes(world)
    boxes = []
    for b in all_boxes:
        if b.type not in _ROAD_TYPES:
            skipped.append(b.id)
            reasons.append("not a road (the SUMO export is the road network only)")
            continue
        sx, sy = b.size[0], b.size[1]
        if max(sx, sy) / min(sx, sy) < _MIN_ASPECT:
            skipped.append(b.id)
            reasons.append("footprint has no dominant direction, so no edge can be derived")
            continue
        boxes.append(b)

    nodes: dict = {}                      # grid key -> (id, x, y)
    edges = []

    def node(x: float, y: float) -> str:
        key = (round(x / _GRID_M), round(y / _GRID_M))
        if key not in nodes:
            nodes[key] = (f"n{len(nodes) + 1}", x, y)
        return nodes[key][0]

    used: set = set()
    for b in boxes:
        cx, cy = b.center[0], b.center[1]
        if b.size[0] >= b.size[1]:
            a, z, width = (cx - b.size[0] / 2, cy), (cx + b.size[0] / 2, cy), b.size[1]
        else:
            a, z, width = (cx, cy - b.size[1] / 2), (cx, cy + b.size[1] / 2), b.size[0]
        edges.append((_id(b.id, used), node(*a), node(*z), width, b))

    nod = ['<?xml version="1.0" encoding="UTF-8"?>', "<nodes>"]
    for nid, x, y in nodes.values():
        nod.append(f'    <node id="{nid}" x="{fmt(x)}" y="{fmt(y)}"/>')
    nod += ["</nodes>", ""]
    edg = ['<?xml version="1.0" encoding="UTF-8"?>', "<edges>"]
    for eid, f, t, width, b in edges:
        edg += [f"    <edge id={quoteattr(eid)} from={quoteattr(f)} to={quoteattr(t)}>",
                f'        <param key="worldir_width_m" value="{fmt(width)}"/>',
                f'        <param key="worldir_entity" value={quoteattr(b.id)}/>', "    </edge>"]
    edg += ["</edges>", ""]
    content = json.dumps({"network.nod.xml": "\n".join(nod), "network.edg.xml": "\n".join(edg)},
                         sort_keys=True, indent=1)
    return content, make_report("sumo", world, content, boxes, skipped, reasons)


def write_sumo_files(content: str, directory) -> list:
    """Write the files of an export_to_sumo_with_report() result into ``directory``; returns their paths."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in json.loads(content).items():
        p = out / name
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths
