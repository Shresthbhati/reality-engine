"""Godot game-export profile ("godot"): a GLB + manifest.json + scene.tscn bundle.

Frame: the canonical WorldIR frame is already Y-up, right-handed, so the GLB carries coordinates UNCHANGED (glTF and
Godot are Y-up right-handed too; Godot's importer maps glTF's +Z front to its own -Z forward). Procedural worlds emit
north as -Z, so north is Godot "forward". No axis swap is applied (exporters/godot/exporter.py's tscn writer assumes
Z-up input and is a different, older format; it is untouched).

Geometry: one shared unit cube (the cube of exporters/gltf/exporter.py) scaled to each entity's REAL bounds -- the
plain gltf exporter ignores bounds and skips transform-less entities, which is every procedural entity. A node's
translation is the bounds centre (+ transform position when the entity has one), scale is the extents.

Collision: Godot's import hint is a "-col" name suffix (StaticBody3D + CollisionShape3D generated on import). It is
added for solid structural kinds only (wall/floor/ceiling/roof/column/beam); doors, windows, stairs and the rest stay
visual-only and the manifest says so.

Units: "metre" only when the world declares metric units (metadata["units"]) or every exported entity is GENERATED
(procedural worlds are authored in metres). Otherwise the manifest says "model-unit" with scale_to_metres = null;
no scale is ever invented. Godot itself is NOT available on the build machine: loading is UNVERIFIED, structure is
validated by tests/test_godot_profile.py.
"""

from __future__ import annotations

import json
import struct
from typing import Dict, Tuple

from exporters.boxes import MIN_DIMENSION_M, collect_boxes
from exporters.gltf.exporter import _CUBE_INDICES, _CUBE_POSITIONS
from exporters.report import ExportReport, content_hash

COLLISION_KINDS = frozenset({"wall", "floor", "ceiling", "roof", "column", "beam"})
_METRIC = {"m", "meter", "meters", "metre", "metres"}
_GLB_MAGIC, _GLB_VERSION, _JSON_CHUNK, _BIN_CHUNK = 0x46546C67, 2, 0x4E4F534A, 0x004E4942


def _unit(world, boxes) -> Tuple[str, str]:
    declared = str((world.metadata or {}).get("units", "")).lower()
    if declared in _METRIC:
        return "metre", "world.metadata['units'] declares metres"
    if boxes and all(b.provenance.upper() == "GENERATED" for b in boxes) and not declared:
        return "metre", "every exported entity is GENERATED (procedural worlds are authored in metres)"
    return "model-unit", "world scale is not declared metric; no conversion was applied or invented"


def _glb(gltf: dict, blob: bytes) -> bytes:
    j = json.dumps(gltf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    j += b" " * (-len(j) % 4)
    blob += b"\x00" * (-len(blob) % 4)
    total = 12 + 8 + len(j) + 8 + len(blob)
    return (struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total)
            + struct.pack("<II", len(j), _JSON_CHUNK) + j + struct.pack("<II", len(blob), _BIN_CHUNK) + blob)


def export_godot_profile(world, artifact_store=None) -> Tuple[Dict[str, bytes], ExportReport]:
    """(files {name: bytes}, ExportReport). Files: <world>.glb, manifest.json, scene.tscn."""
    boxes, skipped, reasons = collect_boxes(world, require_position=False)
    unit, unit_basis = _unit(world, boxes)

    pos = b"".join(struct.pack("<fff", *v) for v in _CUBE_POSITIONS)
    idx = b"".join(struct.pack("<H", i) for i in _CUBE_INDICES)
    lo = [min(v[a] for v in _CUBE_POSITIONS) for a in range(3)]
    hi = [max(v[a] for v in _CUBE_POSITIONS) for a in range(3)]
    nodes, entries = [], []
    for b in boxes:
        solid = b.type in COLLISION_KINDS
        name = f"{b.id}-col" if solid else b.id
        nodes.append({"name": name, "mesh": 0, "translation": list(b.center), "scale": list(b.size),
                      "extras": {"entity_id": b.id, "kind": b.type, "provenance": b.provenance}})
        entries.append({"id": b.id, "name": b.name, "kind": b.type, "provenance": b.provenance,
                        "confidence": b.confidence, "node": name, "collision": solid,
                        "bounds_min": list(b.min), "bounds_max": list(b.max),
                        # only the first BOX/PLANE geometry (the entity envelope) is exported; the rest is dropped
                        "geometries_dropped": max(0, len(world.entities[b.id].geometry_ids) - 1)})
    gltf = {
        "asset": {"version": "2.0", "generator": "reality-engine godot profile"},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(_CUBE_POSITIONS), "type": "VEC3",
             "min": lo, "max": hi},
            {"bufferView": 1, "componentType": 5123, "count": len(_CUBE_INDICES), "type": "SCALAR"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos), "target": 34962},
                        {"buffer": 0, "byteOffset": len(pos), "byteLength": len(idx), "target": 34963}],
        "buffers": [{"byteLength": len(pos) + len(idx)}],
    }
    glb_name = "world.glb"
    glb = _glb(gltf, pos + idx)
    manifest = {
        "profile": "godot", "world_id": world.id, "world_version": getattr(world, "version", 0) or 0,
        "glb": glb_name,
        "frame": {"up": "+Y", "forward": "-Z", "handedness": "right", "axis_remap": "none (WorldIR is Y-up)"},
        "unit": unit, "unit_basis": unit_basis,
        "scale_to_metres": 1.0 if unit == "metre" else None,
        "collision": {"hint": "-col node-name suffix (Godot import)", "kinds": sorted(COLLISION_KINDS)},
        "min_dimension_floor": MIN_DIMENSION_M,
        "entities": entries,
        "skipped": [{"id": i, "reason": r} for i, r in zip(skipped, reasons)],
        "unverified": "not opened in Godot (not available on the build machine)",
    }
    tscn = "\n".join([
        '[gd_scene load_steps=2 format=3]', '',
        f'[ext_resource type="PackedScene" path="res://{glb_name}" id="1"]', '',
        f'[node name="World" type="Node3D"]',
        f'metadata/unit = "{unit}"', '',
        f'[node name="Level" parent="." instance=ExtResource("1")]', ''])
    files = {glb_name: glb, "manifest.json": json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
             "scene.tscn": tscn.encode("utf-8")}
    digest = content_hash(b"".join(files[k] for k in sorted(files)))
    return files, ExportReport(
        format="godot", world_id=world.id, world_version=manifest["world_version"],
        entities_exported=tuple(b.id for b in boxes), entities_skipped=tuple(skipped),
        skip_reasons=tuple(reasons), content_hash=digest)
