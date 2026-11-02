"""3D Tiles 1.1 export profile ("3dtiles"): tileset.json + one GLB per XY grid cell.

Scope (honest): box-based, like the other box compilers (exporters/boxes.py). Only entities with a position and a
BOX/PLANE geometry with REAL bounds become content; everything else is skipped WITH A REASON. One shared unit cube is
scaled per entity (the Godot profile's trick), so no geometry is invented.

Frames:
  - WorldIR boxes are Z-up local metres (exporters/boxes.py). 3D Tiles glTF content is Y-up and the runtime rotates it
    to Z-up, so each node is written as (x, y, z) -> (x, z, -y). Tile bounding volumes are in the tile's Z-up frame.
  - Without ``origin_lonlat`` NO ``transform`` is written: a 3D Tiles viewer treats an identity root as ECEF, so the
    tileset is NOT placeable on a globe, and ``asset.extras.worldir.georeferenced`` says so. With it, the root gets an
    ENU->ECEF matrix at that WGS84 point, ASSUMING local +x = east, +y = north, +z = up (only the caller knows that).

Structure: a root tile without content (refine ADD) whose children are the grid cells, geometricError = 0 for the
leaves. There is a single level of detail: no simplification is performed, so no screen-space-error ladder is claimed.
Validated against the official 3d-tiles JSON schema in tests; NOT loaded in CesiumJS (not available on this machine).
"""

from __future__ import annotations

import json
import math
import struct
from typing import Dict, List, Optional, Tuple

from exporters.boxes import MIN_DIMENSION_M, collect_boxes
from exporters.godot.profile import _glb
from exporters.gltf.exporter import _CUBE_INDICES, _CUBE_POSITIONS
from exporters.report import ExportReport, content_hash

_WGS84_A = 6378137.0
_WGS84_E2 = 0.00669437999014


def enu_to_ecef_matrix(lon_deg: float, lat_deg: float, height_m: float = 0.0) -> List[float]:
    """Column-major 4x4 (3D Tiles `transform`) taking local east/north/up metres to ECEF."""
    lon, lat = math.radians(lon_deg), math.radians(lat_deg)
    sl, cl, sp, cp = math.sin(lon), math.cos(lon), math.sin(lat), math.cos(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sp * sp)
    x, y, z = (n + height_m) * cp * cl, (n + height_m) * cp * sl, (n * (1.0 - _WGS84_E2) + height_m) * sp
    return [-sl, cl, 0.0, 0.0,
            -sp * cl, -sp * sl, cp, 0.0,
            cp * cl, cp * sl, sp, 0.0,
            x, y, z, 1.0]


def _bbox(boxes) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    lo = tuple(min(b.min[a] for b in boxes) for a in range(3))
    hi = tuple(max(b.max[a] for b in boxes) for a in range(3))
    return lo, hi


def _volume(lo, hi) -> List[float]:
    c = [(l + h) / 2.0 for l, h in zip(lo, hi)]
    h = [(hh - l) / 2.0 for l, hh in zip(lo, hi)]
    return [c[0], c[1], c[2], h[0], 0, 0, 0, h[1], 0, 0, 0, h[2]]


def _tile_glb(boxes) -> bytes:
    pos = b"".join(struct.pack("<fff", *v) for v in _CUBE_POSITIONS)
    idx = b"".join(struct.pack("<H", i) for i in _CUBE_INDICES)
    nodes = [{"name": b.id, "mesh": 0, "translation": [b.center[0], b.center[2], -b.center[1]],
              "scale": [b.size[0], b.size[2], b.size[1]],
              "extras": {"entity_id": b.id, "kind": b.type, "provenance": b.provenance, "confidence": b.confidence}}
             for b in boxes]
    gltf = {
        "asset": {"version": "2.0", "generator": "reality-engine 3dtiles profile"},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(_CUBE_POSITIONS), "type": "VEC3",
             "min": [min(v[a] for v in _CUBE_POSITIONS) for a in range(3)],
             "max": [max(v[a] for v in _CUBE_POSITIONS) for a in range(3)]},
            {"bufferView": 1, "componentType": 5123, "count": len(_CUBE_INDICES), "type": "SCALAR"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos), "target": 34962},
                        {"buffer": 0, "byteOffset": len(pos), "byteLength": len(idx), "target": 34963}],
        "buffers": [{"byteLength": len(pos) + len(idx)}],
    }
    return _glb(gltf, pos + idx)


def export_3dtiles_profile(
    world, artifact_store=None, tile_size_m: float = 50.0,
    origin_lonlat: Optional[Tuple[float, float]] = None, origin_height_m: float = 0.0,
) -> Tuple[Dict[str, bytes], ExportReport]:
    """(files {path: bytes}, ExportReport). Files: tileset.json and tiles/<ix>_<iy>.glb."""
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be > 0")
    if origin_lonlat is not None and not (-180.0 <= origin_lonlat[0] <= 180.0 and -90.0 <= origin_lonlat[1] <= 90.0):
        raise ValueError(f"origin_lonlat {origin_lonlat!r} is not a valid (lon, lat)")
    boxes, skipped, reasons = collect_boxes(world, require_position=False)

    cells: Dict[Tuple[int, int], list] = {}
    for b in boxes:
        cells.setdefault((math.floor(b.center[0] / tile_size_m), math.floor(b.center[1] / tile_size_m)), []).append(b)

    files: Dict[str, bytes] = {}
    children = []
    for key in sorted(cells):
        members = cells[key]
        uri = f"tiles/{key[0]}_{key[1]}.glb"
        files[uri] = _tile_glb(members)
        lo, hi = _bbox(members)
        children.append({"boundingVolume": {"box": _volume(lo, hi)}, "geometricError": 0,
                         "content": {"uri": uri}})

    lo, hi = _bbox(boxes) if boxes else ((0.0,) * 3, (0.0,) * 3)
    diag = math.dist(lo, hi)
    root = {"boundingVolume": {"box": _volume(lo, hi)}, "geometricError": diag, "refine": "ADD"}
    if origin_lonlat is not None:
        root["transform"] = enu_to_ecef_matrix(origin_lonlat[0], origin_lonlat[1], origin_height_m)
    if children:
        root["children"] = children
    georef = ({"georeferenced": True, "origin_lonlat": list(origin_lonlat), "origin_height_m": origin_height_m,
               "assumption": "local +x = east, +y = north, +z = up; WGS84 ellipsoidal height"}
              if origin_lonlat is not None else
              {"georeferenced": False,
               "note": "local metres, Z-up; no root transform, so a 3D Tiles viewer would treat it as ECEF"})
    tileset = {
        "asset": {"version": "1.1", "generator": "reality-engine 3dtiles profile",
                  "extras": {"worldir": {"world_id": world.id, "world_version": getattr(world, "version", 0) or 0,
                                         "tile_size_m": tile_size_m, "min_dimension_floor": MIN_DIMENSION_M,
                                         "levels_of_detail": 1, **georef,
                                         "skipped": [{"id": i, "reason": r} for i, r in zip(skipped, reasons)]}}},
        "geometricError": diag, "root": root,
    }
    files["tileset.json"] = json.dumps(tileset, indent=2, sort_keys=True).encode("utf-8")
    digest = content_hash(b"".join(files[k] for k in sorted(files)))
    return files, ExportReport(
        format="3dtiles", world_id=world.id, world_version=getattr(world, "version", 0) or 0,
        entities_exported=tuple(b.id for b in boxes), entities_skipped=tuple(skipped),
        skip_reasons=tuple(reasons), content_hash=digest)
