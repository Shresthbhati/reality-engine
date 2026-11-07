"""Structural validation of a binary glTF (GLB) 2.0 file, in pure Python.

Not a replacement for the Khronos glTF-Validator (run that for release evidence); this catches the failures that make a
file unusable in Blender/Unreal/Unity: a broken container, buffers/views/accessors that run past their data, node/mesh
references that point nowhere, and POSITION data that is empty, non-finite or outside its declared min/max.
"""

from __future__ import annotations

import json
import math
import struct

_COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class GlbError(AssertionError):
    pass


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise GlbError(msg)


def parse_glb(data: bytes) -> tuple[dict, bytes]:
    """Container check -> (json, bin chunk). Raises GlbError on any structural problem."""
    _need(len(data) >= 20, "file shorter than a GLB header")
    magic, version, total = struct.unpack("<4sII", data[:12])
    _need(magic == b"glTF", f"bad magic {magic!r}")
    _need(version == 2, f"glTF version {version}, expected 2")
    _need(total == len(data), f"header length {total} != file length {len(data)}")
    chunks, off = [], 12
    while off < len(data):
        _need(off + 8 <= len(data), "truncated chunk header")
        length, ctype = struct.unpack("<II", data[off:off + 8])
        _need(length % 4 == 0, "chunk length not 4-byte aligned")
        _need(off + 8 + length <= len(data), "chunk runs past end of file")
        chunks.append((ctype, data[off + 8:off + 8 + length]))
        off += 8 + length
    _need(chunks and chunks[0][0] == 0x4E4F534A, "first chunk is not JSON")
    doc = json.loads(chunks[0][1].rstrip(b" \x00").decode("utf-8"))
    binary = next((c for t, c in chunks[1:] if t == 0x004E4942), b"")
    return doc, binary


def validate_glb(data: bytes) -> dict:
    """Validate and return a small summary {meshes, nodes, primitives, vertices, triangles}."""
    doc, binary = parse_glb(data)
    _need(doc.get("asset", {}).get("version") == "2.0", "asset.version must be '2.0'")

    buffers = doc.get("buffers", [])
    views = doc.get("bufferViews", [])
    accessors = doc.get("accessors", [])
    meshes = doc.get("meshes", [])
    nodes = doc.get("nodes", [])

    if buffers:
        _need("uri" not in buffers[0], "GLB buffer 0 must be the embedded BIN chunk (no uri)")
        _need(buffers[0]["byteLength"] <= len(binary), "buffer 0 byteLength exceeds the BIN chunk")
    for i, v in enumerate(views):
        _need(0 <= v["buffer"] < len(buffers), f"bufferView {i}: bad buffer index")
        _need(v.get("byteOffset", 0) + v["byteLength"] <= buffers[v["buffer"]]["byteLength"],
              f"bufferView {i} runs past its buffer")

    for i, a in enumerate(accessors):
        _need(a["componentType"] in _COMPONENT, f"accessor {i}: unknown componentType {a['componentType']}")
        _need(a["type"] in _TYPE_COUNT, f"accessor {i}: unknown type {a['type']}")
        _need(a["count"] > 0, f"accessor {i}: count 0")
        if "bufferView" not in a:
            continue
        _need(0 <= a["bufferView"] < len(views), f"accessor {i}: bad bufferView")
        v = views[a["bufferView"]]
        _, size = _COMPONENT[a["componentType"]]
        stride = v.get("byteStride") or size * _TYPE_COUNT[a["type"]]
        need = a.get("byteOffset", 0) + stride * (a["count"] - 1) + size * _TYPE_COUNT[a["type"]]
        _need(need <= v["byteLength"], f"accessor {i} runs past its bufferView ({need} > {v['byteLength']})")

    vertices = triangles = primitives = 0
    for mi, m in enumerate(meshes):
        _need(m.get("primitives"), f"mesh {mi}: no primitives")
        for pi, p in enumerate(m["primitives"]):
            primitives += 1
            pos = p.get("attributes", {}).get("POSITION")
            _need(pos is not None and 0 <= pos < len(accessors), f"mesh {mi} prim {pi}: missing/invalid POSITION")
            a = accessors[pos]
            _need(a["type"] == "VEC3" and a["componentType"] == 5126, f"mesh {mi} prim {pi}: POSITION must be float VEC3")
            _need("min" in a and "max" in a, f"mesh {mi} prim {pi}: POSITION needs min/max (required by the spec)")
            v = views[a["bufferView"]]
            start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
            stride = v.get("byteStride") or 12
            for k in range(a["count"]):
                xyz = struct.unpack_from("<3f", binary, start + k * stride)
                _need(all(math.isfinite(c) for c in xyz), f"mesh {mi} prim {pi}: non-finite POSITION at {k}")
                _need(all(a["min"][j] - 1e-4 <= xyz[j] <= a["max"][j] + 1e-4 for j in range(3)),
                      f"mesh {mi} prim {pi}: POSITION {k} outside declared min/max")
            vertices += a["count"]
            if "indices" in p:
                idx = accessors[p["indices"]]
                _need(idx["type"] == "SCALAR", f"mesh {mi} prim {pi}: indices must be SCALAR")
                if p.get("mode", 4) == 4:
                    _need(idx["count"] % 3 == 0, f"mesh {mi} prim {pi}: triangle index count not a multiple of 3")
                    triangles += idx["count"] // 3
                iv = views[idx["bufferView"]]
                ifmt, isz = _COMPONENT[idx["componentType"]]
                ibase = iv.get("byteOffset", 0) + idx.get("byteOffset", 0)
                for k in range(idx["count"]):
                    (val,) = struct.unpack_from("<" + ifmt, binary, ibase + k * isz)
                    _need(val < a["count"], f"mesh {mi} prim {pi}: index {val} >= vertex count {a['count']}")

    for ni, n in enumerate(nodes):
        if "mesh" in n:
            _need(0 <= n["mesh"] < len(meshes), f"node {ni}: bad mesh index")
        for c in n.get("children", []):
            _need(0 <= c < len(nodes), f"node {ni}: bad child index")
    for s in doc.get("scenes", []):
        for n in s.get("nodes", []):
            _need(0 <= n < len(nodes), "scene references a missing node")

    return {"meshes": len(meshes), "nodes": len(nodes), "primitives": primitives,
            "vertices": vertices, "triangles": triangles}
