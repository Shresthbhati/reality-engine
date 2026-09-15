"""Dense MVS output parsing (P6-01 remainder).

The CUDA COLMAP dense pipeline (feature_extractor -> mapper ->
patch_match_stereo -> stereo_fusion) produces ``fused.ply``: a binary
little-endian, vertex-only point cloud. This module parses that output
into the repo's canonical `ReconstructedPoint` so the dense path lands
in the same types the rest of the reconstruction stack consumes --
closing the "fused.ply exists on disk but nothing in Python can read
it" gap recorded in the P6-01 ledger.

Honesty rules (same as MeshData.from_ply_bytes, which this deliberately
sits beside rather than duplicates semantics of):
- only binary_little_endian vertex PLY is supported; anything else
  raises a recorded DenseOutputError (never a half-parsed cloud);
- a PLY carrying faces is rejected: fused output is a cloud, and a
  mesh fed here means the caller grabbed the wrong artifact;
- vertex properties beyond position (e.g. uchar colors) are real data:
  the parse records their presence in the facts dict instead of
  silently dropping them;
- fused points have no SfM tracks -- track ids are "dense:<index>",
  a truthful label, not a borrowed sparse concept.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

from reconstruction.backend.interface import ReconstructedPoint


class DenseOutputError(ValueError):
    """A dense-output artifact could not be parsed. Recorded by the
    caller; never swallowed into a partial result."""


_PLY_SIZES = {"float": 4, "double": 8, "uchar": 1, "char": 1,
              "ushort": 2, "short": 2, "uint": 4, "int": 4}


def _parse_header(data: bytes) -> Tuple[List[str], int, bytes]:
    text, sep, body = data.partition(b"end_header\n")
    if not sep:
        raise DenseOutputError("PLY header has no 'end_header' terminator")
    header = text.decode("ascii", errors="replace").splitlines()
    fmt = next((line for line in header if line.startswith("format")), "")
    if "binary_little_endian" not in fmt:
        raise DenseOutputError(
            f"only binary_little_endian PLY supported, got {fmt!r}"
        )
    nv = 0
    in_vertex = False
    for line in header:
        if line.startswith("element"):
            parts = line.split()
            if parts[1] == "vertex":
                in_vertex = True
                nv = int(parts[2])
            elif parts[1] == "face":
                raise DenseOutputError(
                    "PLY contains faces -- this is a meshed artifact, not a "
                    "fused point cloud; use MeshData.from_ply_bytes for meshes"
                )
            else:
                in_vertex = False
            continue
    return header, nv, body


def parse_fused_ply(
    data: bytes,
    *,
    source_evidence_ids: Optional[List[str]] = None,
    return_facts: bool = False,
):
    """Parse COLMAP fused.ply into canonical ReconstructedPoints.

    `source_evidence_ids` is the provenance for the whole cloud (the
    images the dense run consumed); per-point track ids are
    "dense:<index>". With `return_facts=True`, returns
    (points, facts) where `facts` records what the artifact contained
    beyond position -- had_colors, n_points, n_properties -- so
    present-but-unused data is a recorded fact, not silent loss.
    """
    header, nv, body = _parse_header(data)

    vprops: List[Tuple[str, str]] = []
    in_vertex = False
    for line in header:
        if line.startswith("element"):
            in_vertex = line.split()[1] == "vertex"
            continue
        if in_vertex and line.startswith("property"):
            parts = line.split()
            if len(parts) != 3 or parts[1] == "list":
                raise DenseOutputError(f"unsupported vertex property: {line!r}")
            vprops.append((parts[1], parts[2]))

    names = {name for _, name in vprops}
    if not {"x", "y", "z"} <= names:
        raise DenseOutputError(f"PLY vertex element lacks xyz: {names}")

    stride = sum(_PLY_SIZES[t] for t, _ in vprops)
    expected = nv * stride
    if len(body) < expected:
        raise DenseOutputError(
            f"PLY body truncated: {len(body)} < {expected} bytes"
        )

    offsets = {}
    off = 0
    for ptype, pname in vprops:
        offsets[pname] = (ptype, off)
        off += _PLY_SIZES[ptype]

    evidence = source_evidence_ids or []
    points: List[ReconstructedPoint] = []
    for i in range(nv):
        base = i * stride
        chunk = body[base:base + stride]

        def read(name: str) -> float:
            ptype, poff = offsets[name]
            size = _PLY_SIZES[ptype]
            fmt_char = {"float": "f", "double": "d", "uchar": "B", "char": "b",
                        "ushort": "H", "short": "h", "uint": "I", "int": "i"}[ptype]
            return struct.unpack_from("<" + fmt_char, chunk, poff)[0]

        points.append(ReconstructedPoint(
            position=(read("x"), read("y"), read("z")),
            track_id=f"dense:{i}",
            source_evidence_ids=list(evidence),
        ))

    if return_facts:
        facts: Dict[str, object] = {
            "n_points": nv,
            "n_properties": len(vprops),
            "had_colors": bool({"red", "green", "blue"} <= names),
            "had_normals": bool({"nx", "ny", "nz"} <= names),
        }
        return points, facts
    return points
