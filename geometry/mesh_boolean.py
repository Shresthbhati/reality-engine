"""Watertightness check and boolean operations on triangle meshes through manifold3d (Apache-2.0).

    facts = manifold_status(vertices, faces)             # {backend, manifold, status}: is this mesh a valid closed 2-manifold?
    v, f  = boolean("difference", (va, fa), (vb, fb))    # union | difference | intersection, result is manifold by construction

A Poisson surface from sparse points is typically open (trimmed), so ``manifold`` is usually False there and the pipeline records it
as a fact rather than pretending otherwise. Without manifold3d installed the status is ``unavailable`` -- never a guessed True.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

_OPS = ("union", "difference", "intersection")


def _manifold(vertices, faces):
    import manifold3d as m3

    mesh = m3.Mesh(vert_properties=np.asarray(vertices, dtype=np.float32), tri_verts=np.asarray(faces, dtype=np.uint32))
    return m3.Manifold(mesh)


def manifold_status(vertices, faces) -> dict:
    """``{backend, manifold, status}``. ``manifold`` is True only when manifold3d accepts the mesh as a closed, consistently
    oriented 2-manifold; None (with ``status`` ``unavailable``) when the library is missing."""
    try:
        import manifold3d  # noqa: F401
    except ImportError:
        return {"backend": "none", "manifold": None, "status": "unavailable", "note": "pip install manifold3d"}
    v, f = np.asarray(vertices), np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        return {"backend": "manifold3d", "manifold": False, "status": "NotATriangleMesh"}
    status = str(_manifold(v, f).status()).split(".")[-1]
    return {"backend": "manifold3d", "manifold": status == "NoError", "status": status}


def boolean(op: str, a: Tuple, b: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)`` of ``a <op> b`` for two manifold triangle meshes. Raises ValueError when either input is not a valid
    manifold (the library would otherwise return an empty or wrong shape), ImportError when manifold3d is absent."""
    if op not in _OPS:
        raise ValueError(f"op must be one of {_OPS}, got {op!r}")
    ma, mb = _manifold(*a), _manifold(*b)
    for name, m in (("first", ma), ("second", mb)):
        if str(m.status()).split(".")[-1] != "NoError":
            raise ValueError(f"the {name} mesh is not a valid manifold ({m.status()})")
    import manifold3d as m3

    kind = {"union": m3.OpType.Add, "difference": m3.OpType.Subtract, "intersection": m3.OpType.Intersect}[op]
    out = m3.Manifold.batch_boolean([ma, mb], kind).to_mesh()
    return np.asarray(out.vert_properties, dtype=float)[:, :3], np.asarray(out.tri_verts, dtype=int)
