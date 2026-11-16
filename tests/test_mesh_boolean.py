"""manifold3d: an honest watertightness verdict and boolean operations whose result volumes can be checked by hand."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("manifold3d")
from geometry.mesh_boolean import boolean, manifold_status  # noqa: E402


def _box(x0, y0, z0, x1, y1, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
                  [3, 0, 4], [3, 4, 7]])
    return v, f


def _volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def test_a_closed_box_is_manifold():
    f = manifold_status(*_box(0, 0, 0, 1, 1, 1))
    assert f == {"backend": "manifold3d", "manifold": True, "status": "NoError"}


def test_an_open_surface_is_reported_as_not_manifold_not_guessed():
    v, faces = _box(0, 0, 0, 1, 1, 1)
    f = manifold_status(v, faces[:-2])                   # drop two triangles: a hole
    assert f["manifold"] is False and f["status"] != "NoError"


def test_garbage_is_not_a_triangle_mesh():
    assert manifold_status(np.zeros((4, 2)), np.zeros((0, 3), int))["status"] == "NotATriangleMesh"


def test_difference_carves_a_known_volume():
    wall = _box(0, 0, 0, 4, 0.2, 3)                      # a 4 x 0.2 x 3 wall = 2.4 m3
    window = _box(1, -0.1, 1, 2, 0.3, 2)                 # a 1 x 1 opening cut clean through it: removes 1 x 0.2 x 1 = 0.2 m3
    v, f = boolean("difference", wall, window)
    assert abs(_volume(v, f) - 2.2) < 1e-4
    assert manifold_status(v, f)["manifold"] is True


def test_union_and_intersection_volumes():
    a, b = _box(0, 0, 0, 2, 2, 2), _box(1, 1, 1, 3, 3, 3)      # overlap is a 1 m cube
    assert abs(_volume(*boolean("intersection", a, b)) - 1.0) < 1e-4
    assert abs(_volume(*boolean("union", a, b)) - 15.0) < 1e-4


def test_an_invalid_operand_is_refused():
    v, faces = _box(0, 0, 0, 1, 1, 1)
    with pytest.raises(ValueError, match="not a valid manifold"):
        boolean("union", (v, faces[:-2]), _box(0, 0, 0, 1, 1, 1))
    with pytest.raises(ValueError, match="op must be"):
        boolean("xor", _box(0, 0, 0, 1, 1, 1), _box(0, 0, 0, 1, 1, 1))
