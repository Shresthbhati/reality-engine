"""Saved measurements: pure math (world_ir.measure) and the /measurements routes.

Honesty contract under test: the server recomputes the value from the points and
rejects a disagreeing client value; an unknown scale is never stored as metres;
an uncertainty exists only when supplied (authored geometry is exactly 0).
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest

from world_ir import measure
from world_ir.measure import MeasureError


# ---------------------------------------------------------------- pure math

def test_distance_height_and_elevation_difference():
    a, b = [0, 0, 0], [3, 4, 12]
    assert measure.compute("distance", [a, b]) == pytest.approx(13.0)
    assert measure.compute("height", [[0, 5, 0], [9, 2, 9]]) == pytest.approx(3.0)  # unsigned vertical delta
    assert measure.compute("elevation_difference", [[0, 5, 0], [9, 2, 9]]) == pytest.approx(-3.0)  # signed
    assert measure.compute("elevation_difference", [[0, 2, 0], [0, 5, 0]]) == pytest.approx(3.0)


def test_area_rectangle_on_floor_and_on_a_wall():
    floor = [[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]]
    assert measure.compute("area", floor) == pytest.approx(12.0)
    wall = [[0, 0, 0], [4, 0, 0], [4, 2.5, 0], [0, 2.5, 0]]
    assert measure.compute("area", wall) == pytest.approx(10.0)
    # winding order does not change the area
    assert measure.compute("area", list(reversed(floor))) == pytest.approx(12.0)


def test_area_tilted_plane_is_true_area_not_projection():
    # 3-4-5 slope: base 3 wide, slope length 5, depth 2 => 10, not the 6 a floor projection would give
    pts = [[0, 0, 0], [3, 4, 0], [3, 4, 2], [0, 0, 2]]
    assert measure.compute("area", pts) == pytest.approx(10.0)


def test_degenerate_and_collinear_polygons_rejected():
    with pytest.raises(MeasureError, match="degenerate"):
        measure.compute("area", [[0, 0, 0], [1, 0, 0], [2, 0, 0]])  # collinear
    with pytest.raises(MeasureError, match="degenerate"):
        measure.compute("area", [[1, 1, 1]] * 3)  # coincident
    with pytest.raises(MeasureError, match="at least 3"):
        measure.compute("area", [[0, 0, 0], [1, 0, 0]])


def test_non_planar_and_self_intersecting_polygons_rejected():
    with pytest.raises(MeasureError, match="not planar"):
        measure.compute("area", [[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 2, 3]])
    lopsided_bowtie = [[0, 0, 0], [3, 0, 3], [3, 0, 0], [0, 0, 1]]
    with pytest.raises(MeasureError, match="crosses itself"):
        measure.compute("area", lopsided_bowtie)
    symmetric_bowtie = [[0, 0, 0], [2, 0, 2], [2, 0, 0], [0, 0, 2]]  # signed areas cancel -> zero
    with pytest.raises(MeasureError, match="degenerate"):
        measure.compute("area", symmetric_bowtie)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinates_rejected(bad):
    with pytest.raises(MeasureError, match="non-finite"):
        measure.compute("distance", [[0, 0, 0], [bad, 0, 0]])


def test_bad_shapes_and_types_rejected():
    for bad in ([[0, 0, 0]], [[0, 0, 0], [1, 1]], [[0, 0, 0], ["1", 0, 0]], [[0, 0, 0], [True, 0, 0]], "ab", None):
        with pytest.raises(MeasureError):
            measure.compute("distance", bad)
    with pytest.raises(MeasureError, match="unknown measurement kind"):
        measure.compute("volume", [[0, 0, 0], [1, 1, 1]])
    with pytest.raises(MeasureError, match="at most"):
        measure.compute("distance", [[0, 0, 0]] * 3)


def test_distance_of_identical_points_is_zero_not_an_error():
    assert measure.compute("distance", [[1, 2, 3], [1, 2, 3]]) == 0.0


def test_unit_follows_scale_state_and_unknown_is_never_metres():
    assert measure.unit_for("distance", "metric") == "m"
    assert measure.unit_for("area", "authored") == "m2"
    assert measure.unit_for("distance", "unknown") == "model_units"
    assert measure.unit_for("area", "unknown") == "model_units2"
    with pytest.raises(MeasureError):
        measure.unit_for("distance", "nominal")


# ------------------------------------------------------------------- routes

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


@pytest.fixture()
def world_id(client):
    return client.post("/api/worlds", json={"name": "Measured"}).json()["id"]


def _post(client, wid, **over):
    body = {
        "kind": "distance",
        "points": [[0, 0, 0], [3, 4, 0]],
        "scale_state": "metric",
        "method": "two picked points",
    }
    body.update(over)
    return client.post(f"/api/worlds/{wid}/measurements", json=body)


def test_create_list_delete_roundtrip(client, world_id):
    assert client.get(f"/api/worlds/{world_id}/measurements").json() == []
    r = _post(client, world_id)
    assert r.status_code == 201
    row = r.json()
    assert row["value"] == pytest.approx(5.0)
    assert row["unit"] == "m"
    assert row["uncertainty"] is None  # never fabricated
    assert row["scale_state"] == "metric"
    assert row["id"].startswith("msr_") and row["created_at"]

    listed = client.get(f"/api/worlds/{world_id}/measurements").json()
    assert [m["id"] for m in listed] == [row["id"]]

    d = client.delete(f"/api/worlds/{world_id}/measurements/{row['id']}")
    assert d.status_code == 200 and d.json()["deleted"] is True
    assert client.get(f"/api/worlds/{world_id}/measurements").json() == []
    assert client.delete(f"/api/worlds/{world_id}/measurements/{row['id']}").status_code == 404


def test_unknown_world_is_404(client):
    assert client.get("/api/worlds/wld_nope/measurements").status_code == 404
    assert _post(client, "wld_nope").status_code == 404
    assert client.delete("/api/worlds/wld_nope/measurements/msr_x").status_code == 404


def test_client_value_that_disagrees_is_rejected_and_matching_is_accepted(client, world_id):
    assert _post(client, world_id, value=5.5).status_code == 422
    ok = _post(client, world_id, value=5.0)
    assert ok.status_code == 201
    assert ok.json()["value"] == pytest.approx(5.0)


def test_value_is_always_the_server_value(client, world_id):
    r = _post(client, world_id, kind="area", points=[[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]], value=12.0 + 1e-12)
    assert r.status_code == 201
    assert r.json()["value"] == pytest.approx(12.0)
    assert r.json()["unit"] == "m2"


def test_unknown_scale_is_stored_as_model_units_never_metres(client, world_id):
    r = _post(client, world_id, scale_state="unknown")
    assert r.status_code == 201
    assert r.json()["unit"] == "model_units"
    assert r.json()["scale_state"] == "unknown"
    area = _post(client, world_id, kind="area", points=[[0, 0, 0], [1, 0, 0], [1, 0, 1]], scale_state="unknown")
    assert area.json()["unit"] == "model_units2"


def test_authored_geometry_is_exact_uncertainty_zero_even_if_client_lies(client, world_id):
    r = _post(client, world_id, scale_state="authored", uncertainty=0.5)
    assert r.status_code == 201
    assert r.json()["uncertainty"] == 0.0
    assert r.json()["unit"] == "m"


def test_supplied_uncertainty_is_kept_and_bad_ones_rejected(client, world_id):
    assert _post(client, world_id, uncertainty=0.02).json()["uncertainty"] == pytest.approx(0.02)
    assert _post(client, world_id, uncertainty=-1).status_code == 422


def test_invalid_input_rejected(client, world_id):
    assert _post(client, world_id, kind="volume").status_code == 422
    assert _post(client, world_id, scale_state="nominal").status_code == 422
    assert _post(client, world_id, points=[[0, 0, 0]]).status_code == 422
    assert _post(client, world_id, points=[[0, 0, 0], [1, 1, "x"]]).status_code == 422
    assert _post(client, world_id, method="   ").status_code == 422
    assert _post(client, world_id, method="x" * 201).status_code == 422
    assert _post(client, world_id, kind="area", points=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]).status_code == 422
    # NaN literal in the JSON body (Python's json accepts it): must not be stored
    r = client.post(
        f"/api/worlds/{world_id}/measurements",
        content='{"kind":"distance","points":[[0,0,0],[NaN,0,0]],"scale_state":"metric","method":"m"}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422
    assert client.get(f"/api/worlds/{world_id}/measurements").json() == []


def test_point_cap_is_413(client, world_id):
    pts = [[math.cos(i / 10), 0, math.sin(i / 10)] for i in range(measure.POINTS_MAX["area"] + 1)]
    assert _post(client, world_id, kind="area", points=pts).status_code == 413
