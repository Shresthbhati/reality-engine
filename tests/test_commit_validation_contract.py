"""Correction commits must not be able to persist malformed or provenance-destroying data (release gate, Phase 5).

Found by probing the real route: `transform` accepted NaN, arbitrary dicts and string coordinates (a literal NaN in the
stored JSON makes /worldir unparseable in the browser), and `custom_properties` replaced wholesale, erasing the
pipeline-owned `reconstruction` trace from a RECONSTRUCTED entity.
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws

    ws._store = None
    with TestClient(main_mod.app) as c:
        sid = c.post("/api/sessions", json={"name": "C"}).json()["id"]
        wid = c.post("/api/worlds", json={"name": "W"}).json()["id"]
        assert c.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
        for i in range(3):
            r = c.post(f"/api/uploads?session_id={sid}",
                       files={"file": (f"f{i}.jpg", b"\xff\xd8\xff\xe0" + str(i).encode() * 8, "image/jpeg")})
            assert r.status_code == 201
        jid = c.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
        for _ in range(400):
            job = c.get(f"/api/jobs/{jid}").json()
            if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
                break
            time.sleep(0.25)
        assert job["status"] == "succeeded", job
        w = c.get(f"/api/worlds/{wid}/worldir").json()
        eid = next(e for e, x in w["entities"].items() if "reconstruction" in (x.get("custom_properties") or {}))
        yield c, wid, eid


def _commit(c, wid, body, *, raw=False):
    if raw:
        return c.post(f"/api/worlds/{wid}/commit", content=body, headers={"content-type": "application/json"})
    return c.post(f"/api/worlds/{wid}/commit", json=body)


def _head(c, wid):
    return c.get(f"/api/worlds/{wid}").json()["current_version_id"]


GOOD_TRANSFORM = {"position": {"x": 1.0, "y": 2.0, "z": 3.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


def test_a_well_formed_transform_is_accepted(world):
    c, wid, eid = world
    r = _commit(c, wid, {"entity_id": eid, "changes": {"transform": GOOD_TRANSFORM}})
    assert r.status_code == 200, r.text
    got = c.get(f"/api/worlds/{wid}/worldir").json()["entities"][eid]["transform"]
    assert got["position"] == GOOD_TRANSFORM["position"]


@pytest.mark.parametrize(
    "transform",
    [
        {"foo": 1},                                                                        # not a transform
        {"position": {"x": "a", "y": 0, "z": 0}, "rotation": {"w": 1, "x": 0, "y": 0, "z": 0}},   # string coordinate
        {"position": {"x": 0, "y": 0}, "rotation": {"w": 1, "x": 0, "y": 0, "z": 0}},     # missing z
        {"position": {"x": True, "y": 0, "z": 0}, "rotation": {"w": 1, "x": 0, "y": 0, "z": 0}},  # bool is not a number
        {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"w": 0, "x": 0, "y": 0, "z": 0}},    # zero quaternion
        {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"w": 5, "x": 0, "y": 0, "z": 0}},    # not unit length
        {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"w": 1, "x": 0, "y": 0, "z": 0}, "shear": 1},  # unknown key
    ],
    ids=["not-a-transform", "string-coordinate", "missing-z", "bool-coordinate", "zero-quaternion", "non-unit-quaternion",
         "unknown-key"],
)
def test_a_malformed_transform_is_refused_and_head_stays(world, transform):
    c, wid, eid = world
    before = _head(c, wid)
    r = _commit(c, wid, {"entity_id": eid, "changes": {"transform": transform}})
    assert r.status_code == 422, r.text
    assert _head(c, wid) == before


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_refused_anywhere_in_a_correction(world, token):
    c, wid, eid = world
    before = _head(c, wid)
    body = ('{"entity_id": "%s", "changes": {"transform": {"position": {"x": %s, "y": 0, "z": 0}, '
            '"rotation": {"w": 1, "x": 0, "y": 0, "z": 0}}}}') % (eid, token)
    assert _commit(c, wid, body, raw=True).status_code == 422
    body = '{"entity_id": "%s", "changes": {"custom_properties": {"note": {"deep": [1, %s]}}}}' % (eid, token)
    assert _commit(c, wid, body, raw=True).status_code == 422
    assert _head(c, wid) == before
    # and the stored world stays valid JSON a browser can parse
    json.loads(c.get(f"/api/worlds/{wid}/worldir").text, parse_constant=lambda t: pytest.fail(f"stored {t}"))


def test_custom_properties_cannot_erase_the_reconstruction_trace(world):
    c, wid, eid = world
    trace = c.get(f"/api/worlds/{wid}/worldir").json()["entities"][eid]["custom_properties"]["reconstruction"]
    # replacing the properties keeps the pipeline-owned trace
    r = _commit(c, wid, {"entity_id": eid, "changes": {"custom_properties": {"note": "checked on site"}}})
    assert r.status_code == 200, r.text
    props = c.get(f"/api/worlds/{wid}/worldir").json()["entities"][eid]["custom_properties"]
    assert props["note"] == "checked on site" and props["reconstruction"] == trace
    # rewriting the trace itself is refused
    before = _head(c, wid)
    r = _commit(c, wid, {"entity_id": eid, "changes": {"custom_properties": {"reconstruction": {"session_id": "forged"}}}})
    assert r.status_code == 422, r.text
    assert _head(c, wid) == before
