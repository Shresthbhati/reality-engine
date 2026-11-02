"""POST /api/reconstructions accepts the opt-in CPU plane-sweep request and the status carries its facts."""

from __future__ import annotations

import importlib
import io
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")
pytest.importorskip("PIL")
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _jpeg() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.effect_noise((640, 480), 40).convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


def _post(client, **form):
    return client.post("/api/reconstructions", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))], data=form)


def _job_payload(client, job_id):
    import asyncio
    from apps.api import db as db_mod
    from apps.api.models import Job

    async def go():
        async with db_mod.get_sessionmaker()() as s:
            return (await s.get(Job, job_id)).payload
    return asyncio.run(go())


def test_default_request_carries_no_dense_cpu_payload():
    from apps.api.jobs import _dense_cpu_options
    assert _dense_cpu_options(None) == {} and _dense_cpu_options({}) == {}
    assert _dense_cpu_options({"dense_cpu": {"enabled": False}}) == {}


def test_dense_cpu_payload_maps_to_pipeline_options():
    from apps.api.jobs import _dense_cpu_options
    from engine.pipeline.vertical_slice import VerticalSliceOptions
    kw = _dense_cpu_options({"dense_cpu": {"enabled": True, "max_views": 3, "stride": None, "planes": 64}})
    assert kw == {"plane_sweep_enabled": True, "plane_sweep_max_views": 3, "plane_sweep_planes": 64}
    o = VerticalSliceOptions(**kw)                      # every key is a real option; unset ones keep defaults
    assert o.plane_sweep_enabled and o.plane_sweep_max_views == 3 and o.plane_sweep_stride == 2


def test_post_without_dense_cpu_is_unchanged(client):
    r = _post(client)
    assert r.status_code == 202
    assert not (_job_payload(client, r.json()["job_id"]) or {}).get("dense_cpu")


def test_post_with_dense_cpu_records_the_request_on_the_job(client):
    r = _post(client, dense_cpu="true", plane_sweep_max_views="3", plane_sweep_planes="32")
    assert r.status_code == 202
    cfg = _job_payload(client, r.json()["job_id"])["dense_cpu"]
    assert cfg == {"enabled": True, "max_views": 3, "stride": None, "planes": 32}


@pytest.mark.parametrize("field,val", [("plane_sweep_max_views", "1"), ("plane_sweep_stride", "0"),
                                       ("plane_sweep_planes", "9999")])
def test_out_of_range_tuning_is_rejected(client, field, val):
    assert _post(client, dense_cpu="true", **{field: val}).status_code == 422


def test_status_surfaces_plane_sweep_and_device_depth_facts(client):
    import asyncio
    from apps.api import db as db_mod
    from apps.api.models import WorldVersion, new_id

    wid = _post(client).json()["world_id"]
    facts = {"plane_sweep": {"status": "skipped", "reason": "no scale"},
             "device_depth_scale": {"status": "refused", "reason": "spread"}}

    async def attach():
        from apps.api.models import World
        async with db_mod.get_sessionmaker()() as s:
            w = await s.get(World, wid)
            v = WorldVersion(id=new_id("ver"), world_id=wid, report={"stages": facts})
            s.add(v)
            await s.flush()
            w.current_version_id = v.id
            await s.commit()
    asyncio.run(attach())
    m = client.get(f"/api/worlds/{wid}/status").json()["model"]
    assert m["plane_sweep"] == facts["plane_sweep"] and m["device_depth_scale"] == facts["device_depth_scale"]
