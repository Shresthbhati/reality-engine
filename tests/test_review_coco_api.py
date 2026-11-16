"""GET /api/worlds/{id}/review/coco: the pipeline's recorded 2D detections as COCO, at the EXIF-ORIENTED size they were computed on."""
from __future__ import annotations

import asyncio
import importlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

DETECTIONS = {"ev1": [{"region_id": "seg-1", "label": "sofa", "confidence": 0.9, "bbox_px": [10.0, 20.0, 110.0, 120.0]}]}


@pytest.fixture()
def env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield SimpleNamespace(client=c, db=db_mod)


def _rotated_jpeg() -> bytes:
    """320 wide x 240 high pixels carrying EXIF orientation 6 (rotate 90 degrees): upright it is 240 x 320."""
    img = Image.new("RGB", (320, 240), (90, 120, 150))
    exif = img.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _seed(env, report, with_photo=True):
    from apps.api.models import Evidence, World, WorldVersion, new_id
    from apps.api.storage import store_bytes

    wid = env.client.post("/api/worlds", json={"name": "w"}).json()["id"]
    digest, _ = store_bytes(_rotated_jpeg())

    async def go():
        async with env.db.get_sessionmaker()() as s:
            v = WorldVersion(id=new_id("ver"), world_id=wid, report=report)
            s.add(v)
            s.add(Evidence(id="ev1", name="sofa-room.jpg", type="photo",
                           artifact_uri=f"sha256://{digest}" if with_photo else "sha256://" + "0" * 64))
            await s.flush()
            (await s.get(World, wid)).current_version_id = v.id
            await s.commit()
    asyncio.run(go())
    return wid


def test_detections_come_back_as_coco_at_the_oriented_size(env):
    wid = _seed(env, {"stages": {"perception": {"detections_2d": DETECTIONS}}})
    r = env.client.get(f"/api/worlds/{wid}/review/coco")
    assert r.status_code == 200, r.text
    coco = r.json()
    (im,) = coco["images"]
    assert im["file_name"] == "sofa-room.jpg" and im["evidence_id"] == "ev1"
    assert (im["width"], im["height"]) == (240, 320), "detections were computed on the upright image, so the size must be too"
    (a,) = coco["annotations"]
    assert a["bbox"] == [10.0, 20.0, 100.0, 100.0] and a["attributes"]["region_id"] == "seg-1"
    assert [c["name"] for c in coco["categories"]] == ["sofa"]


def test_a_version_without_detections_says_why(env):
    wid = _seed(env, {"stages": {"perception": {"status": "skipped"}}})
    r = env.client.get(f"/api/worlds/{wid}/review/coco")
    assert r.status_code == 404 and "no 2D detections" in r.json()["detail"]


def test_photos_that_are_gone_from_disk_are_not_invented(env):
    wid = _seed(env, {"stages": {"perception": {"detections_2d": DETECTIONS}}}, with_photo=False)
    r = env.client.get(f"/api/worlds/{wid}/review/coco")
    assert r.status_code == 404 and "available to review" in r.json()["detail"]
