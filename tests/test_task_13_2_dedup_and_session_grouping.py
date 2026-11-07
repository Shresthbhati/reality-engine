"""Tests for Task 13.2: pHash perceptual deduplication and time/GPS session grouping callers."""

from __future__ import annotations

import asyncio
import importlib
import io

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from apps.api.models import Evidence, Session, World, new_id
from apps.api.storage import store_bytes


def _make_near_duplicate_pair(seed: int = 42) -> tuple[bytes, bytes]:
    """Create two images that have different SHA256 bytes but near-identical pHash (distance <= 8)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    img_sharp = Image.fromarray(arr, mode="L")
    # Slightly perturbed brightness/compression creates different sha256 but same visual frequency structure
    arr_perturbed = np.clip(arr.astype(int) + 2, 0, 255).astype(np.uint8)
    img_soft = Image.fromarray(arr_perturbed, mode="L")

    b_sharp = io.BytesIO()
    img_sharp.save(b_sharp, format="JPEG", quality=95)

    b_soft = io.BytesIO()
    img_soft.save(b_soft, format="JPEG", quality=80)

    return b_sharp.getvalue(), b_soft.getvalue()


def _make_distinct_image(seed: int = 999) -> bytes:
    """Create a completely distinct image with a high pHash distance."""
    w, h = 256, 256
    arr = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def client_env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "worldstore"))
    monkeypatch.setenv("REALITY_API_TOKEN", "test-token")

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    asyncio.run(db_mod.init_db())

    with TestClient(main_mod.app, headers={"Authorization": "Bearer test-token"}) as client:
        yield client, db_mod


def test_uploads_check_detects_phash_near_duplicates(client_env):
    """POST /api/uploads/check reports near-duplicate photos via pHash and identifies the sharper canonical."""
    client, _ = client_env
    sharp_bytes, soft_bytes = _make_near_duplicate_pair(seed=42)
    diff_bytes = _make_distinct_image(seed=999)

    files = [
        ("files", ("photo_sharp.jpg", sharp_bytes, "image/jpeg")),
        ("files", ("photo_soft.jpg", soft_bytes, "image/jpeg")),
        ("files", ("photo_different.jpg", diff_bytes, "image/jpeg")),
    ]

    r = client.post("/api/uploads/check", files=files)
    assert r.status_code == 200, r.text
    data = r.json()

    files_by_name = {f["name"]: f for f in data["files"]}

    assert files_by_name["photo_sharp.jpg"]["phash"] is not None
    assert files_by_name["photo_soft.jpg"]["phash"] is not None
    assert files_by_name["photo_different.jpg"]["phash"] is not None

    # Distinct SHA-256 digests
    assert files_by_name["photo_sharp.jpg"]["sha256"] != files_by_name["photo_soft.jpg"]["sha256"]

    # One photo must be flagged as near-duplicate of the other
    soft_flag = files_by_name["photo_soft.jpg"]["near_duplicate_of"] == "photo_sharp.jpg"
    sharp_flag = files_by_name["photo_sharp.jpg"]["near_duplicate_of"] == "photo_soft.jpg"
    assert soft_flag or sharp_flag

    # The distinct image should NOT be flagged as duplicate
    assert files_by_name["photo_different.jpg"]["near_duplicate_of"] is None
    assert data["summary"]["near_duplicate_count"] == 1


def test_reconstruction_with_filter_duplicates_flag(client_env, monkeypatch):
    """POST /api/reconstructions prunes burst near-duplicates when filter_duplicates=True."""
    client, _ = client_env
    # Stub the slow reconstruct background job
    monkeypatch.setattr("apps.api.jobs.enqueue_job", lambda *args, **kwargs: type("J", (), {"id": "job_stub", "type": "reconstruct", "status": "queued"})())

    sharp_bytes, soft_bytes = _make_near_duplicate_pair(seed=101)
    other_bytes = _make_distinct_image(seed=888)

    # 1. filter_duplicates=False (default safe mode: keep all photos, flag duplicates)
    files_unfiltered = [
        ("files", ("c_sharp.jpg", sharp_bytes, "image/jpeg")),
        ("files", ("c_soft.jpg", soft_bytes, "image/jpeg")),
        ("files", ("c_other.jpg", other_bytes, "image/jpeg")),
    ]
    r_unfiltered = client.post("/api/reconstructions", files=files_unfiltered, data={"filter_duplicates": "false"})
    assert r_unfiltered.status_code == 202, r_unfiltered.text
    unfiltered_data = r_unfiltered.json()
    accepted_names = [e["name"] for e in unfiltered_data["evidence"]]
    assert "c_sharp.jpg" in accepted_names
    assert "c_soft.jpg" in accepted_names
    assert "c_other.jpg" in accepted_names

    # 2. filter_duplicates=True (prune redundant burst frames)
    files_filtered = [
        ("files", ("c_sharp2.jpg", sharp_bytes, "image/jpeg")),
        ("files", ("c_soft2.jpg", soft_bytes, "image/jpeg")),
        ("files", ("c_other2.jpg", other_bytes, "image/jpeg")),
    ]
    r_filtered = client.post("/api/reconstructions", files=files_filtered, data={"filter_duplicates": "true"})
    assert r_filtered.status_code == 202, r_filtered.text
    filtered_data = r_filtered.json()
    accepted_filtered = [e["name"] for e in filtered_data["evidence"]]
    rejected_filtered = {r["name"]: r["reason"] for r in filtered_data["rejected"]}

    # The distinct photo is retained, along with the canonical photo
    assert "c_other2.jpg" in accepted_filtered
    assert len(accepted_filtered) == 2
    assert len(rejected_filtered) == 1
    rejected_name = next(iter(rejected_filtered))
    assert rejected_name in ("c_sharp2.jpg", "c_soft2.jpg")
    assert "near_duplicate of" in rejected_filtered[rejected_name]


def test_session_grouping_by_time_and_gps_endpoint(client_env):
    """POST /api/sessions/group correctly groups evidence by time gap (>30m) and GPS (>500m)."""
    client, _ = client_env
    t0 = 1700000000.0  # reference epoch
    items = [
        # Session 1: Room A, 10:00 - 10:05
        {"id": "ev_01", "captured_at": t0, "latitude": 40.7128, "longitude": -74.0060},
        {"id": "ev_02", "captured_at": t0 + 120, "latitude": 40.7129, "longitude": -74.0061},
        # Session 2: Room B in different building 2 km away, 10:10
        {"id": "ev_03", "captured_at": t0 + 600, "latitude": 40.7300, "longitude": -74.0060},
        {"id": "ev_04", "captured_at": t0 + 700, "latitude": 40.7301, "longitude": -74.0061},
        # Session 3: Same building as Session 2, but captured 2 hours later (>1800s gap)
        {"id": "ev_05", "captured_at": t0 + 8000, "latitude": 40.7302, "longitude": -74.0062},
    ]

    r = client.post("/api/sessions/group", json={
        "items": items,
        "max_time_gap_s": 1800.0,
        "max_distance_m": 500.0,
        "group_prefix": "site_session",
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total_items"] == 5
    assert data["group_count"] == 3

    groups = data["groups"]
    assert groups[0]["item_ids"] == ["ev_01", "ev_02"]
    assert groups[1]["item_ids"] == ["ev_03", "ev_04"]
    assert groups[2]["item_ids"] == ["ev_05"]
    assert groups[0]["group_id"] == "site_session-001"
    assert groups[0]["center_gps"] is not None


def test_session_deduplicate_endpoint(client_env):
    """POST /api/sessions/{session_id}/deduplicate inspects session evidence in DB."""
    client, db_mod = client_env

    session_id = new_id("ses")
    world_id = new_id("wld")
    img_a, img_b = _make_near_duplicate_pair(seed=55)
    img_c = _make_distinct_image(seed=777)

    digest_a, _ = store_bytes(img_a)
    digest_b, _ = store_bytes(img_b)
    digest_c, _ = store_bytes(img_c)

    ev1_id = new_id("ev")
    ev2_id = new_id("ev")
    ev3_id = new_id("ev")

    async def _setup_data():
        async_session = db_mod.get_sessionmaker()
        async with async_session() as db:
            w = World(id=world_id, name="Dedup Test World", owner_id="user_test", team_id="default")
            s = Session(id=session_id, name="Dedup Test Session", status="uploaded", world_id=world_id, owner_id="user_test", team_id="default")
            db.add_all([w, s])
            ev1 = Evidence(
                id=ev1_id, session_id=session_id, team_id="default", name="sharp.jpg", type="photo",
                artifact_uri=f"sha256://{digest_a}", checksum=digest_a, size=len(img_a),
                metadata_json={"image_facts": {"sharpness": 150.0}}
            )
            ev2 = Evidence(
                id=ev2_id, session_id=session_id, team_id="default", name="soft.jpg", type="photo",
                artifact_uri=f"sha256://{digest_b}", checksum=digest_b, size=len(img_b),
                metadata_json={"image_facts": {"sharpness": 40.0}}
            )
            ev3 = Evidence(
                id=ev3_id, session_id=session_id, team_id="default", name="noise.jpg", type="photo",
                artifact_uri=f"sha256://{digest_c}", checksum=digest_c, size=len(img_c),
                metadata_json={"image_facts": {"sharpness": 120.0}}
            )
            db.add_all([ev1, ev2, ev3])
            await db.commit()

    asyncio.run(_setup_data())

    r = client.post(f"/api/sessions/{session_id}/deduplicate", json={"hash_type": "phash", "dry_run": False})
    assert r.status_code == 200, r.text
    res = r.json()

    assert res["total_input"] == 3
    assert res["retained_count"] == 2
    assert res["duplicate_count"] == 1
    assert ev1_id in res["retained_ids"]
    assert ev2_id in res["duplicate_ids"]
    assert ev3_id in res["retained_ids"]
    assert res["duplicates"][0]["canonical_id"] == ev1_id
    assert res["duplicates"][0]["duplicate_id"] == ev2_id
