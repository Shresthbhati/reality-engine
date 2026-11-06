"""Regression tests: a correction commit inherits its parent's data (BUG 1), the provenance route returns the
entity's own evidence (BUG 4), and the job builds image size / intrinsics only from real sources (BUG 2)."""
from __future__ import annotations

import asyncio
import json
import struct

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from tests.test_commit_validation_contract import _commit, _head, world  # noqa: E402,F401  (fixture reuse)


def _db(fn):
    import apps.api.db as db_mod

    async def run():
        async with db_mod.get_sessionmaker()() as db:
            return await fn(db)

    return asyncio.run(run())


def _seed_parent_sidecars(c, wid):
    """Give HEAD a camera artifact + report, as a real reconstruction does, via the app's own blob store."""
    from apps.api.models import WorldVersion
    from apps.api.storage import store_bytes

    cams = {"image_size": [1024, 768], "cameras": [
        {"evidence_id": f"ev{i}", "position_m": [i, 0.0, 0.0], "rotation_wxyz": [1, 0, 0, 0]} for i in range(3)]}
    digest, _ = store_bytes(json.dumps(cams).encode())

    async def go(db):
        row = await db.get(WorldVersion, _head(c, wid))
        row.cameras_artifact_uri = f"sha256://{digest}"
        row.report = {"status": "SUCCESS", "marker": "parent-report"}
        await db.commit()
        return list(row.source_session_ids or [])

    return _db(go)


def test_correction_commit_inherits_sessions_cameras_and_report(world):
    c, wid, eid = world
    sessions = _seed_parent_sidecars(c, wid)
    assert sessions, "fixture reconstruction must record its source session"
    parent = _head(c, wid)
    before = c.get(f"/api/worlds/{wid}/cameras").json()
    r = _commit(c, wid, {"entity_id": eid, "changes": {"custom_properties": {"note": "checked"}}})
    assert r.status_code == 200, r.text
    head = _head(c, wid)
    assert head != parent
    after = c.get(f"/api/worlds/{wid}/cameras").json()
    assert after["image_size"] == before["image_size"] == [1024, 768]
    assert after["cameras"] == before["cameras"]
    rep = c.get(f"/api/worlds/{wid}/report")
    assert rep.status_code == 200 and rep.json()["marker"] == "parent-report"
    prov = c.get(f"/api/worlds/{wid}/entities/{eid}/provenance").json()
    assert prov["trace_level"] != "none" and prov["source_session_ids"] == sessions

    async def lineage(db):
        from apps.api.models import WorldVersion

        row = await db.get(WorldVersion, head)
        return row.parent_version_id, list(row.source_session_ids)

    assert _db(lineage) == (parent, sessions)


def test_stale_parent_commit_is_a_409(world):
    c, wid, eid = world
    stale = _head(c, wid)
    assert _commit(c, wid, {"entity_id": eid, "changes": {"custom_properties": {"n": 1}}}).status_code == 200
    before = _head(c, wid)
    r = _commit(c, wid, {"entity_id": eid, "parent_version_id": stale,
                         "changes": {"custom_properties": {"n": 2}}})
    assert r.status_code == 409 and _head(c, wid) == before


async def _all_evidence(db):
    from apps.api.models import Evidence
    from sqlalchemy import select

    rows = (await db.execute(select(Evidence).order_by(Evidence.id))).scalars().all()
    return [(r.id, r.name) for r in rows]


def test_provenance_returns_the_entitys_own_supporting_evidence(world):
    c, wid, eid = world
    evs = _db(_all_evidence)
    assert len(evs) >= 3
    mine = [evs[0][0], "not-a-real-evidence-id"]
    r = _commit(c, wid, {"entity_id": eid, "changes": {"custom_properties": {"supporting_evidence_ids": mine}}})
    assert r.status_code == 200, r.text
    prov = c.get(f"/api/worlds/{wid}/entities/{eid}/provenance").json()
    assert prov["trace_level"] == "entity_evidence"
    assert [e["evidence_id"] for e in prov["evidence"]] == [evs[0][0]]      # not every session photo
    assert prov["unresolved_evidence_ids"] == ["not-a-real-evidence-id"]
    assert prov["supporting_evidence_ids"] == mine
    assert set(prov["confidence_vector"]) >= {"classification", "geometry"}   # the stored vector, as stored


def test_provenance_session_fallback_is_labelled(world):
    c, wid, eid = world
    r = _commit(c, wid, {"entity_id": eid, "changes": {"custom_properties": {"supporting_evidence_ids": []}}})
    assert r.status_code == 200, r.text
    prov = c.get(f"/api/worlds/{wid}/entities/{eid}/provenance").json()
    assert prov["trace_level"] == "session" and "session_fallback" in prov["evidence_scope"]


# ---- BUG 2: image size / intrinsics from real sources only ----

def _jpg(path, w, h, exif_f35=None):
    from PIL import Image

    import numpy as np

    img = Image.fromarray(np.random.default_rng(w * h).integers(0, 255, (h, w, 3), dtype=np.uint8))
    if exif_f35:
        exif = Image.Exif()
        exif.get_ifd(0x8769)[0xA405] = exif_f35
        img.save(path, "JPEG", exif=exif)
    else:
        img.save(path, "JPEG")
    return path


def _facts(tmp_path, specs):
    from evidence.image_check import inspect_image

    out = [inspect_image(_jpg(tmp_path / f"i{n}.jpg", w, h, f), None) for n, (w, h, f) in enumerate(specs)]
    assert all(f.ok for f in out), [f.reason for f in out]
    return out


def test_options_use_real_dimensions_and_exif_focal(tmp_path):
    from apps.api.jobs import _intrinsics_options

    facts = _facts(tmp_path, [(1024, 768, 28)] * 2)
    assert all(f.geometry_eligible for f in facts), [f.evidence_class for f in facts]
    kw, rec = _intrinsics_options(facts)
    assert kw["image_size"] == (1024, 768)
    fpx = 28 * (1024 ** 2 + 768 ** 2) ** 0.5 / 43.267
    assert kw["intrinsics"] == pytest.approx((fpx, fpx, 512.0, 384.0))
    assert kw["intrinsics_trust"] == "estimated" and rec["intrinsics"]["source"] == "exif_focal_length_35mm"


def test_options_do_not_invent_a_focal_length(tmp_path):
    from apps.api.jobs import _intrinsics_options

    kw, rec = _intrinsics_options(_facts(tmp_path, [(800, 600, None)] * 2))
    assert kw == {"image_size": (800, 600)} and rec["intrinsics"] is None and "unset" in rec["note"]
    kw, rec = _intrinsics_options([])
    assert kw == {} and rec["intrinsics"] is None


def test_mixed_sizes_report_dominant_size_and_no_intrinsics(tmp_path):
    from apps.api.jobs import _intrinsics_options

    kw, rec = _intrinsics_options(_facts(tmp_path, [(640, 480, 30), (640, 480, 30), (480, 640, 30)]))
    assert kw["image_size"] == (640, 480) and "intrinsics" not in kw and "mixed_image_sizes" in rec


def _fake_session(root, cameras_txt=None, cameras_bin=None):
    from reconstruction.colmap_session import ColmapSession

    s = ColmapSession(root)
    model = s.current / "sparse" / "0"
    model.mkdir(parents=True)
    if cameras_txt is not None:
        (model / "cameras.txt").write_text(cameras_txt)
    if cameras_bin is not None:
        (model / "cameras.bin").write_bytes(cameras_bin)
    (s.current / "manifest.json").write_text(json.dumps({"version": 1, "model_dir": "sparse/0"}))
    return s


def test_colmap_estimated_intrinsics_are_used_and_labelled_estimated(tmp_path):
    from apps.api.jobs import _intrinsics_options

    facts = _facts(tmp_path, [(1024, 768, None)] * 2)
    txt = "# Camera list\n1 SIMPLE_RADIAL 1024 768 900.5 512 384 0.01\n"
    kw, rec = _intrinsics_options(facts, _fake_session(tmp_path / "s1", cameras_txt=txt))
    assert kw["intrinsics"] == (900.5, 900.5, 512.0, 384.0) and kw["intrinsics_trust"] == "estimated"
    assert rec["intrinsics"]["source"] == "colmap_estimated_previous_sparse_model"
    blob = struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, 1024, 768) + struct.pack("<4d", 800.0, 810.0, 500.0, 380.0)
    kw, _ = _intrinsics_options(facts, _fake_session(tmp_path / "s2", cameras_bin=blob))
    assert kw["intrinsics"] == (800.0, 810.0, 500.0, 380.0)
    # a model solved for a different image size is not applied to these photographs
    kw, _ = _intrinsics_options(facts, _fake_session(tmp_path / "s3", cameras_txt="1 PINHOLE 640 480 500 500 320 240\n"))
    assert "intrinsics" not in kw


def test_estimated_trust_label_exists_and_default_stays_trusted():
    from engine.pipeline.vertical_slice import VerticalSliceOptions

    assert VerticalSliceOptions().intrinsics_trust == "trusted"
    assert VerticalSliceOptions(intrinsics=(1, 1, 1, 1), intrinsics_trust="estimated").intrinsics_trust == "estimated"
