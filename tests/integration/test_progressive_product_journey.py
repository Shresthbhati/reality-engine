"""Golden product journeys: REAL photographs -> API -> worker -> real COLMAP /
MiDaS -> WorldIR -> WorldStore -> versions, and back out through the same
routes the Studio reads.

Not synthetic: every scenario feeds the South Building photographs
(datasets/south_building, 1024px derivatives of 128 real photos of a
university building; see its MANIFEST.json for source and credit). Nothing
here mocks the reconstruction: where a scenario asserts a failure it is a
failure the real engine produced on real wide-baseline photos.

Dataset windows (indices into the name-sorted image list; chosen from the
measured pairwise SIFT-match matrix, not by eye):

    ONE_PHOTO   [6]        a single view
    SIX         [15:21]    6 strongly overlapping views (RANSAC-verified matches
                           between neighbours: 97, 558, 556, 623, 363)
    ADD_4       [21:25]    the next 4 views along the walk
    ADD_10      [5:15]     10 more views, weakly linked to SIX (12-40 matches)
    HARD        [0:6]      wide-baseline frames; real COLMAP finds no initial pair

Marked slow: they run the real engine (about 4-6 minutes in total on CPU).
"""

from __future__ import annotations

import importlib
import io
import re
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
from fastapi.testclient import TestClient

DATASET = Path(__file__).resolve().parents[2] / "datasets" / "south_building" / "images"
IMGS = sorted(DATASET.glob("*.JPG")) if DATASET.exists() else []

ONE_PHOTO = IMGS[6:7]
SIX = IMGS[15:21]
ADD_4 = IMGS[21:25]
ADD_10 = IMGS[5:15]
HARD = IMGS[0:6]


def _engines_available() -> tuple[bool, str]:
    try:
        from perception.availability import report_real_models

        status = {m.name: m.status for m in report_real_models()}
    except Exception as exc:  # pragma: no cover
        return False, f"availability probe failed: {exc}"
    missing = [n for n in ("colmap", "midas") if status.get(n) != "available"]
    return (not missing), f"unavailable real engines: {missing}"


_OK, _WHY = _engines_available()
pytestmark = [
    pytest.mark.slow,
    pytest.mark.e2e,
    pytest.mark.skipif(len(IMGS) < 25, reason="South Building dataset not present"),
    pytest.mark.skipif(not _OK, reason=f"real engines required -- UNVERIFIED ({_WHY})"),
]


# ------------------------------------------------------------------ harness


@contextmanager
def app_client(root: Path):
    """A fresh app process's worth of state over the given directory: same
    root -> same database + WorldStore (used for the restart proof)."""
    import os

    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(root / 'app.db').as_posix()}"
    os.environ["STORAGE_ROOT"] = str(root / "artifacts")
    os.environ["WORLDSTORE_ROOT"] = str(root / "ws")
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws

    ws._store = None
    with TestClient(main_mod.app) as c:
        yield c


def _post(c, paths, world_id=None, **form):
    files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in paths]
    data = dict(form)
    if world_id:
        data["world_id"] = world_id
    r = c.post("/api/reconstructions", files=files, data=data)
    assert r.status_code == 202, r.text
    return r.json()


def _settle(c, world_id, timeout=900.0):
    end = time.time() + timeout
    st = None
    while time.time() < end:
        st = c.get(f"/api/worlds/{world_id}/status").json()
        if not st["in_progress"]:
            return st
        time.sleep(1.0)
    raise AssertionError(f"world {world_id} never settled; last status: {st}")


def _worldir(c, wid, version=None):
    r = c.get(f"/api/worlds/{wid}/worldir", params={"version": version} if version else None)
    assert r.status_code == 200, r.text
    return r.json()


def _ply_vertices(raw: bytes) -> int:
    m = re.search(rb"element vertex (\d+)", raw[:2000])
    assert m, "not a PLY stream"
    return int(m.group(1))


# ---------------------------------------------------------------- TEST A


def test_A_one_photograph_gives_a_rough_labelled_model(tmp_path):
    with app_client(tmp_path) as c:
        body = _post(c, ONE_PHOTO)
        assert body["created_world"] is True and len(body["evidence"]) == 1
        wid = body["world_id"]
        st = _settle(c, wid)

        # honest state: a rough single-view hypothesis is PARTIAL, never "ready"
        assert st["state"] == "PARTIALLY_COMPLETE"
        assert st["has_model"] and st["model"]["level"] == 0
        assert st["model"]["model_state"] == "ROUGH"
        assert st["model"]["scale"]["state"] == "relative"
        assert [v["label"] for v in st["versions"]] == ["V1"]

        ents = _worldir(c, wid)["entities"]
        structure = [e for e in ents.values() if e["type"] in ("wall", "floor", "ceiling")]
        assert structure, "the single view produced no inferred planes"
        # nothing in a one-photo world may claim to be observed/reconstructed, or be confident
        for e in ents.values():
            assert e["provenance"] in ("INFERRED", "ESTIMATED", "UNKNOWN"), e["id"]
            if e["provenance"] != "UNKNOWN":
                assert e["confidence"] <= 0.35 + 1e-9, (e["id"], e["confidence"])
        unobserved = ents["boot-unobserved"]
        assert unobserved["provenance"] == "UNKNOWN" and unobserved["confidence"] == 0.0

        # the result is drawable: real points + the (assumed) camera
        ply = c.get(f"/api/worlds/{wid}/points")
        assert ply.status_code == 200 and _ply_vertices(ply.content) >= 1000
        cams = c.get(f"/api/worlds/{wid}/cameras").json()["cameras"]
        assert len(cams) == 1

        # provenance traces back to the uploaded file, exactly (observation level)
        prov = c.get(f"/api/worlds/{wid}/entities/{structure[0]['id']}/provenance").json()
        assert prov["trace_level"] == "observation"
        assert prov["evidence"][0]["evidence_id"] == body["evidence"][0]["id"]

        # advice comes from the frame, and says what is unknown
        kinds = {g["kind"] for g in st["guidance"]}
        assert {"add_viewpoint", "unobserved"} <= kinds
        assert all(g["basis"] for g in st["guidance"])
        report = c.get(f"/api/worlds/{wid}/report").json()
        assert report["level"] == 0 and report["model_state"] == "ROUGH"


# ---------------------------------------------------------------- TEST B


def test_B_six_photographs_give_real_registered_geometry(tmp_path):
    with app_client(tmp_path) as c:
        body = _post(c, SIX)
        wid = body["world_id"]
        st = _settle(c, wid)

        m = st["model"]
        assert st["state"] in ("READY_TO_INSPECT", "PARTIALLY_COMPLETE")
        assert m["level"] >= 2, f"multi-view reconstruction did not run: {m['attempts']}"
        assert m["images_used"] == 6 and m["images_registered"] == 6
        # six photos are a PARTIAL model, not a finished digital twin
        assert m["model_state"] in ("PARTIAL", "REFINED")
        assert m["scale"]["state"] == "relative"  # no metric reference was supplied

        wir = _worldir(c, wid)
        walls = [e for e in wir["entities"].values() if e["type"] == "wall"]
        assert len(walls) >= 3, "a building facade should yield several wall planes"
        for e in wir["entities"].values():
            assert e["provenance"] != "OBSERVED"  # planes are inferred from SfM points
            trace = (e.get("custom_properties") or {}).get("reconstruction")
            if e["provenance"] in ("RECONSTRUCTED", "INFERRED"):
                assert trace and trace["session_id"] and trace["level"] == m["level"]

        assert _ply_vertices(c.get(f"/api/worlds/{wid}/points").content) >= 500
        # evidence classes survive into provenance
        used = wir["metadata"]["evidence"]["used"]
        assert len(used) == 6
        assert used[0]["evidence_class"] in ("photograph", "photograph_unverified")
        # the gravity frame came from the cameras, not from mistaking the facade for the floor
        assert wir["metadata"]["frame"]["up_source"].startswith("camera_up_vectors")

        q = {x["name"]: x["level"] for x in st["evidence_summary"]["quality"]}
        assert q["Overlap"] == "good"
        assert any("one side" in g["message"] for g in st["guidance"])  # sides/back unobserved


# ---------------------------------------------------------------- TEST C + G


def test_C_G_progressive_refinement_keeps_one_world_and_diffs_honestly(tmp_path):
    with app_client(tmp_path) as c:
        wid = _post(c, SIX)["world_id"]
        st1 = _settle(c, wid)
        v1 = st1["model"]["version_id"]
        v1_worldir = _worldir(c, wid, v1)

        b2 = _post(c, ADD_4, world_id=wid)
        assert b2["world_id"] == wid and b2["created_world"] is False
        st2 = _settle(c, wid)
        v2 = st2["model"]["version_id"]

        _post(c, ADD_10, world_id=wid)
        st3 = _settle(c, wid)
        v3 = st3["model"]["version_id"]

        # one persistent world, three chained versions
        assert len({v1, v2, v3}) == 3
        versions = {v["label"]: v for v in st3["versions"]}
        assert set(versions) == {"V1", "V2", "V3"}
        assert versions["V2"]["parent_version_id"] == v1
        assert versions["V3"]["parent_version_id"] == v2
        assert [versions[k]["images_used"] for k in ("V1", "V2", "V3")] == [6, 10, 20]
        assert len(c.get("/api/worlds").json()["items"]) == 1  # not a new world per upload

        # V2 is built from V1's evidence PLUS the new images
        assert st2["model"]["level"] == 5  # incremental refinement of earlier evidence
        assert st2["evidence_summary"]["headline"].startswith("4 new images added.")
        e2 = {e["id"] for e in st2["evidence"] if e["in_current_model"]}
        e1 = {e["id"] for e in st1["evidence"]}
        assert e1 <= e2 and len(e2) == 10

        # placing images is monotone: more evidence never registers FEWER cameras
        regs = [s["model"]["images_registered"] for s in (st1, st2, st3)]
        assert regs == sorted(regs) and regs[-1] > regs[0], regs
        # and the coverage the user is told about grows as they walk around the scene
        def span(s):
            basis = next(q for q in s["evidence_summary"]["quality"] if q["name"] == "Coverage")["basis"]
            return float(re.search(r"span (\d+) of 360", basis).group(1))
        spans = [span(s) for s in (st1, st2, st3)]
        assert spans[-1] > spans[0], spans

        # the previous version is untouched and still inspectable
        assert _worldir(c, wid, v1) == v1_worldir

        # G: the diff separates WHAT kind of thing changed
        d = c.get(f"/api/worlds/{wid}/diff", params={"base": v1, "head": v2}).json()
        cats = d["categories"]
        assert set(cats) >= {"identity", "geometry", "transform", "semantic", "confidence",
                             "uncertainty", "provenance", "topology", "properties"}
        assert sum(cats.values()) > 0 and cats["geometry"] > 0  # geometry really moved
        assert cats["semantic"] == 0  # planes did not change TYPE just because more views arrived
        same = c.get(f"/api/worlds/{wid}/diff", params={"base": v2, "head": v2}).json()
        assert sum(same["categories"].values()) == 0  # no fabricated differences


# ---------------------------------------------------------------- TEST D


def test_D_bad_evidence_is_rejected_classified_or_honestly_degraded(tmp_path):
    from PIL import Image

    with app_client(tmp_path) as c:
        # 1. nothing usable at all -> 422 naming every reason, no world created
        r = c.post("/api/reconstructions", files=[
            ("files", ("frame_00.jpg", b"jpeg-0000", "image/jpeg")),
            ("files", ("frame_01.jpg", b"not an image", "image/jpeg")),
        ])
        assert r.status_code == 422
        assert {x["name"] for x in r.json()["detail"]["rejected"]} == {"frame_00.jpg", "frame_01.jpg"}
        assert c.get("/api/worlds").json()["items"] == []

        # 2. a floor plan among photos: classified, kept as context, never geometry
        buf = io.BytesIO()
        plan = Image.new("RGB", (400, 300), (255, 255, 255))
        for x in range(0, 400, 40):
            for y in range(100, 110):
                plan.putpixel((x, y), (0, 0, 0))
        plan.save(buf, "PNG")
        files = [("files", (ONE_PHOTO[0].name, ONE_PHOTO[0].read_bytes(), "image/jpeg")),
                 ("files", ("floorplan.png", buf.getvalue(), "image/png")),
                 ("files", ("broken.jpg", b"garbage", "image/jpeg"))]
        r = c.post("/api/reconstructions", files=files)
        assert r.status_code == 202
        body = r.json()
        assert [x["name"] for x in body["rejected"]] == ["broken.jpg"]
        st = _settle(c, body["world_id"])
        by_name = {e["name"]: e for e in st["evidence"]}
        assert by_name["floorplan.png"]["evidence_class"] == "flat_graphic"
        assert by_name["floorplan.png"]["used_for_geometry"] is False
        assert by_name[ONE_PHOTO[0].name]["used_for_geometry"] is True
        assert st["model"]["images_used"] == 1
        assert any(x["kind"] == "context" for x in st["evidence_summary"]["excluded"])
        wir = _worldir(c, body["world_id"])
        assert {e["evidence_class"] for e in wir["metadata"]["evidence"]["excluded"]} == {"flat_graphic"}

        # 3. a declared render is visual reference, not observed geometry
        body = _post(c, ONE_PHOTO, evidence_class="render")
        st = _settle(c, body["world_id"])
        assert st["state"] == "NEEDS_MORE_EVIDENCE" and not st["has_model"]
        assert "render" in st["failure"]["message"]


# ---------------------------------------------------------------- TEST D2 / E


def test_D2_hard_low_overlap_photos_fall_back_and_never_fake_success(tmp_path):
    """Real wide-baseline photos: the real COLMAP mapper finds no usable
    initial pair. The product must not return an empty world or a success."""
    with app_client(tmp_path) as c:
        wid = _post(c, HARD)["world_id"]
        st = _settle(c, wid)
        m = st["model"]
        assert st["state"] == "PARTIALLY_COMPLETE"          # not READY, not FAILED-with-nothing
        assert m["level"] == 0 and m["model_state"] == "ROUGH"
        failed = [a for a in m["attempts"] if a["outcome"] == "failed"]
        assert failed and failed[0]["level"] == 2 and "colmap" in failed[0]["detail"].lower()
        # N photographs never silently become 1: all six are used, none is claimed as registered
        assert m["images_used"] == 6 and m["images_registered"] == 0
        wir = _worldir(c, wid, m["version_id"])
        ents = wir["entities"] if "entities" in wir else wir["world"]["entities"]
        surfaces = [e for e in ents if e.endswith("-visible-surface")]
        assert len(surfaces) == 6, surfaces                  # one hypothesis per photograph
        # the user is told, in plain language, what would help
        assert any(g["kind"] == "engine_note" and "multi-view geometry" in g["message"] for g in st["guidance"])
        assert any(g["kind"] == "add_overlap" and "6 photos" in g["message"] for g in st["guidance"])
        # and overlap is NOT shown as good when the engine could not use it
        overlap = next(q for q in st["evidence_summary"]["quality"] if q["name"] == "Overlap")
        assert overlap["level"] == "low" and "no usable image pair" in overlap["basis"]


def test_E_engine_crash_is_a_recorded_attempt_not_an_empty_world(tmp_path, monkeypatch):
    import engine.pipeline.vertical_slice as vs

    def boom(*a, **k):
        raise RuntimeError("COLMAP crashed (simulated)")

    monkeypatch.setattr(vs, "vertical_slice", boom)
    with app_client(tmp_path) as c:
        wid = _post(c, SIX[:2])["world_id"]
        st = _settle(c, wid)
        assert st["has_model"] and st["model"]["level"] == 0 and st["state"] == "PARTIALLY_COMPLETE"
        assert "COLMAP crashed" in st["model"]["attempts"][0]["detail"]


# ---------------------------------------------------------------- TEST F


def test_F_restart_reloads_the_persisted_version(tmp_path):
    with app_client(tmp_path) as c:
        wid = _post(c, ONE_PHOTO)["world_id"]
        before = _settle(c, wid)
        v1 = before["model"]["version_id"]
        wir_before = _worldir(c, wid, v1)
        points_before = c.get(f"/api/worlds/{wid}/points").content

    # a brand-new app over the same database + WorldStore
    with app_client(tmp_path) as c2:
        after = c2.get(f"/api/worlds/{wid}/status").json()
        assert after["has_model"] and after["model"]["version_id"] == v1
        assert after["state"] == "PARTIALLY_COMPLETE" and not after["in_progress"]
        assert _worldir(c2, wid, v1) == wir_before
        assert c2.get(f"/api/worlds/{wid}/points").content == points_before
        # and it keeps refining the SAME world after a restart
        _post(c2, SIX[:2], world_id=wid)
        st = _settle(c2, wid)
        assert [v["label"] for v in st["versions"]] == ["V1", "V2"]


# ---------------------------------------------------------------- TEST H


def _make_video(path: Path, stills: list[Path], repeats: int, fps: int = 6) -> int:
    """A real MP4 (OpenCV/FFMPEG) that dwells on each real photograph
    `repeats` times, like a slow pan that lingers. Returns the frame count."""
    import cv2

    first = cv2.imread(str(stills[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV could not open an MP4 writer"
    n = 0
    for still in stills:
        img = cv2.imread(str(still))
        for _ in range(repeats):
            writer.write(img)
            n += 1
    writer.release()
    return n


def test_H_video_is_sampled_into_frame_evidence_and_duplicates_are_measured(tmp_path):
    video = tmp_path / "walk.mp4"
    frames_written = _make_video(video, SIX, repeats=3)  # 18 frames, 6 distinct views
    assert frames_written == 18
    (tmp_path / "app").mkdir()
    with app_client(tmp_path / "app") as c:
        r = c.post("/api/reconstructions", files=[("files", ("walk.mp4", video.read_bytes(), "video/mp4"))])
        assert r.status_code == 202, r.text
        body = r.json()
        kinds = [e["type"] for e in body["evidence"]]
        assert kinds.count("video") == 1 and kinds.count("photo") == 18  # video kept + sampled frames
        wid = body["world_id"]
        st = _settle(c, wid)

        # only the frames are geometry evidence; the video file itself is not
        assert st["model"]["images_used"] == 18
        assert st["has_model"] and st["model"]["level"] >= 2
        # 6 distinct views repeated 3x: the two repeats of each are near-duplicates
        summary = next(g for g in [st["evidence_summary"]] if g)
        assert "mostly repeat" in (summary["headline"] or "")
        assert any(e["contribution"] == "redundant" for e in st["evidence"])
        # frame identity survives: each frame names its source video and position
        ev = c.get("/api/evidence", params={"session_id": body["session_id"]}).json()["items"]
        frame_meta = [e["metadata"] for e in ev if e["type"] == "photo"]
        assert all(m["source_video"] == "walk.mp4" and "frame_index" in m for m in frame_meta)

        # an unreadable "video" is rejected with a reason, not silently turned into a world
        r2 = c.post("/api/reconstructions", files=[("files", ("bad.mp4", b"not a video at all", "video/mp4"))])
        assert r2.status_code == 422
        assert "video could not be read" in r2.json()["detail"]["rejected"][0]["reason"]
