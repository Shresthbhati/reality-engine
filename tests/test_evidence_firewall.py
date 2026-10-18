"""Evidence Quality Firewall: pure check/score functions + routes + the
'rejected is excluded from reconstruction' contract."""

from __future__ import annotations

import io

import pytest

from evidence import quality_checks as qc

GOOD = {
    "ok": True, "width": 2000, "height": 1500, "evidence_class": "photograph",
    "sharpness": 300.0, "exif": {"make": "Apple", "model": "iPhone 13"},
}


def _by_name(checks):
    return {c["name"]: c for c in checks}


# ---------------------------------------------------------------- pure functions


def test_uninspected_image_has_no_checks_and_no_score():
    assert qc.evaluate_checks(None) == []
    assert qc.evaluate_checks({}) == []
    assert qc.quality_score([]) is None
    assert qc.auto_result([]) is None


def test_clean_photo_passes_everything_and_scores_100():
    checks = qc.evaluate_checks(GOOD)
    assert {c["status"] for c in checks} == {"pass"}
    assert qc.quality_score(checks) == 100
    assert qc.auto_result(checks) == "accepted"
    assert qc.key_issues(checks) == []


def test_undecodable_image_is_rejected_with_score_zero():
    checks = qc.evaluate_checks({"ok": False, "reason": "not a decodable image (UnidentifiedImageError)"})
    assert [(c["name"], c["status"]) for c in checks] == [("Readable", "fail")]
    assert qc.quality_score(checks) == 0
    assert qc.auto_result(checks) == "rejected"


def test_thresholds_and_score_formula():
    soft = dict(GOOD, sharpness=qc.SHARPNESS_WARN - 1, exif={})  # two warnings
    checks = qc.evaluate_checks(soft)
    c = _by_name(checks)
    assert c["Sharpness"]["status"] == "warn" and c["Camera metadata"]["status"] == "warn"
    assert qc.quality_score(checks) == 100 - 2 * qc.WARN_PENALTY
    assert qc.auto_result(checks) == "accepted_with_warning"

    blurry = dict(GOOD, sharpness=qc.SHARPNESS_FAIL - 0.1)
    checks = qc.evaluate_checks(blurry)
    assert _by_name(checks)["Sharpness"]["status"] == "fail"
    assert qc.quality_score(checks) == 100 - qc.FAIL_PENALTY
    assert qc.auto_result(checks) == "needs_review"

    tiny = dict(GOOD, width=400, height=300)  # 0.12 MP
    assert _by_name(qc.evaluate_checks(tiny))["Resolution"]["status"] == "fail"
    low = dict(GOOD, width=800, height=600)  # 0.48 MP
    assert _by_name(qc.evaluate_checks(low))["Resolution"]["status"] == "warn"


def test_missing_sharpness_is_omitted_not_invented():
    names = [c["name"] for c in qc.evaluate_checks(dict(GOOD, sharpness=None))]
    assert "Sharpness" not in names


def test_non_photograph_class_needs_review_and_duplicate_is_rejected():
    plan = dict(GOOD, evidence_class="flat_graphic")
    assert qc.auto_result(qc.evaluate_checks(plan)) == "needs_review"
    dup = qc.evaluate_checks(GOOD, duplicate_of="IMG_1.jpg")
    assert _by_name(dup)["Duplicate"]["status"] == "fail"
    assert qc.auto_result(dup) == "rejected"


def test_explanation_is_built_only_from_the_checks():
    checks = qc.evaluate_checks(dict(GOOD, sharpness=1.0, exif={}))
    text = qc.explain(checks)
    assert text == qc.explain(checks)  # deterministic
    assert "Failed: Sharpness" in text and "Warnings: Camera metadata" in text
    assert text.startswith(f"{sum(c['status'] == 'pass' for c in checks)} of {len(checks)} checks passed.")


# ---------------------------------------------------------------- routes

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
PIL = pytest.importorskip("PIL")
np = pytest.importorskip("numpy")

from tests.test_application_api import _attach, _upload_photo, _wait_job, client  # noqa: E402,F401


def _png(seed: int, noisy: bool = True) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (800, 1200, 3), dtype=np.uint8) if noisy else np.full((800, 1200, 3), 128, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _upload_png(client, sid, name, data):
    r = client.post(f"/api/uploads?session_id={sid}", files={"file": (name, data, "image/png")})
    assert r.status_code == 201, r.text
    return r.json()["evidence_id"]


def test_quality_list_reports_real_checks_and_counts(client):
    sid = client.post("/api/sessions", json={"name": "Q"}).json()["id"]
    good = _upload_png(client, sid, "noise.png", _png(1))
    flat = _upload_png(client, sid, "flat.png", _png(0, noisy=False))
    junk = _upload_photo(client, sid, "broken.jpg", b"not an image")["evidence_id"]

    r = client.get("/api/evidence/quality")  # must not be swallowed by /api/evidence/{id}
    assert r.status_code == 200, r.text
    body = r.json()
    items = {i["id"]: i for i in body["items"]}
    assert body["counts"]["total"] == 3

    g = items[good]
    assert g["inspected"] and g["width"] == 1200 and g["height"] == 800
    assert {c["name"] for c in g["checks"]} >= {"Readable", "Resolution", "Sharpness", "Camera metadata"}
    assert g["result"] == "accepted_with_warning" and g["decision"] is None  # no camera EXIF -> warn only
    assert g["contribution"]["available"] is False and g["contribution"]["reason"]  # no world: says why

    assert items[flat]["result"] == "needs_review"  # flat colour -> not a photograph, zero sharpness
    assert items[junk]["result"] == "rejected" and items[junk]["score"] == 0
    assert body["counts"]["rejected"] == 1 and body["counts"]["needs_review"] == 1

    one = client.get(f"/api/evidence/{good}/quality")
    assert one.status_code == 200 and one.json()["id"] == good
    assert client.get("/api/evidence/ev_nope/quality").status_code == 404


def test_duplicate_upload_is_flagged_against_the_earlier_copy(client):
    sid = client.post("/api/sessions", json={"name": "D"}).json()["id"]
    data = _png(7)
    first = _upload_png(client, sid, "a.png", data)
    second = _upload_png(client, sid, "b.png", data)
    items = {i["id"]: i for i in client.get("/api/evidence/quality").json()["items"]}
    assert items[first]["result"] == "accepted_with_warning"
    assert items[second]["result"] == "rejected"
    assert "a.png" in " ".join(items[second]["key_issues"])


def test_decision_validation_persistence_and_404(client):
    sid = client.post("/api/sessions", json={"name": "Dec"}).json()["id"]
    ev = _upload_png(client, sid, "x.png", _png(3))

    assert client.put(f"/api/evidence/{ev}/decision", json={"decision": "maybe"}).status_code == 422
    assert client.put("/api/evidence/ev_missing/decision", json={"decision": "accepted"}).status_code == 404

    r = client.put(f"/api/evidence/{ev}/decision", json={"decision": "needs_review", "reason": "check angle"})
    assert r.status_code == 200 and r.json()["decision"] == "needs_review"
    r = client.put(f"/api/evidence/{ev}/decision", json={"decision": "accepted"})  # upsert, not a second row
    assert r.json()["reason"] is None

    item = client.get(f"/api/evidence/{ev}/quality").json()
    assert item["decision"]["decision"] == "accepted" and item["result"] == "accepted"
    assert item["auto_result"] == "accepted_with_warning"  # the checks are unchanged by the decision

    assert client.delete(f"/api/evidence/{ev}").status_code == 204  # decision row goes with its evidence
    assert client.get(f"/api/evidence/{ev}/quality").status_code == 404


def test_rejected_evidence_is_excluded_from_reconstruction(client, tmp_path, monkeypatch):
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None
    sid = client.post("/api/sessions", json={"name": "Excl"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Excl world"}).json()["id"]
    _attach(client, wid, sid)
    ids = [_upload_photo(client, sid, f"frame_{i:02d}.jpg", b"jpeg-" + f"{i}".encode() * 8)["evidence_id"]
           for i in range(3)]
    assert client.put(f"/api/evidence/{ids[0]}/decision", json={"decision": "rejected"}).status_code == 200

    job = _wait_job(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"], timeout=90.0)
    assert job["status"] == "succeeded", job.get("error")
    assert job["payload"]["cameras_registered"] == 2  # three uploaded, one rejected
    assert job["payload"]["skipped_evidence"] == []   # a decision is not a failure
