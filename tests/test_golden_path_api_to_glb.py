"""Golden path through the HTTP API, ending in a GLB that validates (plan task T-1.3).

    create session + world -> attach -> upload photos -> reconstruct (job) -> world version -> export GLB -> download -> validate

PLUMBING TEST, NOT A CAPABILITY CLAIM. The reconstruction step uses the sanctioned `REALITY_TEST_BACKEND` seam (a
deterministic two-view backend, no pixels read), so this proves the product path is connected and the export is a
well-formed GLB, and it runs on any CPU runner. It proves nothing about reconstruction quality on real photographs: that is
`tests/integration/test_release_gate.py` and `tests/integration/test_golden_path_real_photos.py`.
"""

from __future__ import annotations

import struct

import pytest

pytest.importorskip("fastapi")

from tests.gltf_validation import GlbError, parse_glb, validate_glb  # noqa: E402
from tests.test_reconstruction_lifecycle_semantics import _setup, _wait, client  # noqa: E402,F401


def _export_glb(client, wid: str) -> tuple[dict, bytes]:
    r = client.post(f"/api/worlds/{wid}/export", json={"format": "glb"})
    assert r.status_code == 200, r.text
    body = r.json()
    d = client.get(body["download_url"])
    assert d.status_code == 200, d.text
    return body, d.content


def test_photos_to_validated_glb_through_the_api(client, tmp_path, monkeypatch):
    sid, wid = _setup(client, tmp_path, monkeypatch)

    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200, r.text
    job = _wait(client, r.json()["job_id"])
    assert job["status"] in ("succeeded", "partial"), job

    versions = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    assert versions, "a finished job must leave a world version to export"
    world = client.get(f"/api/worlds/{wid}/worldir").json()
    assert world["entities"], "the world has no entities"

    body, glb = _export_glb(client, wid)
    assert body["entities_exported"], f"nothing was exported; skipped: {body['skip_reasons']}"
    # every entity the exporter says it wrote is a real entity of the stored world
    assert set(body["entities_exported"]) <= set(world["entities"])

    summary = validate_glb(glb)
    assert summary["meshes"] >= 1 and summary["vertices"] >= 3, summary

    # content-addressed: the same version exports to byte-identical bytes
    body2, glb2 = _export_glb(client, wid)
    assert body2["content_hash"] == body["content_hash"] and glb2 == glb


def test_export_before_any_version_is_refused_not_an_empty_file(client, tmp_path, monkeypatch):
    _, wid = _setup(client, tmp_path, monkeypatch)  # photos uploaded, never reconstructed
    r = client.post(f"/api/worlds/{wid}/export", json={"format": "glb"})
    assert r.status_code in (404, 409, 422), r.text


# --- the validator itself: it must reject broken files, or the test above proves nothing -----------------------------


def _good_glb(client, tmp_path, monkeypatch) -> bytes:
    sid, wid = _setup(client, tmp_path, monkeypatch)
    _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    return _export_glb(client, wid)[1]


def test_validator_rejects_broken_glbs(client, tmp_path, monkeypatch):
    good = _good_glb(client, tmp_path, monkeypatch)
    validate_glb(good)

    with pytest.raises(GlbError, match="magic"):
        validate_glb(b"XXXX" + good[4:])
    with pytest.raises(GlbError, match="length"):
        validate_glb(good[:-8])                      # truncated file: header length no longer matches
    with pytest.raises(GlbError):
        validate_glb(good + b"\x00" * 4)             # trailing bytes

    doc, binary = parse_glb(good)
    # an accessor that points past its bufferView must be caught
    import json

    bad = json.loads(json.dumps(doc))
    bad["accessors"][0]["count"] += 10_000
    payload = json.dumps(bad).encode("utf-8")
    payload += b" " * (-len(payload) % 4)
    chunk_json = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    chunk_bin = struct.pack("<II", len(binary), 0x004E4942) + binary
    total = 12 + len(chunk_json) + len(chunk_bin)
    forged = struct.pack("<4sII", b"glTF", 2, total) + chunk_json + chunk_bin
    with pytest.raises(GlbError, match="runs past"):
        validate_glb(forged)


def test_export_reports_per_element_evidence_state_and_confidence(client, tmp_path, monkeypatch):
    """T-2.7: what was measured vs inferred, per exported element, with confidence."""
    sid, wid = _setup(client, tmp_path, monkeypatch)
    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] in ("succeeded", "partial"), job
    world = client.get(f"/api/worlds/{wid}/worldir").json()
    body, _ = _export_glb(client, wid)

    rows = body["elements"]
    assert [r["id"] for r in rows] == body["entities_exported"]  # one row per exported element, same order
    for r in rows:
        ent = world["entities"][r["id"]]
        assert r["type"] == ent["type"] and r["state"] in ("measured", "inferred", "estimated", "unknown")
        assert r["confidence"] is None or 0.0 <= r["confidence"] <= 1.0
    assert body["evidence_summary"]["total"] == len(rows)
    assert sum(body["evidence_summary"]["by_state"].values()) == len(rows)
