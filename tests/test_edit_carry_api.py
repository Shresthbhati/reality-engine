"""T-4.2 end to end: edit -> reprocess -> the edit survives (rule 18)."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from tests.test_commit_validation_contract import _commit, world  # noqa: E402,F401  (fixture reuse)


def _reprocess(c, wid):
    versions = c.get(f"/api/worlds/{wid}/versions").json()
    versions = versions["items"]
    sid = next(v["source_session_ids"][0] for v in versions if v.get("source_session_ids"))
    jid = c.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
    for _ in range(400):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
            return job
        time.sleep(0.25)
    raise AssertionError("reprocess did not finish")


def test_a_manual_rename_and_property_survive_a_reprocess(world):
    c, wid, eid = world
    r = _commit(c, wid, {"entity_id": eid, "changes": {"name": "Party wall (checked on site)",
                                                       "custom_properties": {"site_note": "load bearing"}}})
    assert r.status_code == 200, r.text
    stored = c.get(f"/api/worlds/{wid}/worldir").json()
    assert {e["field"] for e in stored["metadata"]["user_edits"]} >= {"name", "custom_properties.site_note"}

    job = _reprocess(c, wid)
    assert job["status"] in ("succeeded", "partial"), job
    after = c.get(f"/api/worlds/{wid}/worldir").json()
    assert after["metadata"].get("edit_carry", {}).get("applied", 0) >= 2
    e = after["entities"][eid]
    assert e["name"] == "Party wall (checked on site)" and e["custom_properties"]["site_note"] == "load bearing"
