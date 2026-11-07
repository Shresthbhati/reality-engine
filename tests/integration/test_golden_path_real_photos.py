"""REAL golden path: South Building photographs -> real COLMAP -> world -> exported GLB that validates (plan task T-1.3).

Nothing is mocked. This is the capability version of tests/test_golden_path_api_to_glb.py (which is plumbing only).
Skips when the dataset or the real engines are absent. Run: python -m pytest -q tests/integration/test_golden_path_real_photos.py
(several minutes on CPU).
"""

from __future__ import annotations

import time

import pytest

from tests.gltf_validation import validate_glb
from tests.integration.test_progressive_product_journey import IMGS, SIX, _OK, _WHY, _post, app_client

pytest.importorskip("fastapi")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.e2e,
    pytest.mark.skipif(len(IMGS) < 25, reason="BLOCKED: South Building dataset not present"),
    pytest.mark.skipif(not _OK, reason=f"BLOCKED: real engines required ({_WHY})"),
]

TERMINAL = ("succeeded", "partial", "failed", "cancelled")


def _wait_job(c, job_id: str, timeout: float = 1500.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in TERMINAL:
            return j
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_real_photos_to_validated_glb(tmp_path):
    with app_client(tmp_path) as c:
        body = _post(c, SIX)
        wid = body["world_id"]
        job = _wait_job(c, body["job_id"])
        assert job["status"] == "succeeded", (job["status"], job.get("error"))

        world = c.get(f"/api/worlds/{wid}/worldir").json()
        assert world["entities"], "reconstruction produced no entities"

        r = c.post(f"/api/worlds/{wid}/export", json={"format": "glb"})
        assert r.status_code == 200, r.text
        exp = r.json()
        glb = c.get(exp["download_url"]).content

        # the GLB is a well-formed file ...
        summary = validate_glb(glb)
        # ... that actually carries the reconstruction, not an empty shell
        assert exp["entities_exported"], f"nothing exported; skipped: {exp['skip_reasons']}"
        assert set(exp["entities_exported"]) <= set(world["entities"])
        assert summary["vertices"] > 100 and summary["meshes"] >= 1, summary
        print(f"\nREAL GLB: {len(glb)} bytes, {summary}, exported {len(exp['entities_exported'])} "
              f"of {len(world['entities'])} entities, skipped {len(exp['entities_skipped'])}")
