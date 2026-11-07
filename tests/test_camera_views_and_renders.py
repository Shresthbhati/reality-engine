"""T-4b.6: Still renders (PNG) from saved camera views for client decks.

Verification:
1. Saving camera views (position, target, fov_deg, name) via POST /api/worlds/{id}/views.
2. Listing saved camera views via GET /api/worlds/{id}/views.
3. Deleting saved views via DELETE /api/worlds/{id}/views/{view_id}.
4. Rendering still PNG image for a saved view via GET /api/worlds/{id}/render?view_id=...
5. Rendering for a specific named design option (e.g. Option A).
6. Validating image/png content type, valid PNG magic header, and exact resolution.
"""

from __future__ import annotations

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from tests.test_reconstruction_lifecycle_semantics import _setup, _wait, client  # noqa: F401


def test_saved_camera_views_crud(client: TestClient, tmp_path, monkeypatch):
    """CRUD lifecycle for saved camera views."""
    sid, wid = _setup(client, tmp_path, monkeypatch)

    # 1. Create two saved views
    v1_res = client.post(
        f"/api/worlds/{wid}/views",
        json={
            "name": "Living Room Overview",
            "position": {"x": 1.5, "y": 1.6, "z": -3.0},
            "target": {"x": 0.0, "y": 1.2, "z": 1.0},
            "fov_deg": 65.0,
        },
    )
    assert v1_res.status_code == 200
    v1_data = v1_res.json()
    assert v1_data["name"] == "Living Room Overview"
    assert v1_data["fov_deg"] == 65.0
    v1_id = v1_data["id"]

    v2_res = client.post(
        f"/api/worlds/{wid}/views",
        json={
            "name": "Kitchen Eye Level",
            "position": {"x": -2.0, "y": 1.6, "z": 0.0},
            "target": {"x": 1.0, "y": 1.4, "z": 0.5},
            "fov_deg": 55.0,
        },
    )
    assert v2_res.status_code == 200
    v2_id = v2_res.json()["id"]

    # 2. List views
    list_res = client.get(f"/api/worlds/{wid}/views")
    assert list_res.status_code == 200
    items = list_res.json()["items"]
    assert len(items) == 2
    names = [v["name"] for v in items]
    assert "Living Room Overview" in names
    assert "Kitchen Eye Level" in names

    # 3. Delete one view
    del_res = client.delete(f"/api/worlds/{wid}/views/{v1_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Confirm remaining
    list_res_2 = client.get(f"/api/worlds/{wid}/views")
    assert len(list_res_2.json()["items"]) == 1
    assert list_res_2.json()["items"][0]["id"] == v2_id


def test_render_endpoint_returns_png_for_saved_view(client: TestClient, tmp_path, monkeypatch):
    """T-4b.6: Render endpoint returns high-resolution still PNG for a saved camera view."""
    sid, wid = _setup(client, tmp_path, monkeypatch)

    # 1. Run reconstruction so world has a version and entities
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    _wait(client, r.json()["job_id"])

    # 2. Save a client deck camera view
    view_res = client.post(
        f"/api/worlds/{wid}/views",
        json={
            "name": "Deck Hero Perspective",
            "position": {"x": 0.0, "y": 1.6, "z": -4.0},
            "target": {"x": 0.0, "y": 1.2, "z": 0.0},
            "fov_deg": 60.0,
        },
    )
    assert view_res.status_code == 200
    view_id = view_res.json()["id"]

    # 3. Request still render for saved view
    render_res = client.get(
        f"/api/worlds/{wid}/render",
        params={"view_id": view_id, "width": 1280, "height": 720},
    )
    assert render_res.status_code == 200
    assert render_res.headers["content-type"] == "image/png"
    assert "filename=" in render_res.headers.get("content-disposition", "")

    # Validate raw bytes: PNG magic signature
    png_bytes = render_res.content
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Rendered output must be a valid PNG image"

    # Decode with Pillow to confirm exact dimensions and validity
    img = Image.open(io.BytesIO(png_bytes))
    assert img.format == "PNG"
    assert img.size == (1280, 720)
    assert img.mode == "RGB"

    # 4. Request still render for a named design option
    opt_res = client.post(
        f"/api/worlds/{wid}/options",
        json={
            "label": "Option A - Chantilly Paint",
            "base_version": "as-built",
        },
    )
    assert opt_res.status_code == 200

    render_opt = client.get(
        f"/api/worlds/{wid}/render",
        params={"view_id": view_id, "version": "Option A - Chantilly Paint", "width": 640, "height": 360},
    )
    assert render_opt.status_code == 200
    assert render_opt.headers["content-type"] == "image/png"
    img_opt = Image.open(io.BytesIO(render_opt.content))
    assert img_opt.size == (640, 360)
