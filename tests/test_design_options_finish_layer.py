"""T-4b.2 & T-4b.3: Surface finish design layers and named design options.

Verification:
1. T-4b.2: Surface finish edits (wall paint color, floor material) are stored as a derived design layer.
   The initial as-built survey is NEVER overwritten: exporting the as-built survey after edits produces
   the exact same artifact hash and content.
2. T-4b.3: Named design options are saved world versions with plain-language labels ("Option A", "Option B").
   Comparing as-built vs option surfaces detailed finish and geometry diffs.
   Exporting options side-by-side exports both versions simultaneously.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.test_reconstruction_lifecycle_semantics import _setup, _wait, client  # noqa: F401


def test_as_built_export_unchanged_after_surface_finish_edits(client: TestClient, tmp_path, monkeypatch):
    """T-4b.2: Change surface finish as a derived design layer; verify as-built export is unchanged."""
    sid, wid = _setup(client, tmp_path, monkeypatch)

    # 1. Run reconstruction to create initial as-built survey version
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    _wait(client, r.json()["job_id"])

    # 2. Get WorldIR to find an entity to edit
    wir_res = client.get(f"/api/worlds/{wid}/worldir")
    assert wir_res.status_code == 200
    entities = wir_res.json()["entities"]
    assert entities, "At least one entity must exist in reconstructed world"
    target_id = list(entities.keys())[0]

    # 3. Export as-built survey prior to any edits
    exp_as_built_1 = client.post(f"/api/worlds/{wid}/export", json={"format": "gltf", "version": "as-built"})
    assert exp_as_built_1.status_code == 200
    hash_as_built_1 = exp_as_built_1.json()["content_hash"]
    download_url_1 = exp_as_built_1.json()["download_url"]

    dl_res_1 = client.get(download_url_1)
    assert dl_res_1.status_code == 200
    content_as_built_1 = dl_res_1.content

    # 4. Commit a surface finish edit on the entity as a derived design layer
    finish_change = {
        "surface_finish": {
            "surface_finish": "smooth",
            "color_rgb": [0.92, 0.92, 0.90],
            "name": "Chantilly Lace Paint",
            "class_name": "paint",
        },
        "design_layer": "finish",
    }
    commit_res = client.post(
        f"/api/worlds/{wid}/commit",
        json={
            "entity_id": target_id,
            "changes": finish_change,
            "commit_message": "Apply smooth Chantilly Lace paint to accent surface",
            "label": "Option A - Paint Finish",
        },
    )
    assert commit_res.status_code == 200
    new_v_id = commit_res.json()["version_id"]

    # Verify new version has the updated surface finish with GENERATED provenance
    wir_v2 = client.get(f"/api/worlds/{wid}/worldir?version={new_v_id}").json()
    edited_entity = wir_v2["entities"][target_id]
    assert edited_entity["custom_properties"]["surface_finish"] == "smooth"
    assert edited_entity["custom_properties"]["material"] == "paint"
    assert edited_entity["custom_properties"]["design_layer"] == "finish"

    # 5. Export the new version and verify it has the new export hash
    exp_option_a = client.post(f"/api/worlds/{wid}/export", json={"format": "gltf", "version": new_v_id})
    assert exp_option_a.status_code == 200
    assert exp_option_a.json()["version_id"] == new_v_id

    # 6. CRITICAL T-4b.2 REQUIREMENT: Re-export as-built survey; must be EXACTLY IDENTICAL to step 3!
    exp_as_built_2 = client.post(f"/api/worlds/{wid}/export", json={"format": "gltf", "version": "as-built"})
    assert exp_as_built_2.status_code == 200
    hash_as_built_2 = exp_as_built_2.json()["content_hash"]
    download_url_2 = exp_as_built_2.json()["download_url"]

    dl_res_2 = client.get(download_url_2)
    assert dl_res_2.status_code == 200
    content_as_built_2 = dl_res_2.content

    assert hash_as_built_1 == hash_as_built_2, "As-built export hash must be completely unchanged after edits"
    assert content_as_built_1 == content_as_built_2, "As-built exported bytes must be identical before and after edits"


def test_named_design_options_lifecycle_and_side_by_side_export(client: TestClient, tmp_path, monkeypatch):
    """T-4b.3: Plain-language design options, comparison, and side-by-side export."""
    sid, wid = _setup(client, tmp_path, monkeypatch)

    # 1. Run reconstruction
    r = client.post(f"/api/sessions/{sid}/reconstruct")
    assert r.status_code == 200
    _wait(client, r.json()["job_id"])

    wir = client.get(f"/api/worlds/{wid}/worldir").json()
    entities = wir["entities"]
    assert entities, "At least one entity must exist in reconstructed world"
    target_id = list(entities.keys())[0]

    # 2. Create Named Design Option A via POST /api/worlds/{id}/options
    opt_a_res = client.post(
        f"/api/worlds/{wid}/options",
        json={
            "label": "Option A - Warm Minimalist",
            "base_version": "as-built",
            "surface_finishes": [
                {
                    "entity_id": target_id,
                    "surface_finish": "smooth",
                    "color_rgb": [0.95, 0.93, 0.90],
                    "name": "Warm Linen",
                    "class_name": "paint",
                }
            ],
        },
    )
    assert opt_a_res.status_code == 200
    opt_a_vid = opt_a_res.json()["version_id"]
    assert opt_a_res.json()["label"] == "Option A - Warm Minimalist"

    # 3. Create Named Design Option B via POST /api/worlds/{id}/options
    opt_b_res = client.post(
        f"/api/worlds/{wid}/options",
        json={
            "label": "Option B - Industrial Charcoal",
            "base_version": "as-built",
            "surface_finishes": [
                {
                    "entity_id": target_id,
                    "surface_finish": "rough",
                    "color_rgb": [0.2, 0.2, 0.22],
                    "name": "Charcoal Concrete",
                    "class_name": "concrete",
                }
            ],
        },
    )
    assert opt_b_res.status_code == 200
    opt_b_vid = opt_b_res.json()["version_id"]
    assert opt_b_res.json()["label"] == "Option B - Industrial Charcoal"

    # 4. List options via GET /api/worlds/{id}/options
    opts_list = client.get(f"/api/worlds/{wid}/options").json()
    labels = [opt["label"] for opt in opts_list["options"]]
    assert "as-built" in labels or any(opt.get("is_as_built") for opt in opts_list["options"])
    assert "Option A - Warm Minimalist" in labels
    assert "Option B - Industrial Charcoal" in labels

    # 5. Rename Option A label via PATCH /api/worlds/{id}/versions/{version_id}/label
    patch_lbl = client.patch(
        f"/api/worlds/{wid}/versions/{opt_a_vid}/label",
        json={"label": "Option A - Nordic White & Oak"},
    )
    assert patch_lbl.status_code == 200
    assert patch_lbl.json()["label"] == "Option A - Nordic White & Oak"

    # 6. Compare as-built vs Option A via GET /api/worlds/{id}/options/compare
    cmp_res = client.get(
        f"/api/worlds/{wid}/options/compare",
        params={"base": "as-built", "target": "Option A - Nordic White & Oak"},
    )
    assert cmp_res.status_code == 200
    cmp_data = cmp_res.json()
    assert cmp_data["base"]["label"] in ("as-built", None) or cmp_data["base"]["version_id"]
    assert cmp_data["target"]["label"] == "Option A - Nordic White & Oak"
    assert cmp_data["summary"]["finishes_changed_count"] >= 1
    diff = next(d for d in cmp_data["finish_differences"] if d["entity_id"] == target_id)
    assert diff["target"]["surface_finish"] == "smooth"
    assert diff["target"]["material"] == "paint"

    # 7. Side-by-side export via POST /api/worlds/{id}/export/compare
    side_res = client.post(
        f"/api/worlds/{wid}/export/compare",
        json={
            "format": "gltf",
            "version_a": "as-built",
            "version_b": "Option A - Nordic White & Oak",
        },
    )
    assert side_res.status_code == 200
    side_data = side_res.json()
    assert side_data["side_by_side"] is True
    assert "option_a" in side_data
    assert "option_b" in side_data
    assert side_data["option_a"]["download_url"].startswith("/api/worlds/")
    assert side_data["option_b"]["download_url"].startswith("/api/worlds/")
