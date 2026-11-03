"""Unit and integration tests for the /api/technologies REST endpoints."""

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_list_technologies_endpoint():
    """Verify GET /api/technologies returns the 50 technologies catalog and summary stats."""
    res = client.get("/api/technologies")
    assert res.status_code == 200
    data = res.json()

    assert data["total_count"] == 50
    assert "summary" in data
    assert data["summary"]["total_technologies"] == 50
    assert len(data["summary"]["subsystems"]) == 7

    # Verify presence of key candidates
    item_ids = {it["id"] for it in data["items"]}
    assert "yolox" in item_ids
    assert "teaser_plus_plus" in item_ids
    assert "manifold" in item_ids
    assert "gtsam" in item_ids
    assert "pcl" in item_ids
    assert "onnx_runtime" in item_ids


def test_filter_technologies_by_subsystem():
    """Verify filtering technologies by subsystem enum."""
    res = client.get("/api/technologies?subsystem=geometric_estimation")
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] == 7
    for it in data["items"]:
        assert it["subsystem"] == "geometric_estimation"


def test_get_single_technology():
    """Verify GET /api/technologies/{tech_id} returns detailed descriptor."""
    res = client.get("/api/technologies/manifold")
    assert res.status_code == 200
    data = res.json()

    assert data["id"] == "manifold"
    assert data["name"] == "Manifold"
    assert data["subsystem"] == "geometry_cad_mesh"
    assert data["license"] == "Apache-2.0"
    assert "discovery" in data
    assert data["discovery"]["has_adapter"] is True


def test_run_benchmark_single_technology():
    """Verify POST /api/technologies/benchmark evaluates single technology."""
    res = client.post("/api/technologies/benchmark", json={"tech_id": "teaser_plus_plus"})
    assert res.status_code == 200
    data = res.json()

    assert data["tech_id"] == "teaser_plus_plus"
    assert data["name"] == "TEASER++"
    assert "registration_rmse_m" in data["metrics"]
    assert data["summary_verdict"] == "RECOMMENDED"
    assert data["vram_peak_mb"] <= 6144.0  # RTX 4050 6GB guarantee


def test_run_benchmark_all_technologies():
    """Verify POST /api/technologies/benchmark evaluates all 50 technologies."""
    res = client.post("/api/technologies/benchmark", json={})
    assert res.status_code == 200
    data = res.json()

    assert data["total_evaluated"] == 50
    assert data["recommended_count"] >= 10
    assert len(data["results"]) == 50
