"""The Next.js API routes must proxy real Reality Engine API contracts.

The frontend bridge under `frontend/src/app/api` is a thin proxy onto
`apps/api`. A proxy that names a path the backend does not serve is the
classic way fake state creeps back in: the fetch 404s, the route falls through
to some other source, and the UI shows a world that never existed (this is
exactly how `GET /api/evidence/{id}/file` shipped -- the API serves
`/artifact`, so every image request silently missed and hit the bundled
dataset fallback).

This test holds the two sides together: for every proxied contract the
frontend depends on, the FastAPI app must expose that method + path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_API_DIR = REPO_ROOT / "frontend" / "src" / "app" / "api"

pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def api_contract() -> dict[str, set[str]]:
    """The API's own published contract: path -> methods it serves.

    Read from the OpenAPI document rather than by walking router objects:
    `app.routes` is a framework internal (and under FastAPI 0.141 the
    included routers are wrapped), while the schema is the contract clients
    are expected to code against.
    """
    import apps.api.main as main_mod

    return {
        path: {m.upper() for m in operations}
        for path, operations in main_mod.app.openapi().get("paths", {}).items()
    }


# (Next.js route file, HTTP method, backend path it proxies)
PROXIED_CONTRACTS = (
    ("worlds/route.ts", "GET", "/api/worlds"),
    ("worlds/route.ts", "POST", "/api/worlds"),
    ("worlds/[id]/worldir/route.ts", "GET", "/api/worlds/{world_id}/worldir"),
    ("worlds/[id]/points/route.ts", "GET", "/api/worlds/{world_id}/points"),
    ("worlds/[id]/cameras/route.ts", "GET", "/api/worlds/{world_id}/cameras"),
    ("worlds/[id]/report/route.ts", "GET", "/api/worlds/{world_id}/report"),
    ("worlds/[id]/versions/route.ts", "GET", "/api/worlds/{world_id}/versions"),
    ("worlds/[id]/commit/route.ts", "POST", "/api/worlds/{world_id}/commit"),
    ("worlds/[id]/diff/route.ts", "GET", "/api/worlds/{world_id}/diff"),
    ("worlds/[id]/space-graph/route.ts", "GET", "/api/worlds/{world_id}/space-graph"),
    (
        "worlds/[id]/entities/[entityId]/provenance/route.ts",
        "GET",
        "/api/worlds/{world_id}/entities/{entity_id}/provenance",
    ),
    (
        "worlds/[id]/evidence/[evidenceId]/image/route.ts",
        "GET",
        "/api/evidence/{evidence_id}/artifact",
    ),
    ("sessions/route.ts", "GET", "/api/sessions"),
    ("sessions/route.ts", "POST", "/api/sessions"),
    ("sessions/[sessionId]/reconstruct/route.ts", "POST", "/api/sessions/{session_id}/reconstruct"),
    ("jobs/[jobId]/route.ts", "GET", "/api/jobs/{job_id}"),
    ("jobs/ingest/route.ts", "POST", "/api/uploads"),
)


@pytest.mark.parametrize(
    ("route_file", "method", "backend_path"),
    PROXIED_CONTRACTS,
    ids=[f"{m} {p}" for _, m, p in PROXIED_CONTRACTS],
)
def test_frontend_bridge_proxies_a_real_api_contract(
    route_file: str, method: str, backend_path: str, api_contract: dict[str, set[str]]
):
    path = FRONTEND_API_DIR / route_file
    assert path.is_file(), f"expected the Next.js route at {path}"

    assert backend_path in api_contract, (
        f"apps/api does not expose {backend_path} - the frontend route "
        f"{route_file} proxies a path the API does not serve"
    )
    assert method in api_contract[backend_path], (
        f"apps/api serves {backend_path} with {sorted(api_contract[backend_path])}, "
        f"not {method}"
    )


def test_every_world_route_is_backed_by_the_api():
    """No world resource may be served by a hard-coded id allow-list."""
    world_routes = sorted((FRONTEND_API_DIR / "worlds").rglob("*.ts"))
    assert world_routes, "expected the worlds bridge routes to exist"
    for path in world_routes:
        source = path.read_text(encoding="utf-8")
        code = source.split("*/")[-1]  # drop the leading doc comment
        assert "startsWith(" not in code, f"{path.name} dispatches on a world-id prefix"
        assert "getLocalDatasetPath" not in code, f"{path.name} still reads a bundled dataset"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real API instance over throwaway DB + artifact storage."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))

    import importlib

    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)

    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as test_client:
        yield test_client


# The world resources the Studio viewport/proxy reads. Each must answer a
# world that has never been reconstructed with an explicit failure, never
# 200-with-invented-content.
UNCOMPILED_WORLD_RESOURCES = ("worldir", "points", "cameras", "report")


def test_uncompiled_world_resources_fail_instead_of_inventing_content(client):
    world_id = client.post("/api/worlds", json={"name": "Honest empty world"}).json()["id"]

    for resource in UNCOMPILED_WORLD_RESOURCES:
        response = client.get(f"/api/worlds/{world_id}/{resource}")
        assert response.status_code == 404, (
            f"/{resource} answered {response.status_code} for a world with no version; "
            "the UI would render that as real reconstructed data"
        )
        body = response.json()
        assert "detail" in body, f"/{resource} must explain why it has no data"

    # Lineage for a world that was never reconstructed is honestly empty.
    versions = client.get(f"/api/worlds/{world_id}/versions")
    assert versions.status_code == 200
    assert versions.json()["items"] == []

    # The world list carries only real records.
    listed = client.get("/api/worlds").json()["items"]
    assert [w["id"] for w in listed] == [world_id]


def test_evidence_image_route_target_serves_the_real_stored_bytes(client):
    session_id = client.post("/api/sessions", json={"name": "Capture"}).json()["id"]
    jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00" + bytes(64) + b"\xff\xd9"
    )
    upload = client.post(
        "/api/uploads",
        params={"session_id": session_id},
        files={"file": ("frame.jpg", jpeg, "image/jpeg")},
    )
    assert upload.status_code == 201
    evidence_id = upload.json()["evidence_id"]

    artifact = client.get(f"/api/evidence/{evidence_id}/artifact")
    assert artifact.status_code == 200
    assert artifact.content == jpeg, "the artifact must be the exact stored bytes"

    # A world that does not exist must not resolve someone else's evidence.
    missing = client.get(f"/api/evidence/ev-does-not-exist/artifact")
    assert missing.status_code == 404
