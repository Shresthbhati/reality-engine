"""Contract test verifying all Reality Studio client endpoints exist in apps/api."""
from __future__ import annotations

from apps.api.main import app


def test_studio_client_endpoints_exist_in_openapi() -> None:
    schema = app.openapi()
    paths = schema.get("paths", {})

    expected = [
        ("/api/worlds", "get"),
        ("/api/worlds", "post"),
        ("/api/reconstructions", "post"),
        ("/api/worlds/{world_id}/status", "get"),
        ("/api/worlds/{world_id}/worldir", "get"),
        ("/api/worlds/{world_id}/versions", "get"),
        ("/api/worlds/{world_id}/points", "get"),
        ("/api/worlds/{world_id}/cameras", "get"),
        ("/api/worlds/{world_id}/quality", "get"),
        ("/api/worlds/{world_id}/coverage", "get"),
        ("/api/worlds/{world_id}/entities/{entity_id}/provenance", "get"),
        ("/api/worlds/{world_id}/entities/{entity_id}/confidence", "get"),
        ("/api/worlds/{world_id}/corrections/reclassify", "post"),
        ("/api/worlds/{world_id}/corrections/split", "post"),
        ("/api/worlds/{world_id}/corrections/merge", "post"),
        ("/api/worlds/{world_id}/rollback", "post"),
        ("/api/worlds/{world_id}/diff", "get"),
        ("/api/worlds/{world_id}/export/formats", "get"),
        ("/api/worlds/{world_id}/export", "post"),
    ]
    missing = [
        f"{method.upper()} {path}"
        for path, method in expected
        if path not in paths or method not in paths[path]
    ]
    assert not missing, f"Missing API routes expected by Reality Studio: {missing}"
