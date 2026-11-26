"""Upload staging path security (Task 5: path traversal prevention).

The staging filename is derived from a caller-supplied upload id. The storage
layer must reject anything that could escape staging_root(), even if an
upstream route failed to validate. This is defense in depth: DB visibility
checks still apply, but they are not the only barrier.
"""
from __future__ import annotations

import pytest

from apps.api.storage import staging_path, validate_staging_id


@pytest.mark.parametrize(
    "bad",
    [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "a/b",
        "a\\b",
        ".",
        "..",
        "",
        "a\x00b",
        "%2e%2e%2fsecret",
        "upl_abc/../../x",
        "x" * 65,
        12345,
        None,
    ],
)
def test_validate_staging_id_rejects_traversal(bad):
    with pytest.raises(ValueError, match="invalid upload id|must be a string"):
        validate_staging_id(bad)


@pytest.mark.parametrize("good", ["upl_abcdef123456", "upl_0123456789ab", "ab_-CD09"])
def test_validate_staging_id_accepts_issued_shape(good):
    assert validate_staging_id(good) == good


def test_staging_path_is_confined_to_staging_root(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    import importlib

    import apps.api.storage as storage

    importlib.reload(storage)
    p = storage.staging_path("upl_abcdef123456")
    assert p.parent == storage.staging_root()
    assert p.name == "upl_abcdef123456.partial"
    # A traversal attempt must never produce a path outside staging_root.
    with pytest.raises(ValueError):
        storage.staging_path("../../outside")


def test_stage_chunk_rejects_traversal_without_writing(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    import importlib

    import apps.api.storage as storage

    importlib.reload(storage)

    with pytest.raises(ValueError):
        storage.stage_chunk("../escape", 0, b"data")
    # Nothing escaped into the parent of the artifacts dir.
    assert not (tmp_path / "escape.partial").exists()


def test_finalize_requires_owner_auth(tmp_path, monkeypatch):
    """resumable_finalize must not let a foreign tenant finalize an upload.

    Regression test for the previously missing authorization check. Tenant A
    starts and stages a pending upload; tenant B (same role) must get 404 and
    the upload must remain pending. Tenant A can then finalize it.
    """
    import hashlib
    import importlib

    from fastapi.testclient import TestClient

    from apps.api.auth import Role, issue_scoped_token, rate_limiter

    secret = "upload-auth-test-secret"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_SIGNING_KEY", secret)
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    rate_limiter.reset()

    def hdr(tenant):
        return {"Authorization": "Bearer " + issue_scoped_token(
            tenant, f"user-{tenant}", Role.EDITOR, secret=secret)}

    with TestClient(main_mod.app) as c:
        # Tenant A creates a session and a pending resumable upload.
        s = c.post("/api/sessions", json={"name": "s"}, headers=hdr("tenant-a")).json()["id"]
        r = c.post("/api/uploads/resumable/start",
                   data={"filename": "a.jpg", "mime_type": "image/jpeg", "total_size": "8", "session_id": s},
                   headers=hdr("tenant-a"))
        assert r.status_code == 201, r.text
        upload_id = r.json()["upload_id"]
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 4
        r = c.post("/api/uploads/resumable/chunk",
                   data={"upload_id": upload_id, "offset": "0", "total_size": "8"},
                   files={"file": ("p", data, "image/jpeg")}, headers=hdr("tenant-a"))
        assert r.status_code == 200, r.text

        # Tenant B cannot finalize A's upload.
        foreign = c.post(f"/api/uploads/resumable/{upload_id}/finalize", headers=hdr("tenant-b"))
        assert foreign.status_code == 404, foreign.text

        # Owner still can.
        ok = c.post(f"/api/uploads/resumable/{upload_id}/finalize", headers=hdr("tenant-a"))
        assert ok.status_code == 201, ok.text
        assert ok.json()["checksum"] == hashlib.sha256(data).hexdigest()
    rate_limiter.reset()
