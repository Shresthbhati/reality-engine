"""T-4b.4: Read-only client share link (token-scoped, revocable, no account needed).

Security verification:
1. An unauthenticated client with a valid share token can view the shared world.
2. The token is strictly scoped: attempting to access or view any OTHER world is forbidden (403).
3. The token is strictly read-only: attempting to commit edits, mutate, or delete the world is forbidden (403).
4. Revocation is instant: revoking the link immediately denies all subsequent access (401/403).
5. Expiration is enforced: an expired token is refused (401).
"""

from __future__ import annotations

import time
import pytest

from fastapi.testclient import TestClient
from tests.test_reconstruction_lifecycle_semantics import _setup, _wait, client  # noqa: F401
from apps.api.auth import issue_share_token, verify_share_token, Role


def test_issue_and_verify_share_token_pure_crypto():
    """Verify cryptographically signed share tokens without database dependencies."""
    token = issue_share_token("wld_test1", "sh_123")
    assert token.startswith("sh1:wld_test1:sh_123:0:")

    ctx = verify_share_token(token)
    assert ctx is not None
    assert ctx.role == Role.VIEWER
    assert ctx.project_id == "wld_test1"
    assert ctx.user_id == "client:sh_123"

    # Expired token is rejected
    past = int(time.time()) - 100
    expired_token = issue_share_token("wld_test1", "sh_exp", expires_at=past)
    assert verify_share_token(expired_token) is None

    # Tampered token is rejected
    tampered = token[:-4] + "ffff"
    assert verify_share_token(tampered) is None


def test_client_share_link_lifecycle_and_read_only_access(client: TestClient, tmp_path, monkeypatch):
    """End-to-end client share link: creation, listing, unauthenticated view, and revocation."""
    sid_a, wid_a = _setup(client, tmp_path, monkeypatch)

    # Reconstruct World A so it has a version
    r = client.post(f"/api/sessions/{sid_a}/reconstruct")
    assert r.status_code == 200
    _wait(client, r.json()["job_id"])

    # 1. Create a share link for World A
    share_res = client.post(
        f"/api/worlds/{wid_a}/shares",
        json={"label": "Client Presentation A", "expires_in_hours": 48},
    )
    assert share_res.status_code == 200, share_res.text
    share_data = share_res.json()
    assert share_data["world_id"] == wid_a
    assert share_data["token"].startswith("sh1:")
    assert share_data["label"] == "Client Presentation A"
    token_a = share_data["token"]
    share_id = share_data["id"]

    # 2. List share links
    list_res = client.get(f"/api/worlds/{wid_a}/shares")
    assert list_res.status_code == 200
    items = list_res.json()["items"]
    assert any(it["id"] == share_id and it["is_active"] is True for it in items)

    # 3. Access shared world via public endpoint (no login / no account needed)
    public_res = client.get(f"/api/worlds/shared/access/{token_a}")
    assert public_res.status_code == 200, public_res.text
    public_data = public_res.json()
    assert public_data["world_id"] == wid_a
    assert public_data["read_only"] is True
    assert public_data["worldir"] is not None
    assert public_data["worldir"]["entities"]

    # 4. Access world endpoints via Bearer token
    headers_a = {"Authorization": f"Bearer {token_a}"}
    world_res = client.get(f"/api/worlds/{wid_a}/worldir", headers=headers_a)
    assert world_res.status_code == 200
    assert world_res.json()["entities"]

    # Also supports query parameter ?share_token=
    param_res = client.get(f"/api/worlds/{wid_a}/worldir?share_token={token_a}")
    assert param_res.status_code == 200


def test_security_share_link_cannot_see_other_worlds(client: TestClient):
    """Security check: a share token for World A CANNOT view or access World B."""
    wid_a = client.post("/api/worlds", json={"name": "World A"}).json()["id"]
    wid_b = client.post("/api/worlds", json={"name": "World B"}).json()["id"]

    # Create share link for World A only
    share_res = client.post(f"/api/worlds/{wid_a}/shares", json={"label": "Deck A"})
    token_a = share_res.json()["token"]

    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Attempt to view World B with World A's share token: MUST BE FORBIDDEN (403)
    resp_b_details = client.get(f"/api/worlds/{wid_b}", headers=headers_a)
    assert resp_b_details.status_code == 403, (
        f"Cross-world access must be forbidden, got status {resp_b_details.status_code}: {resp_b_details.text}"
    )

    resp_b_ir = client.get(f"/api/worlds/{wid_b}/worldir", headers=headers_a)
    assert resp_b_ir.status_code == 403, (
        f"Cross-world access must be forbidden, got status {resp_b_ir.status_code}: {resp_b_ir.text}"
    )


def test_security_share_link_cannot_edit_or_delete(client: TestClient):
    """Security check: a share token is strictly read-only and CANNOT mutate the world."""
    wid_a = client.post("/api/worlds", json={"name": "World A"}).json()["id"]

    share_res = client.post(f"/api/worlds/{wid_a}/shares", json={"label": "Client Readonly"})
    token_a = share_res.json()["token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Try commit / edit
    commit_attempt = client.post(
        f"/api/worlds/{wid_a}/commit",
        json={"entity_id": "dummy", "changes": {"name": "Hacked Name"}},
        headers=headers_a,
    )
    assert commit_attempt.status_code == 403, (
        f"Mutating edit with share token must return 403, got {commit_attempt.status_code}"
    )

    # Try delete
    delete_attempt = client.delete(f"/api/worlds/{wid_a}", headers=headers_a)
    assert delete_attempt.status_code == 403, (
        f"Delete with share token must return 403, got {delete_attempt.status_code}"
    )


def test_security_revocation_immediately_denies_access(client: TestClient):
    """Security check: once revoked, all subsequent access with the share token is denied."""
    wid_a = client.post("/api/worlds", json={"name": "World A"}).json()["id"]

    share_res = client.post(f"/api/worlds/{wid_a}/shares", json={"label": "Revocable Link"})
    token_a = share_res.json()["token"]
    share_id = share_res.json()["id"]

    # Verify access works prior to revocation
    assert client.get(f"/api/worlds/shared/access/{token_a}").status_code == 200

    # Revoke link
    revoke_res = client.post(f"/api/worlds/{wid_a}/shares/{share_id}/revoke")
    assert revoke_res.status_code == 200
    assert revoke_res.json()["status"] == "revoked"

    # Subsequent access MUST fail with 401/403
    after_public = client.get(f"/api/worlds/shared/access/{token_a}")
    assert after_public.status_code in (401, 403), f"Expected 401 or 403, got {after_public.status_code}"

    after_bearer = client.get(f"/api/worlds/{wid_a}/worldir", headers={"Authorization": f"Bearer {token_a}"})
    assert after_bearer.status_code in (401, 403), f"Expected 401 or 403, got {after_bearer.status_code}"
