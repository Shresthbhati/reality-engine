"""The `GET /api/sessions` proxy must not turn failures into empty successes.

`frontend/src/app/api/sessions/route.ts` is the browser's door to
`apps/api/routes_sessions.py`, whose list endpoint answers `{"items": [...]}`.

It previously answered every failure with HTTP 200 and a body of
`{error, sessions: [], count: 0}`. That envelope is a fabrication twice over:
`sessions` and `count` are not keys the backend emits (so a consumer reading
`data.sessions` got `undefined` even on the success path), and an empty list
with `count: 0` is indistinguishable from a genuine "this user has captured
nothing". A total API outage therefore rendered as a plausible empty
dashboard instead of an error the operator could act on.

These tests execute the real route module against a stubbed backend via
`frontend/tests/contract/sessions-proxy.contract.mjs` and assert on observed
behaviour, rather than grepping the source for banned strings.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "frontend" / "tests" / "contract" / "sessions-proxy.contract.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to execute the Next.js route"
)


@pytest.fixture(scope="module")
def observed() -> dict:
    """Run the route under test once and return every scenario's result."""
    proc = subprocess.run(
        ["node", str(HARNESS)],
        cwd=HARNESS.parent.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "sessions proxy harness failed to run the route:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # The harness prints the JSON document first; Node may append warnings.
    stdout = proc.stdout.strip()
    end = stdout.rindex("}") + 1
    return json.loads(stdout[:end])


def assert_not_a_fabricated_empty_list(result: dict, context: str) -> None:
    """A failure may not carry a session list, a zero count, or a 200."""
    body = result.get("body") or {}
    assert "sessions" not in body, (
        f"{context}: the proxy emitted a `sessions` key, which the backend "
        f"contract does not contain -- this is the fabricated envelope: {body!r}"
    )
    assert "count" not in body, (
        f"{context}: the proxy emitted a `count` key, which the backend "
        f"contract does not contain: {body!r}"
    )
    assert result["status"] >= 400, (
        f"{context}: an outage must not answer 2xx; got {result['status']} with {body!r}"
    )
    assert body.get("detail"), f"{context}: the failure must explain itself: {body!r}"


def test_successful_backend_response_preserves_the_items_contract(observed: dict):
    result = observed["success"]
    assert result["status"] == 200
    assert result["body"] == {"items": [{"id": "s-1", "name": "Kitchen sweep"}]}
    assert "items" in result["body"], "the list contract is {items: [...]}"
    assert "sessions" not in result["body"], (
        "the proxy must not rename the backend's `items` into its own envelope"
    )
    assert result["requests"] == ["http://localhost:8100/api/sessions"]


def test_world_id_filter_is_forwarded_to_the_backend(observed: dict):
    assert observed["successWithFilter"]["requests"] == [
        "http://localhost:8100/api/sessions?world_id=w-1"
    ]


@pytest.mark.parametrize(
    ("scenario", "expected_status"),
    [
        ("backendClientError", 422),
        ("backendServerError", 500),
    ],
)
def test_backend_errors_stay_errors(observed: dict, scenario: str, expected_status: int):
    """A 4xx/5xx from the API propagates as that status and that body."""
    result = observed[scenario]
    assert result["status"] == expected_status
    assert result["body"]["detail"], "the backend's own message is preserved"
    assert_not_a_fabricated_empty_list(result, scenario)


def test_unparseable_backend_error_body_still_fails_loudly(observed: dict):
    """An HTML error page from a reverse proxy must not become a 200."""
    result = observed["backendUnparseableError"]
    assert result["status"] == 502
    assert "502" in result["body"]["detail"]
    assert_not_a_fabricated_empty_list(result, "backendUnparseableError")


@pytest.mark.parametrize("scenario", ["timeout", "unreachable"])
def test_transport_failures_answer_503_never_an_empty_success(observed: dict, scenario: str):
    """A timeout or a refused connection is a 503, not "you have no sessions"."""
    result = observed[scenario]
    assert result["status"] == 503, (
        f"{scenario}: expected 503 for an unreachable backend, got {result['status']}"
    )
    assert result["body"]["available"] is False
    assert_not_a_fabricated_empty_list(result, scenario)


def test_no_scenario_throws(observed: dict):
    """The route must always answer; an unhandled throw hides which layer broke."""
    for name, result in observed.items():
        assert not result.get("threw"), f"{name} threw instead of returning a response: {result}"
