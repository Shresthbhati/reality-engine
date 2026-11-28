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
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"
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


# ---------------------------------------------------------------------------
# The same invariant, applied to every proxy route rather than just sessions.
# ---------------------------------------------------------------------------

ADVERSARIAL_HARNESS = (
    REPO_ROOT / "frontend" / "tests" / "contract" / "adversarial-routes.contract.mjs"
)


@pytest.fixture(scope="module")
def adversarial() -> dict:
    """Run the whole-surface adversarial sweep once."""
    proc = subprocess.run(
        ["node", str(ADVERSARIAL_HARNESS)],
        cwd=ADVERSARIAL_HARNESS.parent.parent,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        "adversarial route harness failed to run:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    stdout = proc.stdout.strip()
    end = stdout.rindex("}") + 1
    return json.loads(stdout[:end])


def test_no_proxy_route_turns_a_failure_into_an_empty_success(adversarial: dict):
    """Every route, every failure mode: never a 2xx, never a 404 from a timeout.

    This is the general form of the sessions bug. A proxy that answers 200 (or
    404) for a backend failure makes an outage indistinguishable from a
    genuinely empty resource, which is exactly the fabrication the UI must
    never present.
    """
    violations = adversarial.get("violations") or []
    assert not violations, (
        "proxy routes converted backend/transport failures into successful or "
        "not-found responses:\n" + "\n".join(violations)
    )


def test_adversarial_sweep_actually_exercised_the_routes(adversarial: dict):
    """Guard against the harness silently importing nothing and passing."""
    results = adversarial.get("results") or {}
    assert len(results) >= 10, f"expected the sweep to cover the GET surface, got {len(results)}"
    for route, entry in results.items():
        assert "success" in entry, f"{route}: success path was never probed"
        for mode in ("backend500", "backend503", "timeout", "unreachable"):
            assert mode in entry, f"{route}: {mode} failure mode was never probed"


# ---------------------------------------------------------------------------
# POST routes: a write that fabricates success is worse than a read that does,
# because it tells the operator a correction/version/job was persisted.
# ---------------------------------------------------------------------------

POST_HARNESS = (
    REPO_ROOT / "frontend" / "tests" / "contract" / "adversarial-post.contract.mjs"
)


@pytest.fixture(scope="module")
def post_adversarial() -> dict:
    proc = subprocess.run(
        ["node", str(POST_HARNESS)],
        cwd=POST_HARNESS.parent.parent,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        "adversarial POST harness failed to run:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    stdout = proc.stdout.strip()
    end = stdout.rindex("}") + 1
    return json.loads(stdout[:end])


def test_no_post_route_fabricates_a_write(post_adversarial: dict):
    """No POST route may turn a failure, or an unusable 2xx, into a success.

    Covers: transport failures never 2xx; a 404 stays a 404 (not a transport
    error); a 409 conflict stays a 409; and a 2xx that does not carry the id
    that proves the write happened (version_id / job_id / world id) is a
    failure, not a success.
    """
    violations = post_adversarial.get("violations") or []
    assert not violations, (
        "POST routes fabricated writes or lost their error contract:\n"
        + "\n".join(violations)
    )


def test_post_sweep_covered_every_write_route(post_adversarial: dict):
    results = post_adversarial.get("results") or {}
    for expected in (
        "sessions/[sessionId]/reconstruct/route.ts",
        "worlds/[id]/commit/route.ts",
        "worlds/route.ts",
        "jobs/ingest/route.ts",
        "jobs/compile/route.ts",
        "sessions/mobile-tasks/route.ts",
        "sessions/ingest-mobile/route.ts",
        "jobs/[jobId]/cancel/route.ts",
    ):
        assert expected in results, f"{expected}: never exercised by the POST harness"
        entry = results[expected]
        for mode in (
            "malformedRequest",
            "backend404",
            "backend409",
            "backend500",
            "backend503",
            "malformedSuccess",
            "emptySuccess",
            "timeout",
            "unreachable",
        ):
            assert mode in entry, f"{expected}: {mode} was never probed"


def test_reconstruction_success_requires_a_persisted_version(post_adversarial: dict):
    """The exact success rule, now enforced by `reconstruction-rule.contract.mjs`.

    The rule itself lives in ONE place (`lib/api/reconstruction.ts`) and is
    tested by importing that module directly, so the UI and the tests cannot
    disagree. The full status x version_id matrix is asserted in
    `test_canonical_reconstruction_rule_matches_the_spec_table`; this check
    only confirms the POST sweep still runs the route surface.
    """
    results = post_adversarial.get("results") or {}
    assert "worlds/[id]/commit/route.ts" in results, (
        "the commit route must still be exercised by the POST sweep"
    )


CANCEL_HARNESS = REPO_ROOT / "frontend" / "tests" / "contract" / "cancel-contract.contract.mjs"


@pytest.fixture(scope="module")
def cancel_contract() -> dict:
    proc = subprocess.run(
        ["node", str(CANCEL_HARNESS)],
        cwd=CANCEL_HARNESS.parent.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"cancel harness failed to run:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    stdout = proc.stdout.strip()
    end = stdout.rindex("}") + 1
    return json.loads(stdout[:end])


def test_cancel_contract_matches_backend_semantics(cancel_contract: dict):
    """queued->200, running->202, terminal->409, unknown->404, transport->503."""
    violations = cancel_contract.get("violations") or []
    assert not violations, "cancel contract violations:\n" + "\n".join(violations)


def test_a_202_cancel_is_never_reported_as_cancelled(cancel_contract: dict):
    """A 202 means 'asked to stop'; the job is not terminal until it says so."""
    observations = cancel_contract.get("observations") or {}
    running = observations.get("running job is still winding down (202 is NOT cancelled)")
    assert running is not None, "the 202 case was not exercised"
    assert running["status"] == 202
    assert running["body"].get("cancel_requested") is True


def _run_harness(name: str) -> dict:
    """Execute a node contract harness and parse its JSON document."""
    harness = REPO_ROOT / "frontend" / "tests" / "contract" / name
    proc = subprocess.run(
        ["node", str(harness)],
        cwd=harness.parent.parent,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"{name} failed to run:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    stdout = proc.stdout.strip()
    return json.loads(stdout[: stdout.rindex("}") + 1])


# ---------------------------------------------------------------------------
# Canonical reconstruction rule -- tested against the REAL module the UI
# imports, so the UI and its tests cannot drift apart.
# ---------------------------------------------------------------------------


def test_canonical_reconstruction_rule_matches_the_spec_table():
    """The full status x version_id matrix, from the shared helper itself."""
    doc = _run_harness("reconstruction-rule.contract.mjs")
    assert not doc["violations"], "canonical rule violations:\n" + "\n".join(doc["violations"])

    results = doc["results"]
    expected = {
        "succeeded + version": "success",
        "succeeded + missing version": "failure",
        "partial + version": "partial",
        "partial + missing version": "failure",
        "failed + version": "failure",
        "cancelled + version": "failure",
        "queued + version": "pending",
        "running + version": "pending",
        "succeeded + missing payload": "failure",
    }
    for name, kind in expected.items():
        assert name in results, f"rule case '{name}' was not exercised"
        assert results[name]["actualKind"] == kind, (
            f"{name}: expected '{kind}', got '{results[name]['actualKind']}'"
        )


def test_production_ui_uses_the_canonical_rule():
    """The modal must import the shared helper, not re-derive the rule."""
    modal = (
        FRONTEND_SRC / "components" / "workspace" / "RoomConstructionModal.tsx"
    ).read_text(encoding="utf-8")
    assert "classifyReconstruction" in modal, (
        "the reconstruction modal must use the canonical helper"
    )
    assert "isPersistedSuccess" in modal, (
        "the modal must gate success on the canonical helper, not a local check"
    )


# ---------------------------------------------------------------------------
# Runtime cancellation lifecycle (mandatory): a real simulation of the
# component's poll loop against a scripted backend.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cancel_lifecycle() -> dict:
    return _run_harness("cancel-lifecycle.runtime.mjs")


def test_runtime_cancellation_never_manufactures_a_cancelled_state(
    cancel_lifecycle: dict,
):
    """Reconstruct -> Running -> Cancel -> requested -> backend terminal."""
    violations = cancel_lifecycle.get("violations") or []
    assert not violations, "cancellation lifecycle violations:\n" + "\n".join(violations)


def test_runtime_cancellation_waits_for_the_backend_terminal_state(
    cancel_lifecycle: dict,
):
    """The timeline must show 'requested' while running, then the terminal state."""
    joined = "\n".join(cancel_lifecycle.get("timeline") or [])
    assert "after click: cancelState=requested status=running" in joined, (
        "after a 202 cancel the UI must show a requested cancellation and still be "
        f"running; timeline was:\n{joined}"
    )
    assert "after backend terminal: status=cancelled" in joined, (
        f"the terminal cancelled state must come from a poll; timeline was:\n{joined}"
    )


def test_runtime_duplicate_cancels_are_collapsed(cancel_lifecycle: dict):
    """Four clicks must produce exactly one cancel request."""
    joined = "\n".join(cancel_lifecycle.get("timeline") or [])
    assert "duplicate-cancel: 1 request(s) for 4 clicks" in joined, (
        f"duplicate cancel requests were not suppressed; timeline was:\n{joined}"
    )


def test_runtime_cancel_races_resolve_to_the_real_outcome(cancel_lifecycle: dict):
    """A cancel that loses the race must show the job's real terminal state."""
    joined = "\n".join(cancel_lifecycle.get("timeline") or [])
    assert "race-success: final status=succeeded" in joined, (
        f"a cancel that lost to a successful job must display 'succeeded'; got:\n{joined}"
    )
    assert "race-failure: final status=failed" in joined, (
        f"a cancel that lost to a failed job must display 'failed'; got:\n{joined}"
    )


# ---------------------------------------------------------------------------
# DELETE + upload contracts.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evidence_mutations() -> dict:
    return _run_harness("evidence-mutations.contract.mjs")


def test_delete_and_upload_contracts_hold(evidence_mutations: dict):
    """DELETE proves deletion with 204; uploads prove ingestion with evidence_id."""
    violations = evidence_mutations.get("violations") or []
    assert not violations, "evidence mutation violations:\n" + "\n".join(violations)


def test_delete_is_only_confirmed_on_204(evidence_mutations: dict):
    """A non-204 2xx must never be reported as a completed deletion."""
    obs = evidence_mutations.get("observations") or {}
    assert obs["delete/unexpected 200 is not a deletion"]["status"] == 502
    # 404 stays 404 (already gone), distinct from 503 (cannot tell).
    assert obs["delete/404 means the item is already gone"]["status"] == 404
    assert obs["delete/unreachable is 503, never 404"]["status"] == 503


def test_upload_is_only_confirmed_with_an_evidence_id(evidence_mutations: dict):
    """An arbitrary 2xx must never be read as a successful ingestion."""
    obs = evidence_mutations.get("observations") or {}
    for name in (
        "upload/201 without evidence_id is NOT ingestion",
        "upload/200 without evidence_id is not ingestion",
        "upload/empty 201 body is not ingestion",
        "upload/malformed 201 body is not ingestion",
    ):
        assert obs[name]["status"] == 502, (
            f"{name}: a 2xx without an evidence_id was treated as ingestion "
            f"(status {obs[name]['status']})"
        )
    assert obs["upload/201 with evidence_id is a real ingestion"]["status"] == 201


# ---------------------------------------------------------------------------
# Runtime async stale-data protection (scenarios A / B / C).
# ---------------------------------------------------------------------------


def test_stale_data_scenarios_pass_at_runtime():
    """A: a failed switch shows an error, not the old world.

    B: a successful switch shows only the new world.
    C: a LATE response from a superseded request is ignored.
    """
    doc = _run_harness("stale-data.runtime.mjs")
    violations = doc.get("violations") or []
    assert not violations, "stale-data violations:\n" + "\n".join(violations)

    notes = doc.get("notes") or []
    assert any(n.startswith("A: data=null") and "error=B failed" in n for n in notes), (
        f"scenario A: a failed switch must clear data and raise an error; got {notes}"
    )
    assert "B: data=world-B error=null" in notes, (
        f"scenario B: only the new world may be shown; got {notes}"
    )
    assert "C: data=world-C after a late B response" in notes, (
        f"scenario C: a late superseded response must be ignored; got {notes}"
    )
    assert any("after a late B failure" in n and "error=null" in n for n in notes), (
        f"scenario C: a late superseded failure must not surface; got {notes}"
    )


def test_hooks_pass_an_abort_signal_downstream():
    """The obsolete request must actually be cancelled, not just ignored."""
    client = (FRONTEND_SRC / "lib" / "api" / "client.ts").read_text(encoding="utf-8")
    assert "options.signal?.aborted" in client, (
        "a caller-driven abort must be re-thrown, not reported as a timeout"
    )
    worldir = (FRONTEND_SRC / "lib" / "api" / "worldir.ts").read_text(encoding="utf-8")
    assert "signal?: AbortSignal" in worldir, (
        "world resources must accept an AbortSignal so a superseded world "
        "switch can cancel its in-flight requests"
    )

