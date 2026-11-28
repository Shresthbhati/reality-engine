"""Frontend honesty guards: no production surface may present state the
backend did not produce (P0).

The Next.js route handlers under `frontend/src/app/api` are the browser's
door to `apps/api`. They previously also read bundled datasets off disk
(`datasets/room_capture/pipeline_out/*`) for any world id in a hard-coded
allow-list, so a world with no reconstruction could show someone else's
WorldIR, points, cameras, report or provenance -- and the world list gained a
fabricated `world-compiled-seed42` entry with invented coordinates. Several
screens also printed invented versions (`v1.0.0`) and a hard-coded geographic
anchor.

These tests are source-level guards over the production tree: comments are
stripped first, so the explanatory notes that document the removals stay
welcome, while live code is held to the contract. Legitimate fixtures live in
`frontend/src/test/fixtures` and are explicitly exempt.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"
PRODUCTION_API_DIR = FRONTEND_SRC / "app" / "api"

# Identifiers only ever produced by the removed fabricated paths.
FABRICATED_IDENTIFIERS = (
    "world-compiled-seed42",
    "dataset-room-capture",
    "room-capture",  # allow-list only; the real dataset folder is room_capture
    "v1-canonical-baseline",
    "session-room-capture",
)

# Invented values the UI used to print as if they were backend state.
FABRICATED_VALUES = (
    "37.7749",  # hard-coded San Francisco anchor in the worlds list + map panel
    "122.4194",
    "v1.0.0",  # invented version label for worlds with empty lineage
)

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"(?<!:)//[^\n]*")


def _strip_comments(source: str) -> str:
    """Drop TS/JS comments so guards assert on live code only."""
    without_block = _BLOCK_COMMENT.sub("", source)
    return _LINE_COMMENT.sub("", without_block)


def _production_files() -> list[Path]:
    """Every production TS/TSX file (test fixtures are exempt by design)."""
    files: list[Path] = []
    for path in FRONTEND_SRC.rglob("*"):
        if path.suffix not in {".ts", ".tsx"}:
            continue
        rel = path.relative_to(FRONTEND_SRC)
        if rel.parts and rel.parts[0] == "test":
            continue
        files.append(path)
    return files


def _code(path: Path) -> str:
    return _strip_comments(path.read_text(encoding="utf-8"))


def test_production_sources_reference_no_fabricated_identifiers():
    offenders: list[str] = []
    for path in _production_files():
        code = _code(path)
        for identifier in FABRICATED_IDENTIFIERS:
            if identifier in code:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {identifier}")
    assert not offenders, (
        "production frontend code still references fabricated world/version ids:\n"
        + "\n".join(offenders)
    )


def test_production_sources_print_no_fabricated_values():
    offenders: list[str] = []
    for path in _production_files():
        code = _code(path)
        for value in FABRICATED_VALUES:
            if value in code:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {value}")
    assert not offenders, (
        "production frontend code still hard-codes invented values:\n" + "\n".join(offenders)
    )


def test_api_routes_never_read_bundled_datasets_or_the_filesystem():
    offenders: list[str] = []
    for path in sorted(PRODUCTION_API_DIR.rglob("*.ts")):
        code = _code(path)
        rel = path.relative_to(REPO_ROOT)
        if "datasets" in code:
            offenders.append(f"{rel} reads the datasets/ tree")
        if re.search(r'from\s+"(node:)?fs"', code) or 'require("fs")' in code:
            offenders.append(f"{rel} touches the filesystem")
    assert not offenders, (
        "API routes must proxy the Reality Engine API instead of reading bundled "
        "dataset files:\n" + "\n".join(offenders)
    )


def test_evidence_image_route_uses_the_real_artifact_contract():
    route = (
        PRODUCTION_API_DIR / "worlds" / "[id]" / "evidence" / "[evidenceId]" / "image" / "route.ts"
    )
    code = _code(route)
    # apps/api/routes_misc.py serves GET /api/evidence/{id}/artifact.
    assert "/artifact" in code, "evidence image route must call /api/evidence/{id}/artifact"
    assert "/file" not in code, "evidence image route still calls a non-existent /file endpoint"


def test_reconstruction_ui_accepts_only_real_backend_job_statuses():
    """The outcome decision is delegated to the canonical helper.

    The modal no longer branches on status strings itself, so the guard is
    that it imports `classifyReconstruction` and never compares a job status
    against a non-backend word like "completed"/"done"/"finished".
    """
    modal = FRONTEND_SRC / "components" / "workspace" / "RoomConstructionModal.tsx"
    code = _code(modal)
    assert "classifyReconstruction" in code, (
        "the modal must decide outcomes via the canonical helper"
    )
    # apps/api/jobs.py grades succeeded/partial; there is no "completed".
    for bogus in ('"completed"', "'completed'", '"done"', '"finished"'):
        assert bogus not in code, (
            f"the modal must not treat a {bogus} status as success"
        )


def test_reconstruction_ui_never_infers_success_from_a_polling_budget():
    """When the poll budget runs out the job is still running server-side."""
    modal = FRONTEND_SRC / "components" / "workspace" / "RoomConstructionModal.tsx"
    code = _code(modal)
    assert "may still finish server-side" in code
    # A run starts from a cleared outcome, and an outcome is only recorded on a
    # proven persisted result via the canonical helper.
    assert "setOutcome(null);" in code, "each run must clear the previous outcome first"
    assert "setOutcome(" in code, "a proven outcome must still be recorded"
    assert "isPersistedSuccess(verdict)" in code, (
        "the outcome must be gated on the canonical persisted-success check"
    )


def test_frontend_data_module_exposes_no_rows():
    """`lib/data.ts` is a compatibility shim: every collection is empty."""
    data = FRONTEND_SRC / "lib" / "data.ts"
    code = _code(data)
    declarations = re.findall(r"export const (\w+)[^=]*=\s*(\[[^\]]*\])", code)
    assert declarations, "expected lib/data.ts to declare empty row arrays"
    for name, literal in declarations:
        assert literal.strip() == "[]", f"{name} still carries rows: {literal!r}"


def test_sessions_consumer_reads_the_items_envelope():
    """The client must consume the `{items}` contract the API actually serves.

    The list functions in `lib/api/*` all unwrap `ApiList<T>` via `data.items`.
    A consumer that read `data.sessions` (or fell back to `data.sessions ?? []`)
    would be reading a key the proxy's old fabricated envelope invented and the
    backend never emits -- it would silently render zero sessions even when the
    API is healthy.
    """
    offenders: list[str] = []
    for path in sorted((FRONTEND_SRC / "lib" / "api").glob("*.ts")):
        code = _code(path)
        for number, line in enumerate(code.splitlines(), start=1):
            if not re.search(r"\b(data|res|body)\.(sessions|worlds|evidence|jobs)\b", line):
                continue
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{number} -> {line.strip()}")
    assert not offenders, (
        "list consumers must unwrap the `items` envelope, not a key the API "
        "never emits:\n" + "\n".join(offenders)
    )


def test_test_fixtures_stay_out_of_production_code():
    """Fixtures are legitimate fakes, but only inside the test tree."""
    offenders: list[str] = []
    for path in _production_files():
        code = _code(path)
        if "test/fixtures" in code:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "production code imports the fake fixture module:\n" + "\n".join(offenders)
    )


# Measured values must never be invented. Confidence is produced by the
# compiler; when it is absent the UI must say so rather than assume a number.
# The check is line-scoped to confidence so unrelated 0.5s (alpha, HSL
# lightness) are not false positives.
_FALLBACK_ON_CONFIDENCE = ("?? 0.5", "|| 0.5", ": 0.5", "0.5)")


def test_no_fabricated_confidence_defaults():
    offenders: list[str] = []
    for path in _production_files():
        for number, line in enumerate(_code(path).splitlines(), start=1):
            if "confidence" not in line.lower() and "conf" not in line.lower():
                continue
            if any(pattern in line for pattern in _FALLBACK_ON_CONFIDENCE):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number} -> {line.strip()}")
    assert not offenders, (
        "confidence is a compiler measurement, not a UI default:\n" + "\n".join(offenders)
    )


def test_inspector_never_persists_an_invented_confidence():
    """A correction must not write a confidence the user never entered."""
    inspector = FRONTEND_SRC / "components" / "inspector" / "AdaptiveInspector.tsx"
    code = _code(inspector)
    assert "const conf: number | null" in code, (
        "Inspector confidence must be modelled as number | null"
    )
    assert "confidenceChange" in code, (
        "commit payloads must only carry confidence when the operator set it"
    )
    # A bare `confidence: editedConfidence` in a payload would ship whatever
    # the form happened to hold -- including a seeded default.
    assert "confidence: editedConfidence" not in code, (
        "commit payload still sends the raw confidence form state"
    )


def test_viewport_does_not_paint_unmeasured_entities_as_mid_confidence():
    """Uncertainty colouring must not invent a confidence for the view."""
    scene = FRONTEND_SRC / "lib" / "viewport" / "three-scene.ts"
    code = _code(scene)
    assert "typeof conf === \"number\"" in code, (
        "uncertainty colouring must require a recorded confidence"
    )
    assert "mesh.userData.confidence ?? 0.5" not in code


def test_production_sources_have_no_arbitrary_defaults():
    """Verify that arbitrary defaults (e.g. confidence = 0.85/0.5, coverage = 0.05,
    updatedAt = 'Current') are never substituted for missing backend values."""
    nav_code = _code(FRONTEND_SRC / "components" / "navigation" / "WorldNavPanel.tsx")
    inspector_code = _code(FRONTEND_SRC / "components" / "inspector" / "AdaptiveInspector.tsx")
    three_code = _code(FRONTEND_SRC / "lib" / "viewport" / "three-scene.ts")

    # Assert no arbitrary fallback to 0.5 confidence
    assert "room.confidence : 0.5" not in nav_code
    assert "entity.confidence : 0.5" not in nav_code
    assert "e.confidence : 0.5" not in three_code
    assert 'thickness_m as number | undefined)?.toFixed(3) ?? "0.100"' not in inspector_code

    # Assert truthful 'Not available' labels for unmeasured/unrecorded properties
    assert "Not available" in nav_code
    assert "Not available" in inspector_code


def test_building_topology_inspector_is_canonical():
    """Verify that BuildingTopologyInspector exists, consumes canonical /space-graph,
    and exposes room adjacencies, corridor routes, vertical stairs, and portal openings."""
    nav_code = _code(FRONTEND_SRC / "components" / "navigation" / "WorldNavPanel.tsx")
    assert "BuildingTopologyInspector" in nav_code
    assert "/space-graph" in nav_code
    assert "Room Adjacencies" in nav_code
    assert "Corridor Circulation Routes" in nav_code
    assert "Vertical Level Transitions" in nav_code
    assert "Portals & Openings" in nav_code


def test_viewport_synchronization_on_selection():
    """Verify that selecting any entity in the hierarchy explorer calls both onSelectEntity
    and onFrameEntity to focus the 3D viewport on that object."""
    nav_code = _code(FRONTEND_SRC / "components" / "navigation" / "WorldNavPanel.tsx")
    # Both callbacks must be invoked in hierarchy explorer click handlers
    assert "onSelectEntity(buildingEntity.id);" in nav_code and "onFrameEntity(buildingEntity.id);" in nav_code
    assert "onSelectEntity(lvl.entity.id);" in nav_code and "onFrameEntity(lvl.entity.id);" in nav_code
    assert "onSelectEntity(room.id);" in nav_code and "onFrameEntity(room.id);" in nav_code
    assert "onSelectEntity(corridor.id);" in nav_code and "onFrameEntity(corridor.id);" in nav_code
    assert "onSelectEntity(stair.id);" in nav_code and "onFrameEntity(stair.id);" in nav_code


# Storey/level identity must come from the backend. The Studio used to invent
# "level-ground"/"level-upper" and slice entities at a hard-coded y = 2.2 m,
# displaying a building the reconstruction never produced.
FABRICATED_LEVEL_IDS = ("level-ground", "level-upper")


def test_no_hard_coded_level_identity():
    offenders: list[str] = []
    for path in _production_files():
        code = _code(path)
        for identifier in FABRICATED_LEVEL_IDS:
            if f'"{identifier}"' in code or f"'{identifier}'" in code:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {identifier}")
    assert not offenders, (
        "level ids must come from WorldIR, not from string constants:\n"
        + "\n".join(offenders)
    )


def test_studio_levels_derive_from_canonical_data():
    panel = FRONTEND_SRC / "components" / "navigation" / "WorldNavPanel.tsx"
    code = _code(panel)
    # Levels come from the compiler's space graph or explicit level entities.
    assert "interior_space_graph" in code
    # No Y-threshold storey invention.
    assert "y <= 2.2" not in code and "y > 2.2" not in code


# ---------------------------------------------------------------------------
# Failure must never be laundered into valid-looking empty data.
#
# `catch { return null }` and `catch { return [] }` are the two shapes that made
# a total backend outage indistinguishable from "this world has no data":
# the viewport rendered empty, the activity log read "nothing has happened".
# A 404 is the ONLY case that legitimately renders as empty, and it is
# identified by its code, not by the absence of an exception.
# ---------------------------------------------------------------------------


def test_api_resource_modules_do_not_swallow_errors_into_empty_values() -> None:
    """Resource fetchers must propagate failure rather than return null / []."""
    offenders: list[str] = []
    for name in ("worldir.ts", "activity.ts", "sessions.ts", "worlds.ts", "evidence.ts"):
        path = FRONTEND_SRC / "lib" / "api" / name
        if not path.exists():
            continue
        code = _code(path)
        # A `catch` in these modules may only re-throw or narrow; it may not
        # produce an empty collection as its result.
        for match in re.finditer(r"catch\s*(?:\([^)]*\))?\s*\{", code):
            tail = code[match.end() : match.end() + 220]
            if re.search(r"return\s+(null|\[\])\s*;", tail):
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> catch returning empty")
    assert not offenders, (
        "an unreachable backend must not resolve to null/[] (that renders as a "
        "valid empty state):\n" + "\n".join(offenders)
    )


def test_use_api_distinguishes_absent_from_failed() -> None:
    """`useApi` must expose whether the backend confirmed absence (404)."""
    hooks = _code(FRONTEND_SRC / "lib" / "api" / "hooks.ts")
    assert "absent" in hooks, "useApi must distinguish a 404 'absent' from a failure"
    assert 'err.code === "not_found"' in hooks, (
        "only a backend 404 may be treated as an honest empty state"
    )
    # A failed request must clear stale data so the UI cannot show yesterday's
    # world while displaying an error.
    assert "setData(null)" in hooks


def test_diff_proxy_does_not_launder_failures_into_404() -> None:
    """A transport failure must not be reported as "no diff registered"."""
    route = _code(PRODUCTION_API_DIR / "worlds" / "[id]" / "diff" / "route.ts")
    assert "has not registered diff artifacts" not in route, (
        "the diff proxy must not invent a 'no diff registered' explanation for "
        "failures it never asked the backend about"
    )
    assert "status: 503" in route, "an unreachable backend must surface as 503"
    assert "res.status" in route, "a real backend status must be forwarded, not replaced"


def test_evidence_count_is_never_defaulted_to_zero() -> None:
    """"0 Evidence" is a claim the API did not make when it sent no count."""
    adapters = _code(FRONTEND_SRC / "lib" / "api" / "adapters.ts")
    assert "evidenceCount = 0" not in adapters, (
        "toWorldRow must not default the evidence count to 0; use null (unknown)"
    )
    card = _code(FRONTEND_SRC / "components" / "ui" / "WorldCard.tsx")
    assert "Evidence unknown" in card, (
        "an unreported evidence count must render as explicitly unknown"
    )


def test_studio_never_synthesizes_level_names_or_elevations() -> None:
    """Storey identity is backend truth, never derived from array position."""
    panel = _code(FRONTEND_SRC / "components" / "navigation" / "WorldNavPanel.tsx")
    assert "Level ${" not in panel, (
        "level names must come from WorldIR, not be synthesized from the array index"
    )
    assert "elevationY: Number(lvl.elevation_m ?? 0)" not in panel, (
        "an unrecorded elevation must not be coerced to 0.0 m"
    )
    assert "lvl.transform?.position?.y ?? 0" not in panel, (
        "an unrecorded entity elevation must not be coerced to 0.0 m"
    )


def test_activity_surfaces_failure_distinct_from_an_empty_log() -> None:
    """Both activity surfaces must render a failure, not 'no events yet'."""
    page = _code(FRONTEND_SRC / "app" / "(desktop)" / "activity" / "page.tsx")
    assert "setError(" in page, "the activity page must capture a load failure"
    assert "role=\"alert\"" in page, "the failure must be announced to the user"
    bar = _code(FRONTEND_SRC / "components" / "context" / "BottomContextBar.tsx")
    assert "activityError" in bar, (
        "the context bar must show an activity load failure separately from empty"
    )


def test_workspace_surfaces_backend_failure_over_an_empty_viewport() -> None:
    """An outage must produce a visible error, not a silently empty 3D scene."""
    station = _code(FRONTEND_SRC / "components" / "workspace" / "SpatialWorkstation.tsx")
    assert "workspaceFailure" in station, (
        "the workspace must compute and render a backend-failure banner"
    )
    assert "irAbsent" in station, (
        "a 404 'not reconstructed' must render its own honest empty state"
    )



# ---------------------------------------------------------------------------
# Counts derived from a fetch that may have FAILED are still fabrications.
# `list.length` is 0 both when the backend said "none" and when the request
# never succeeded; only the first may be printed as a count.
# ---------------------------------------------------------------------------


def test_session_detail_does_not_print_zero_items_after_a_failed_load() -> None:
    page = _code(FRONTEND_SRC / "app" / "(desktop)" / "sessions" / "[id]" / "page.tsx")
    assert "String(evidenceList.length)" not in page, (
        "a failed evidence load must not render as '0 Items'; show the failure"
    )
    assert "evidenceError" in page and "Unavailable" in page, (
        "the evidence count must degrade to an explicit 'Unavailable' on error"
    )


def test_inspector_does_not_claim_a_single_storey_building() -> None:
    """A building with no recorded storeys must not claim '1 Levels'."""
    inspector = _code(FRONTEND_SRC / "components" / "inspector" / "AdaptiveInspector.tsx")
    assert '?? "1"' not in inspector, (
        "storey count must not default to 1 when the backend recorded none"
    )
    assert "Levels unknown" in inspector, (
        "an unrecorded storey count must render as explicitly unknown"
    )


# ---------------------------------------------------------------------------
# Status vocabulary: a state the backend never emits must not render as
# success. apps/api/jobs.py grades exactly six states.
# ---------------------------------------------------------------------------

CANONICAL_JOB_STATES = ("queued", "running", "partial", "succeeded", "failed", "cancelled")


def test_no_frontend_component_treats_a_bogus_status_as_success() -> None:
    """'completed'/'done'/'success' are not in the backend's job vocabulary."""
    bogus = ("completed", "done", "finished", "success")
    offenders: list[str] = []
    for path in _production_files():
        for number, line in enumerate(_code(path).splitlines(), start=1):
            if "status" not in line:
                continue
            for word in bogus:
                # Only flag a *comparison* against the word, not a doc comment.
                if f'=== "{word}"' in line or f"== '{word}'" in line:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{number} -> {line.strip()}"
                    )
    assert not offenders, (
        f"job status must use the canonical {CANONICAL_JOB_STATES} vocabulary:\n"
        + "\n".join(offenders)
    )


def test_viewport_geometry_is_derived_from_backend_bounds() -> None:
    """The one BoxGeometry must be an entity bbox from real WorldIR bounds."""
    scene = _code(FRONTEND_SRC / "lib" / "viewport" / "three-scene.ts")
    # A single renderer, driven by bounds_min/bounds_max, is legitimate.
    assert scene.count("new THREE.BoxGeometry") == 1, (
        "only the entity bounding-box renderer may construct a BoxGeometry"
    )
    assert "bounds_min" in scene and "bounds_max" in scene, (
        "the box must be sized from the geometry's recorded bounds"
    )
    # Entities with no recorded confidence must not be painted as mid-confidence.
    assert "e.confidence === \"number\"" in scene, (
        "uncertainty colouring must require a recorded confidence"
    )


# ---------------------------------------------------------------------------
# POST route truth: a write must prove it happened.
#
# A 2xx from a write endpoint is not evidence the write happened. The backend
# names the thing it created (version_id / job_id / world id); without that the
# UI would show a correction, a reconstruction, or a world that do not exist.
# ---------------------------------------------------------------------------


def test_write_routes_require_the_identifier_that_proves_persistence() -> None:
    reconstruct = _code(
        PRODUCTION_API_DIR / "sessions" / "[sessionId]" / "reconstruct" / "route.ts"
    )
    assert "job_id" in reconstruct, (
        "the reconstruct proxy must require a job_id; without one there is "
        "nothing to poll and no job was created"
    )

    commit = _code(PRODUCTION_API_DIR / "worlds" / "[id]" / "commit" / "route.ts")
    assert "version_id" in commit, (
        "the commit proxy must require a version_id; a 2xx without one is not a "
        "persisted WorldStore version"
    )
    # The pre-existing 'data ?? {}' returned 200 with an invented body.
    assert "data ?? {}" not in commit, "the commit proxy still invents a success body"

    worlds = _code(PRODUCTION_API_DIR / "worlds" / "route.ts")
    assert "createdId" in worlds or ".id" in worlds, (
        "world creation must confirm the created world id before reporting success"
    )


def test_write_routes_do_not_turn_a_bad_request_into_an_outage() -> None:
    """A malformed request body is a 400, not 'Failed to connect to backend'."""
    for name in (
        "jobs/ingest/route.ts",
        "jobs/compile/route.ts",
        "sessions/mobile-tasks/route.ts",
        "sessions/ingest-mobile/route.ts",
    ):
        path = PRODUCTION_API_DIR / name
        code = _code(path)
        # request.json() must be parsed in its own try/catch, before the
        # transport try, so a client error is not reported as a 503 outage.
        body_parse = code.find("await request.json()")
        assert body_parse != -1, f"{name}: expected an explicit request body parse"
        first_transport_catch = code.find("Failed to connect to backend")
        assert first_transport_catch != -1, f"{name}: expected a transport catch"
        # The parse must be preceded by its own 400 handler.
        assert "must be valid JSON" in code, (
            f"{name}: a malformed body must be rejected with 400, not reported "
            "as a backend connection failure"
        )


def test_cancel_route_exists_and_preserves_202() -> None:
    """The backend serves POST /api/jobs/{id}/cancel; the proxy must match it."""
    cancel = PRODUCTION_API_DIR / "jobs" / "[jobId]" / "cancel" / "route.ts"
    assert cancel.exists(), (
        "the backend exposes POST /api/jobs/{job_id}/cancel; the frontend has no "
        "proxy for it, so a running reconstruction cannot be stopped from the UI"
    )
    code = _code(cancel)
    assert "/cancel" in code
    # The status must be forwarded, not collapsed: 202 (winding down) is not
    # the same as 200 (already cancelled).
    assert "response.status" in code, (
        "the cancel proxy must forward the backend's real status so 202/409 "
        "stay distinguishable"
    )


def test_cancel_client_does_not_treat_202_as_cancelled() -> None:
    platform = _code(FRONTEND_SRC / "lib" / "api" / "platform.ts")
    assert '"cancelling"' in platform, (
        "a 202 cancel must be modelled as 'cancelling', distinct from 'cancelled'"
    )
    assert "res.status === 202" in platform, (
        "the client must branch on 202 explicitly"
    )
    assert "pending: true" in platform, (
        "a pending cancellation must be marked so the caller keeps polling"
    )


# ---------------------------------------------------------------------------
# Stale data: a failed fetch must not leave the previous world's data visible
# as if it belonged to the current request.
# ---------------------------------------------------------------------------


def test_failed_fetches_do_not_render_as_the_previous_empty_result() -> None:
    """`catch { setX([]) }` around a NETWORK fetch reuses the empty rendering.

    Scoped to catches that sit near a real API call. A `catch` around
    `localStorage` is a different concern -- losing a bookmark is not a backend
    outage, and there is nothing to retry -- so those are not flagged.
    """
    offenders: list[str] = []
    for path in _production_files():
        if path.suffix != ".tsx":
            continue
        code = _code(path)
        for match in re.finditer(r"catch\s*(?:\([^)]*\))?\s*\{", code):
            tail = code[match.end() : match.end() + 260]
            hit = re.search(r"set[A-Z]\w*\(\s*(\[\])\s*\)", tail)
            if not hit:
                continue
            # Only flag when the catch belongs to a network call. `localStorage`
            # also has getItem/removeItem, so exclude any lookback that mentions
            # it: a lost bookmark is not a backend outage.
            start = max(0, match.start() - 900)
            preceding = code[start : match.start()]
            if "localStorage" in preceding or "sessionStorage" in preceding:
                continue
            is_network = bool(
                re.search(r"\b(fetch|list[A-Z]\w*|get[A-Z]\w*|searchAll)\s*\(", preceding)
            )
            if is_network:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)} -> catch sets {hit.group(1)}"
                )
    assert not offenders, (
        "a failed API fetch must not write an empty collection into state; an "
        "empty list is the correct rendering for 'none exist' and would hide "
        "the outage:\n" + "\n".join(offenders)
    )


def test_mobile_camera_does_not_call_a_failed_world_load_standalone() -> None:
    camera = _code(FRONTEND_SRC / "app" / "m" / "camera" / "page.tsx")
    assert '"error"' in camera, (
        "a failed World load must have its own state, not null, which renders as "
        "'None â€” standalone Session'"
    )
    assert "Could not load World" in camera or "World unavailable" in camera, (
        "the capture HUD must say the World could not be loaded"
    )


def test_space_graph_clears_the_previous_world_on_switch() -> None:
    """World A's topology must not remain visible while/after world B loads."""
    panel = _code(FRONTEND_SRC / "components" / "navigation" / "WorldNavPanel.tsx")
    assert "setGraphData(null)" in panel, (
        "the space-graph effect must clear the previous world's graph, or a "
        "failed fetch leaves world A's rooms/adjacencies on screen as world B"
    )
    assert "graphError" in panel, (
        "a failed space-graph read must be recorded and surfaced, not silently "
        "replaced by a partial local derivation"
    )


def test_use_api_clears_data_when_its_target_changes() -> None:
    """A world switch must not display the previous world's spatial data.

    This is a SOURCE-level backstop only. The behavioural proof is the runtime
    harness `frontend/tests/contract/stale-data.runtime.mjs`, which executes the
    generation/abort algorithm against a controllable fetcher and covers the
    late-response case a source check cannot.
    """
    hooks = _code(FRONTEND_SRC / "lib" / "api" / "hooks.ts")
    effect = hooks.find("useEffect(")
    assert effect != -1
    body = hooks[effect : effect + 1400]
    # The reset must happen at the START of the effect, before the fetch, so
    # stale data is not shown under the new world's label.
    assert "setData(null)" in body, (
        "useApi must clear the previous result when its dependencies change"
    )
    assert body.index("setData(null)") < body.index("fetcher("), (
        "data must be cleared BEFORE the new request is issued"
    )
    # A late response from a superseded request must be dropped by generation.
    assert "generationRef" in hooks, "useApi must use a request generation guard"
    assert "abortRef" in hooks, "useApi must abort the superseded request"





def test_successful_reconstruction_survives_worldir_refetch():
    """Defect 1 regression: success -> onReconstructionSuccess -> refetchIr()
    -> WorldIR 404 must NOT unmount the modal and wipe the committed result.

    The full-page WorldIR loading gate replaced this component's entire
    returned tree. Re-entering it after a successful reconstruction moved the
    RoomConstructionModal to a different position in the tree, so React
    unmounted/remounted it and destroyed jobStatus/outcome. The user saw no
    success at all. The gate must only apply to the FIRST load of a world.
    """
    src = (FRONTEND_SRC / "components" / "workspace" / "SpatialWorkstation.tsx").read_text(encoding="utf-8")

    # The gate must be conditional on the IR having settled at least once.
    assert "showInitialIrLoading" in src, (
        "SpatialWorkstation must gate the full-page WorldIR loading state on "
        "the first load only; a background refetch must not tear down the tree."
    )
    assert "irHasSettled" in src, (
        "SpatialWorkstation must track that the WorldIR request has settled so "
        "a later refetch cannot re-enter the unmounting loading branch."
    )

    # The gate must no longer be a bare `isIrLoading` early return.
    assert "if (isIrLoading) {" not in src, (
        "The loading gate must not be keyed on isIrLoading alone: refetchIr() "
        "sets it true again and would unmount the modal holding the success state."
    )

    # The modal must still be rendered in the main tree, and the success
    # callback must still refresh the world behind it.
    assert src.count("<RoomConstructionModal") == 1, (
        "RoomConstructionModal must be rendered exactly once; rendering it in "
        "two branches makes React remount it and drop terminal job state."
    )
    assert "refetchIr()" in src, "Success must still refresh the reconstructed world."


def test_cancel_button_is_wired_to_the_cancel_handler():
    """Defect 2 regression: the cancel button lost its onClick handler.

    Without onClick the button rendered but did nothing: no request reached the
    backend, so a 409/404 refusal produced no badge, no error, and no
    'Cancel refused' state. The UI looked live and was completely inert.
    """
    modal = (FRONTEND_SRC / "components" / "workspace" / "RoomConstructionModal.tsx").read_text(encoding="utf-8")

    # The button carrying the test hook MUST be the button wired to the handler.
    import re

    m = re.search(
        r'<button[^>]*data-testid="cancel-job-button"[^>]*>', modal, re.DOTALL
    )
    assert m, "The cancel button must expose data-testid=cancel-job-button."
    assert "onClick={handleCancel}" in m.group(0), (
        "The cancel button must be wired to handleCancel. A cancel button with "
        "no handler silently discards 409/404 refusals: the user sees an active "
        "affordance that does nothing and is never told the cancel was refused."
    )


def test_cancel_refusal_states_are_reachable_from_the_ui():
    """A 409/404 refusal must render an explicit refusal badge, never a
    'cancelled' claim. The refusal is decided by the backend's answer, not by
    the fact that a request was sent."""
    modal = (FRONTEND_SRC / "components" / "workspace" / "RoomConstructionModal.tsx").read_text(encoding="utf-8")

    assert 'data-testid="cancel-refused-badge"' in modal, (
        "A refused cancellation must render a visible refusal badge."
    )
    assert 'kind: "refused"' in modal, (
        "The cancel handler must set an explicit refused state on rejection."
    )
    # The refusal must never be reported as a terminal cancellation.
    assert 'Cancellation accepted' in modal, (
        "Only a 2xx answer may claim acceptance; 409/404 must not."
    )

