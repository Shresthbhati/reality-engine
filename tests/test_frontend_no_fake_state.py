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
    modal = FRONTEND_SRC / "components" / "workspace" / "RoomConstructionModal.tsx"
    code = _code(modal)
    # apps/api/jobs.py grades a run as succeeded or partial; there is no
    # "completed" status, and success is only meaningful alongside the
    # version the worker actually adopted.
    assert 'job.status === "completed"' not in code
    assert '"completed"' not in code, (
        "RoomConstructionModal must not treat a 'completed' status as success"
    )
    assert 'job.status === "partial"' in code
    assert "payload" in code and "version_id" in code, (
        "a succeeded/partial job must be gated on the committed version id it carries"
    )


def test_reconstruction_ui_never_infers_success_from_a_polling_budget():
    modal = FRONTEND_SRC / "components" / "workspace" / "RoomConstructionModal.tsx"
    code = _code(modal)
    # When the poll budget runs out the job is still running server-side:
    # the modal must say so, and must not reach setOutcome / the success
    # banner on that path.
    assert "may still finish server-side" in code
    # A run starts from a cleared outcome, and an outcome is only recorded
    # with the version the worker actually adopted.
    assert "setOutcome(null);" in code, "each run must clear the previous outcome first"
    assert (
        "setOutcome({ status: job.status, versionId, degraded });" in code
    ), "outcome must be recorded from a real terminal status plus its committed version"


def test_frontend_data_module_exposes_no_rows():
    """`lib/data.ts` is a compatibility shim: every collection is empty."""
    data = FRONTEND_SRC / "lib" / "data.ts"
    code = _code(data)
    declarations = re.findall(r"export const (\w+)[^=]*=\s*(\[[^\]]*\])", code)
    assert declarations, "expected lib/data.ts to declare empty row arrays"
    for name, literal in declarations:
        assert literal.strip() == "[]", f"{name} still carries rows: {literal!r}"


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

