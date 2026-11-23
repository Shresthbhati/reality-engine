"""PROPOSED initial benchmark selection. None of this data is available, downloaded or verified: every entry is ``PROPOSED``.

Counts are scenes per dataset (3RScan counts reference+rescan PAIRS). Scene ids are deliberately NOT chosen here: choosing them requires the
licensed data, and the choice must be made by a fixed rule (seeded, by physical group -- ``splits.assign_splits``), not by looking at results.
"""

from __future__ import annotations

PROPOSED_SELECTION: dict[str, dict[str, object]] = {
    "scannet": {"count": 30, "unit": "scenes", "status": "PROPOSED"},
    "scannetpp": {"count": 15, "unit": "scenes", "status": "PROPOSED"},
    "arkitscenes": {"count": 20, "unit": "scenes", "status": "PROPOSED"},
    "multiscan": {"count": 10, "unit": "scenes", "status": "PROPOSED"},
    "structured3d": {"count": 30, "unit": "scenes", "status": "PROPOSED"},
    "3rscan": {"count": 10, "unit": "reference+rescan pairs", "status": "PROPOSED"},
}
AVAILABLE = False  # flipped only when the licensed data is on disk, hashed into manifests and checked by splits.leakage_problems


def selection_report() -> dict[str, object]:
    return {"status": "PROPOSED", "available": AVAILABLE, "total_scene_units": sum(int(v["count"]) for v in PROPOSED_SELECTION.values()),
            "datasets": PROPOSED_SELECTION}
