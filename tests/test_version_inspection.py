"""An older version explains itself from its OWN report (changes, build strategy, verdict, dense), not the head's."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
from apps.api.routes_reconstructions import _rows_to_versions

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _row(i, parent, report):
    return SimpleNamespace(id=f"v{i}", parent_version_id=parent, created_at=T0 + timedelta(minutes=i), report=report)


def test_each_version_carries_its_own_build_and_change_record():
    counts = {"preserved": 3, "regrouped": 1, "new": 2}
    rows = [
        _row(1, None, {"changes": ["First model built with 6 placed photos."], "verdict": {"verdict": "ACCEPT"}}),
        _row(2, "v1", {
            "changes": ["2 more photos placed in the model."],
            "physical": {"available": True, "counts": counts},
            "stages": {"reconstruction": {"colmap_session": {"mode": "incremental"}}},
            "verdict": "ACCEPT_WITH_UNCERTAINTY",
        }),
    ]
    v1, v2 = _rows_to_versions(rows, "v2")
    assert v1["changes"] == ["First model built with 6 placed photos."] and v1["change_counts"] is None
    assert v1["verdict"] == "ACCEPT" and v1["strategy"] is None
    assert v2["change_counts"] == counts and v2["strategy"] == {"mode": "incremental"}
    assert v2["verdict"] == "ACCEPT_WITH_UNCERTAINTY" and v2["is_current"] and not v1["is_current"]


def test_a_version_without_a_record_reports_none_not_the_heads_facts():
    (v,) = _rows_to_versions([_row(1, None, {})], "v1")
    assert v["changes"] is None and v["change_counts"] is None and v["strategy"] is None
