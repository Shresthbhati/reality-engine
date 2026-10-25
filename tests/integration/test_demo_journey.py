"""The canonical demo journey (scripts/demo_journey.py) as a test: real South Building photographs through the real
application and real COLMAP -- 6 photos -> V1, +4 -> V2, +10 -> V3, inspect what changed, inspect the previous version,
export the current world in all six core formats.

The journey asserts only STRUCTURAL facts (one world, three distinct chained versions, evidence accumulates, earlier
versions byte-identical, every export non-empty and parsed back); the numbers it measures are recorded, never compared
with hard-coded expectations. Marked slow: it runs the real engines (several minutes on a CPU/GPU workstation).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("demo_journey", ROOT / "scripts" / "demo_journey.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

missing = demo._prerequisites(demo.DEFAULT_DATASET)
pytestmark = [pytest.mark.slow, pytest.mark.integration,
              pytest.mark.skipif(bool(missing), reason="; ".join(missing) or "ok")]


def test_the_canonical_demo_journey_completes_on_real_photographs(tmp_path):
    m = demo.run_journey(demo.DEFAULT_DATASET, tmp_path / "demo")
    assert m["passed"] and not any(not c["ok"] for c in m["checks"])
    # one world, three versions, evidence accumulating -- measured, not expected
    assert len(m["steps"]) == 3 and len({s["version"] for s in m["steps"]}) == 3
    used = m["accumulated_evidence"]
    assert used == sorted(used) and used[0] < used[-1]
    # the metrics file is what a reader audits: it must exist and parse
    metrics = json.loads((tmp_path / "demo" / "demo_metrics.json").read_text(encoding="utf-8"))
    assert metrics["world_id"] == m["world_id"] and metrics["step6_exports"]
    for fmt, e in metrics["step6_exports"].items():
        assert e["ok"] and e["bytes"] > 0 and e["entities_exported"] > 0, (fmt, e)
    # what changed is reported with all ten categories once a previous version exists
    counts = [s["change_counts"] for s in m["steps"][1:] if s["change_counts"]]
    assert counts and all(len(c) == 10 for c in counts)
