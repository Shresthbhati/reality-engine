"""The audit doc must name exactly the stages the production job runs, in order (docs/audits/PRODUCTION_STAGE_MAP.md)."""
from __future__ import annotations

import re
from pathlib import Path

from engine.pipeline.stages.registry import PIPELINE_PROFILES, PROFILE_NAME, build_stages

DOC = Path(__file__).resolve().parents[1] / "docs" / "audits" / "PRODUCTION_STAGE_MAP.md"


def _doc_stage_line() -> list[str]:
    m = re.search(r"^PIPELINE_STAGES:\s*(.+)$", DOC.read_text(encoding="utf-8"), re.MULTILINE)
    assert m, "PRODUCTION_STAGE_MAP.md must carry a 'PIPELINE_STAGES: a, b, c' line"
    return [s.strip() for s in m.group(1).split(",")]


def test_doc_stage_list_matches_the_job_profile():
    assert _doc_stage_line() == PIPELINE_PROFILES[PROFILE_NAME]


def test_job_profile_builds_one_stage_per_name():
    assert [s.spec.name for s in build_stages(PROFILE_NAME)] == PIPELINE_PROFILES[PROFILE_NAME]
