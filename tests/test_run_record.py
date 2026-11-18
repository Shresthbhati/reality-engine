"""RE-MLOPS-000001: a job's result can be traced to code, seeds, stage/artifact hashes and model bytes."""
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.run_record import build_run_record, models_available
from engine.pipeline.stages.runner import run_stages


class _S:
    def __init__(self, name, out):
        self.spec, self._out = StageSpec(name, "1", "cpu", True), out

    def run(self, ctx):
        now = datetime.now(timezone.utc)
        return StageResult(self.spec.name, "succeeded", "", self._out, {}, [], None, now, now, None)


def test_record_carries_commit_seeds_stage_hashes_and_lineage(tmp_path):
    m = run_stages([_S("a", {"pts": "h1"}), _S("b", {})], StageContext("j", tmp_path, {}, {"x": 1}),
                   pipeline_profile="p", pipeline_version="v9")
    rec = build_run_record(m, seeds={"pipeline": 42}, commit="abc123", models=[])
    assert rec["code_commit"] == "abc123" and rec["seeds"] == {"pipeline": 42}
    assert rec["pipeline_profile"] == "p" and rec["pipeline_version"] == "v9"
    assert [s["name"] for s in rec["stages"]] == ["a", "b"] and rec["stages"][0]["outputs"] == {"pts": "h1"}
    assert rec["stages"][0]["inputs_hash"] and rec["lineage"]["producer"] == {"pts": "a"}
    json.dumps(rec)                                                      # it goes straight onto the job payload


def test_models_available_hashes_present_active_checkpoints_only(tmp_path):
    ck = tmp_path / "w.pt"
    ck.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    specs = {
        "here": NS(checkpoint=ck, status="active", sha256=good),
        "wrong_hash": NS(checkpoint=ck, status="active", sha256="0" * 64),
        "missing": NS(checkpoint=tmp_path / "nope.pt", status="active", sha256=None),
        "old": NS(checkpoint=ck, status="deprecated", sha256=None),
        "heuristic": NS(checkpoint=None, status="active", sha256=None),
    }
    got = {r["name"]: r for r in models_available(specs)}
    assert set(got) == {"here", "wrong_hash"}
    assert got["here"]["sha256"] == good and got["here"]["declared_sha256_matches"] is True
    assert got["wrong_hash"]["declared_sha256_matches"] is False
