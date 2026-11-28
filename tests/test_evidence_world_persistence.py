"""Evidence -> WorldIR -> WorldStore -> fresh-process reload -> correction -> new version -> diff -> export, through the stage entry
point. The engines are SYNTHETIC scripted doubles, so this proves the lifecycle plumbing (persistence, validation, integrity, versioning,
export), not real reconstruction or accuracy -- the real-model production run is a separate, opt-in test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from apps.api import evidence_world_stage as ew
from tests.test_multiengine_evidence_flow import _setup
from world_ir.corrections import reclassify_entity
from world_ir.diff import diff_worlds
from worldstore.store import WorldStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def run(tmp_path):
    files = []
    for n in ("img1.jpg", "img2.jpg"):
        p = tmp_path / "src" / n
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff\xd9")
        files.append(str(p))

    def inputs():
        reg, adapters = _setup(())
        return reg, adapters, None

    facts = ew.run_for_job(team_id="t1", world_id="w1", job_id="job-1", sources=files, base=tmp_path / "ws", flow_inputs=inputs)
    root = tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-1"
    return facts, root


def test_the_stage_persists_a_validated_world_version_and_says_so(run):
    facts, root = run
    w = facts["world_ir"]
    assert facts["status"] == "complete" and w["status"] == "persisted" and w["reload_verified"] is True and w["validation"] == "valid"
    assert w["entities"] == facts["counts"]["entities"] == 3 and (root / "worldstore" / "versions" / f"{w['version_id']}.json").is_file()


def test_the_world_reloads_in_a_fresh_process_with_the_same_content(run):
    facts, root = run
    vid = facts["world_ir"]["version_id"]
    code = (
        "import json; from worldstore.store import WorldStore; "
        f"s=WorldStore(r'{root / 'worldstore'}'); w=s.load_version('{vid}'); "
        "print(json.dumps({'problems': s.verify_version('" + vid + "'), 'entities': sorted(w.entities), "
        "'dump': json.dumps(w.to_dict(), sort_keys=True)}))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    here = WorldStore(root / "worldstore").load_version(vid)
    assert got["problems"] == [] and got["entities"] == sorted(here.entities)
    assert got["dump"] == json.dumps(here.to_dict(), sort_keys=True)  # a different process sees byte-identical content


def test_provenance_evidence_and_unmeasured_confidence_survive_persistence(run):
    facts, root = run
    world = WorldStore(root / "worldstore").load_version(facts["world_ir"]["version_id"])
    for ent in world.entities.values():
        assert ent.confidence is None  # nothing was measured, so nothing was invented
        ev = ent.custom_properties["evidence"]
        assert ev["members"] and ent.observations  # the supporting observation ids and the evidence references remain
        assert ev["ontology"]["path"]  # the semantic decision and its basis are kept
    snap = json.loads((root / "snapshot.json").read_text())
    assert sorted(f"ent-{e['entity_id']}" for e in snap["entities"]) == sorted(world.entities)  # the snapshot and the world agree


def test_correction_creates_a_new_valid_version_a_diff_and_an_export_that_reflects_it(run):
    facts, root = run
    store = WorldStore(root / "worldstore")
    v1 = facts["world_ir"]["version_id"]
    base = store.load_version(v1)
    chair_id = next(i for i, e in base.entities.items() if "chair" in (e.semantic_labels[0] if e.semantic_labels else ""))
    corrected = store.load_version(v1)
    reclassify_entity(corrected, chair_id, "furniture/seating/sofa", note="user says sofa", user_id="tester")
    v2 = store.save_version(corrected, parent=v1)
    assert store.verify_version(v2.version_id) == [] and v2.version_id != v1
    assert store.load_version(v1).to_dict() == base.to_dict()  # the first version is immutable
    diff = diff_worlds(store.load_version(v1), store.load_version(v2.version_id))
    assert [d.entity_id for d in diff.entity_diffs] == [chair_id]
    from sdk.reality import export

    persisted = store.load_version(v2.version_id)
    for fmt in ("usda", "gltf", "geojson"):
        _, rep = export(persisted, fmt)
        # 2D-only evidence has no position, so the exporters place nothing -- and say so for EVERY entity (no prim is fabricated)
        assert rep.entities_exported == () and set(rep.entities_skipped) == set(persisted.entities), fmt
        assert all("transform" in r for r in rep.skip_reasons), fmt
    assert persisted.entities[chair_id].name == "Sofa"  # while the persisted world itself carries the correction


def test_the_multi_engine_pass_runs_once_and_the_report_keeps_its_matrix_import(tmp_path, monkeypatch):
    """A second, matrix-less multi-engine pass used to replace facts["multi_engine"] and drop the lineage.

    The stage ran the whole engine registry twice: once with ``matrix_root`` (which imports the observations into this
    run's Evidence Matrix and returns ``matrix_import``/``mask_store_uri``) and once without it. The second result won,
    so the job report lost the mask store and the import lineage while every engine was paid for twice. One pass only.
    """
    import perception.multi_engine_stage as mes

    calls: list[dict] = []
    matrix_import = {
        "status": "imported",
        "n_observations_imported": 4,
        "n_run_records": 1,
        "observation_content_ids_by_engine": {"grounding_dino": ["c1", "c2"]},
        "engine_statuses": {"grounding_dino": "succeeded"},
        "mask_store_uri": "sha256://deadbeef",
    }

    def fake_multi_engine(**kwargs):
        calls.append(kwargs)
        base = {"status": "succeeded", "engines_run": {"grounding_dino": "succeeded"}, "n_observations": 4}
        if kwargs.get("matrix_root") is None:  # what a matrix-less pass returns in production: no import, no mask store
            return {**base, "mask_store_uri": None}
        return {**base, "mask_store_uri": "sha256://deadbeef", "matrix_import": matrix_import}

    monkeypatch.setattr(mes, "run_multi_engine_for_job", fake_multi_engine)
    files = []
    for n in ("img1.jpg", "img2.jpg"):
        p = tmp_path / n
        p.write_bytes(b"\xff\xd8\xff\xd9")
        files.append(str(p))

    def inputs():
        reg, adapters = _setup(())
        return reg, adapters, None

    facts = ew.run_for_job(team_id="t1", world_id="w1", job_id="job-me", sources=files, base=tmp_path / "ws", flow_inputs=inputs)
    root = tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-me"
    assert len(calls) == 1, "the engine registry must run once per job"
    assert calls[0]["matrix_root"] == root  # and that single pass is the one that imports into the run's matrix
    assert facts["multi_engine"]["matrix_import"] == matrix_import
    assert facts["multi_engine"]["mask_store_uri"] == "sha256://deadbeef"
    assert facts["multi_engine_evidence"]["status"] == "imported"
    assert facts["multi_engine_evidence"]["mask_store_uri"] == "sha256://deadbeef"
    assert facts["multi_engine_evidence"]["observation_content_ids_by_engine"] == {"grounding_dino": ["c1", "c2"]}
    recorded = json.loads((root / "multi_engine_evidence.json").read_text(encoding="utf-8"))
    assert recorded["mask_store_uri"] == "sha256://deadbeef"  # the lineage is on disk, not only in the report
    world = WorldStore(root / "worldstore").load_version(facts["world_ir"]["version_id"])
    assert world.metadata["multi_engine_evidence"]["mask_store_uri"] == "sha256://deadbeef"  # and the world inherits it


def test_a_world_that_cannot_be_validated_is_not_persisted_and_the_stage_is_not_complete(tmp_path, monkeypatch):
    import world_ir.validation as v

    class Bad:
        errors = ["[error] forced: invalid world"]

    monkeypatch.setattr(v, "validate_world_ir", lambda w: Bad())
    files = []
    for n in ("img1.jpg", "img2.jpg"):
        p = tmp_path / n
        p.write_bytes(b"\xff\xd8\xff\xd9")
        files.append(str(p))

    def inputs():
        reg, adapters = _setup(())
        return reg, adapters, None

    facts = ew.run_for_job(team_id="t1", world_id="w1", job_id="job-9", sources=files, base=tmp_path / "ws", flow_inputs=inputs)
    assert facts["world_ir"]["status"] == "failed" and "NOT persisted" in facts["world_ir"]["reason"]
    assert facts["status"] == "partial"  # never "complete" when the canonical world was not persisted
    assert not (tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-9" / "worldstore" / "versions").exists()


def test_no_entities_means_nothing_to_persist_not_an_empty_world(tmp_path):
    out = ew.persist_world(tmp_path, "job-0", None)
    assert out["status"] == "not_persisted" and out["entities"] == 0 and not (tmp_path / "worldstore").exists()


def test_persisting_the_same_world_again_reuses_the_version_instead_of_duplicating(tmp_path, monkeypatch):
    seen = {}
    real = ew.persist_world

    def spy(root, job_id, bridge):
        seen["bridge"] = bridge
        return real(root, job_id, bridge)

    monkeypatch.setattr(ew, "persist_world", spy)
    files = []
    for n in ("img1.jpg", "img2.jpg"):
        p = tmp_path / n
        p.write_bytes(b"\xff\xd8\xff\xd9")
        files.append(str(p))

    def inputs():
        reg, adapters = _setup(())
        return reg, adapters, None

    facts = ew.run_for_job(team_id="t1", world_id="w1", job_id="job-1", sources=files, base=tmp_path / "ws", flow_inputs=inputs)
    root = tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-1"
    again = real(root, "job-1", seen["bridge"])  # the same world, persisted a second time (e.g. a retry after a crash)
    assert again["status"] == "persisted" and again["reused_existing_version"] is True
    assert again["version_id"] == facts["world_ir"]["version_id"]
    assert len(list((root / "worldstore" / "versions").glob("*.json"))) == 1  # no duplicate version
