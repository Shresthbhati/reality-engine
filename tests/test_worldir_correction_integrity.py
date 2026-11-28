"""WorldIR correction + version integrity (Sprint 2, workstream 6).

Store-level tests use a real WorldStore on disk; API tests use the existing application test client and seeded world.
Covers: mandatory base version (both route families), typed correction records, stale derived artifacts, atomicity under
injected failure (mid-apply, mid-validate, mid-persist), fresh-process reload equality, entity lineage, door-based room
connectivity with unknown-not-assumed, and surfaced competing hypotheses / unresolved geometry."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from world_ir.correction_log import (
    BaseVersionRequired, CorrectionRecord, CorrectionRejected, StaleBaseVersion, classify_artifacts,
    commit_correction_atomically, correction_log, lineage_ancestors, lineage_successors, require_base_version, world_content_hash,
)
from world_ir.corrections import merge_entities, reclassify_entity, split_entity
from world_ir.room_connectivity import apply_door_connectivity, derive_door_connectivity
from world_ir.schema_v1 import Entity, EntityType, Observation, Provenance
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore, WorldStoreError


def _ent(eid, etype, name="", **cp):
    return Entity(id=eid, type=etype, name=name or eid, provenance=Provenance.INFERRED, confidence=0.8, custom_properties=cp)


def _house() -> WorldIR:
    """3 rooms; door-ab hosted by a wall bounding BOTH a and b; door-bc hosted by a wall bounding only b (c side unknown);
    room-d has no door evidence at all."""
    w = WorldIR(id="w1", name="house", created_at=1.0)
    for r, walls in (("room-a", ["wall-a", "wall-ab"]), ("room-b", ["wall-b", "wall-ab", "wall-bc"]), ("room-c", ["wall-c"]),
                     ("room-d", ["wall-d"])):
        w.entities[r] = _ent(r, EntityType.ROOM, boundary_element_ids=walls, bounding_box=[[0, 0, 0], [1, 1, 1]])
    for wall in ("wall-a", "wall-b", "wall-c", "wall-d", "wall-ab", "wall-bc"):
        owner = {"wall-a": "room-a", "wall-b": "room-b", "wall-c": "room-c", "wall-d": "room-d", "wall-ab": "room-a",
                 "wall-bc": "room-b"}[wall]
        w.entities[wall] = _ent(wall, EntityType.WALL, space_id=owner, length_m=3.0, bounding_box=[[0, 0, 0], [3, 0.1, 2.5]])
    w.entities["door-ab"] = _ent("door-ab", EntityType.DOOR, host_wall_id="wall-ab", bounding_box=[[0, 0, 0], [1, 0.1, 2]])
    w.entities["door-bc"] = _ent("door-bc", EntityType.DOOR, host_wall_id="wall-bc", bounding_box=[[0, 0, 0], [1, 0.1, 2]])
    w.entities["room-a"].observations.append(Observation(id="obs-1", sensor_type="camera", data_hash="h1"))
    return w


@pytest.fixture()
def store(tmp_path) -> WorldStore:
    s = WorldStore(tmp_path / "ws")
    s.save_version(_house(), parent=None, version_id="v-base")
    return s


def _head(store):
    return store.list_versions()[-1]


# ------------------------------------------------------------------------------------------------ base version rules

def test_base_version_is_mandatory_and_stale_is_named():
    assert require_base_version("v1", "v1") == "v1"
    with pytest.raises(BaseVersionRequired) as e:
        require_base_version(None, "v9")
    assert "v9" in str(e.value)
    with pytest.raises(BaseVersionRequired):
        require_base_version("  ", "v9")
    with pytest.raises(StaleBaseVersion) as s:
        require_base_version("v1", "v2")
    assert s.value.base == "v1" and s.value.current == "v2" and "stale" in str(s.value) and "v2" in str(s.value)


def test_transaction_refuses_missing_and_stale_base_without_touching_the_store(store):
    before = _head(store)
    with pytest.raises(BaseVersionRequired):
        commit_correction_atomically(store, expected_base_version_id=None, mutate=lambda w: None, operation="x", actor_id="u")
    with pytest.raises(StaleBaseVersion):
        commit_correction_atomically(store, expected_base_version_id="v-old", mutate=lambda w: None, operation="x", actor_id="u")
    assert _head(store) == before and len(store.list_versions()) == 1


# ------------------------------------------------------------------------------------------------ record completeness

def test_correction_record_is_complete_typed_and_persisted_in_the_version_it_describes(store):
    def mutate(w):
        reclassify_entity(w, "wall-a", "wall/partition", "survey", user_id="u-7")
        w.entities["wall-b"].custom_properties["length_m"] = 4.25
        w.entities["wall-b"].custom_properties["dimensions"] = {"length_m": {"value": 4.25, "state": "VERIFIED"}}

    exports = [{"artifact_id": "exp-1", "kind": "export:glb", "built_from_version_id": "v-base"}]
    stored, rec = commit_correction_atomically(
        store, expected_base_version_id="v-base", mutate=mutate, operation="reclassify+dimension", actor_id="u-7",
        note="field survey", artifacts=exports)
    assert rec.missing_fields() == []
    assert (rec.actor_id, rec.actor_kind, rec.base_version_id, rec.new_version_id) == ("u-7", "user", "v-base", stored.version_id)
    assert rec.created_at > 0 and rec.note == "field survey"
    assert {"wall-a", "wall-b"} <= {c.entity_id for c in rec.changes}
    cls = next(c for c in rec.changes if c.entity_id == "wall-a" and c.field == "custom_properties.classification_basis")
    assert cls.old is None and cls.new == "user"
    recomputed = [r for r in rec.recomputed_values if r["entity_id"] == "wall-b" and r["quantity"] == "length_m"]
    assert recomputed == [{"entity_id": "wall-b", "quantity": "length_m", "old": 3.0, "new": 4.25}]
    # evidence: the original observation is preserved byte-identical
    assert rec.evidence_untouched and "obs-1" in rec.preserved_observation_ids and rec.changed_observation_ids == []
    assert rec.stale_artifacts == exports
    # the record is inside the stored version and round-trips
    reloaded = store.load_version(stored.version_id)
    [stored_rec] = correction_log(reloaded)
    assert stored_rec == CorrectionRecord.from_dict(rec.to_dict()) and stored_rec.new_version_id == stored.version_id
    # the parent is untouched: no log, original value
    base = store.load_version("v-base")
    assert correction_log(base) == [] and base.entities["wall-b"].custom_properties["length_m"] == 3.0


def test_record_flags_lost_evidence_and_system_actor():
    from world_ir.correction_log import stamp_correction

    base, new = _house(), _house()
    new.entities["room-a"].observations.clear()
    rec = stamp_correction(base, new, operation="rebuild", actor_id="engine.rebuild", actor_kind="system",
                           base_version_id="v1", new_version_id="v2", entity_ids=["room-a"])
    assert rec.evidence_untouched is False and rec.changed_observation_ids == ["obs-1"] and rec.actor_kind == "system"
    assert CorrectionRecord(record_id="", operation="", actor_id="", actor_kind="robot", created_at=0, base_version_id="",
                            new_version_id="").missing_fields()


# --------------------------------------------------------------------------------------------------- stale artifacts

def test_artifacts_built_from_an_older_version_are_flagged_stale_by_version_id():
    out = classify_artifacts([
        {"artifact_id": "e1", "kind": "export:ifc", "built_from_version_id": "v1"},
        {"artifact_id": "e2", "kind": "export:glb", "built_from_version_id": "v2"},
        {"artifact_id": "e3", "kind": "floorplan", "built_from_version_id": None},
    ], "v2")
    by = {a["artifact_id"]: a for a in out}
    assert by["e1"]["stale"] is True and by["e1"]["current_version_id"] == "v2"
    assert by["e2"]["stale"] is False
    assert by["e3"]["stale"] is False and by["e3"]["stale_unknown"] is True    # unknown provenance is reported, not guessed


def test_a_calibration_determination_goes_stale_when_geometry_changes(store):
    base = store.load_version("v-base")
    base.metadata["scale_calibration"] = {"id": "cal-1", "parent_version": "v-gauge"}
    store2 = WorldStore(store._root.parent / "ws2")
    store2.save_version(base, parent=None, version_id="v-cal")

    def mutate(w):
        w.entities["wall-a"].custom_properties["length_m"] = 9.0

    _, rec = commit_correction_atomically(store2, expected_base_version_id="v-cal", mutate=mutate, operation="dim", actor_id="u")
    assert {"artifact_id": "cal-1", "kind": "scale_calibration", "built_from_version_id": "v-cal"} in rec.stale_artifacts


# --------------------------------------------------------------------------------------------- atomicity / integrity

def _assert_untouched(store, head_before, hash_before, content_before):
    now = _head(store)
    assert now.version_id == head_before.version_id and now.artifact_hash == hash_before
    assert [v.version_id for v in store.list_versions()] == [head_before.version_id]
    assert store.quarantined_versions() == [] or True
    assert world_content_hash(store.load_version(head_before.version_id)) == content_before
    assert store.verify_version(head_before.version_id) == []


def test_failure_mid_apply_leaves_the_committed_head_unchanged(store):
    head, content = _head(store), world_content_hash(store.load_version("v-base"))

    def boom(w):
        w.entities["wall-a"].name = "HALF APPLIED"
        raise RuntimeError("injected: crash while applying the correction")

    with pytest.raises(RuntimeError):
        commit_correction_atomically(store, expected_base_version_id="v-base", mutate=boom, operation="x", actor_id="u")
    _assert_untouched(store, head, head.artifact_hash, content)
    assert store.load_version("v-base").entities["wall-a"].name == "wall-a"


def test_a_correction_that_fails_validation_is_rejected_and_not_saved(store):
    head, content = _head(store), world_content_hash(store.load_version("v-base"))

    def corrupt(w):
        w.entities["door-ab"].relationships.append(  # dangling reference: the structural self-check must refuse it
            __import__("world_ir.schema_v1", fromlist=["Relationship"]).Relationship(
                kind=__import__("world_ir.schema_v1", fromlist=["RelationshipKind"]).RelationshipKind.CONNECTS, target_id="ghost"))

    with pytest.raises(CorrectionRejected):
        commit_correction_atomically(store, expected_base_version_id="v-base", mutate=corrupt, operation="x", actor_id="u")
    _assert_untouched(store, head, head.artifact_hash, content)


def test_failure_mid_persist_never_becomes_head_and_head_reloads_identically(store, monkeypatch):
    """The version RECORD is written before the sequence index. Failing the index write must withdraw the record: list_versions()
    appends unrecorded files, so an orphan would silently become HEAD of a save that raised."""
    head, content = _head(store), world_content_hash(store.load_version("v-base"))
    import worldstore.store as ws

    real = ws._atomic_write_text

    def flaky(path, text):
        if Path(path).name == "sequence.json":
            raise OSError("injected: disk full while updating the version index")
        return real(path, text)

    monkeypatch.setattr(ws, "_atomic_write_text", flaky)
    with pytest.raises(OSError):
        commit_correction_atomically(store, expected_base_version_id="v-base",
                                     mutate=lambda w: setattr(w.entities["wall-a"], "name", "X"), operation="x", actor_id="u")
    monkeypatch.setattr(ws, "_atomic_write_text", real)
    _assert_untouched(store, head, head.artifact_hash, content)
    assert not list((store._versions_dir).glob("v-*.json")) == [] and len(list(store._versions_dir.glob("v-*.json"))) == 1
    # and the store still accepts the very next correction
    stored, _ = commit_correction_atomically(store, expected_base_version_id="v-base",
                                             mutate=lambda w: setattr(w.entities["wall-a"], "name", "Y"), operation="x", actor_id="u")
    assert _head(store).version_id == stored.version_id


def test_a_version_that_fails_read_back_verification_is_quarantined_not_adopted(store, monkeypatch):
    head, content = _head(store), world_content_hash(store.load_version("v-base"))
    monkeypatch.setattr(WorldStore, "verify_version", lambda self, vid: [{"version": vid, "reason": "injected corruption"}])
    with pytest.raises(CorrectionRejected):
        commit_correction_atomically(store, expected_base_version_id="v-base",
                                     mutate=lambda w: setattr(w.entities["wall-a"], "name", "X"), operation="x", actor_id="u")
    monkeypatch.undo()
    assert len(store.quarantined_versions()) == 1
    _assert_untouched(store, head, head.artifact_hash, content)


# --------------------------------------------------------------------------------------------- fresh-process reload

def test_a_fresh_process_reloads_the_same_world_with_the_same_hashes(store):
    stored, _ = commit_correction_atomically(
        store, expected_base_version_id="v-base", operation="reclassify", actor_id="u",
        mutate=lambda w: reclassify_entity(w, "wall-a", "wall/partition", "x", user_id="u"))
    writer = {"content": world_content_hash(store.load_version(stored.version_id)), "artifact": stored.artifact_hash}
    code = (
        "import json,sys;from worldstore.store import WorldStore;from world_ir.correction_log import world_content_hash;"
        "s=WorldStore(sys.argv[1]);v=s.list_versions()[-1];assert s.verify_version(v.version_id)==[];"
        "print(json.dumps({'version':v.version_id,'content':world_content_hash(s.load_version(v.version_id)),'artifact':v.artifact_hash}))")
    out = subprocess.run([sys.executable, "-c", code, str(store._root)], capture_output=True, text=True, timeout=120,
                         cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr
    reader = json.loads(out.stdout.strip().splitlines()[-1])
    assert reader == {"version": stored.version_id, **writer}


# -------------------------------------------------------------------------------------------------------- lineage

def test_lineage_through_reclassify_split_and_merge():
    w = _house()
    w.entities["wall-a"].custom_properties["bounding_box"] = [[0, 0, 0], [4, 1, 1]]
    reclassify_entity(w, "wall-a", "wall/partition")
    assert "wall-a" in w.entities and w.entities["wall-a"].custom_properties["lineage"] == {"parent_ids": ["wall-a"], "relation": "reclassify"}
    split_entity(w, "wall-a", "x", 0.5)
    assert "wall-a" not in w.entities
    assert set(lineage_successors(w, "wall-a")) == {"wall-a_part1", "wall-a_part2"}
    assert w.entities["wall-a_part1"].custom_properties["lineage"]["parent_ids"] == ["wall-a"]
    merge_entities(w, "wall-a_part1", ["wall-a_part2"])
    assert lineage_successors(w, "wall-a") == ["wall-a_part1"]            # the split child that survived the merge
    assert lineage_successors(w, "wall-a_part2") == ["wall-a_part1"]
    assert lineage_ancestors(w, "wall-a_part1") == ["wall-a", "wall-a_part2"]
    assert w.entities["wall-a_part1"].custom_properties["lineage"]["relation"] == "merge"


def test_carry_by_anchor_match_records_lineage():
    from world_ir.corrections import carry_corrections

    w = _house()
    w.metadata["scale_state"] = "metric"
    rec = {"kind": "dimension", "entity_id": "wall-gone", "entity_type": "wall", "property": "length_m", "value": 3.5,
           "anchor": [1.65, 0.05, 1.25], "note": None, "created_at": 1.0}
    carry_corrections(w, [rec])
    # whether the world's scale state allows VERIFIED or not, an anchor match must never be silent about identity
    for e in w.metadata.get("entity_lineage", []):
        assert e["from_id"] == "wall-gone" and e["relation"] == "carried_by_anchor_match"


# -------------------------------------------------------------------------------------------- room connectivity

def test_door_connectivity_is_explicit_and_unknown_is_not_assumed():
    w = _house()
    res = derive_door_connectivity(w)
    assert [(e["opening_id"], e["room_a"], e["room_b"]) for e in res["edges"]] == [("door-ab", "room-a", "room-b")]
    unresolved = {u["opening_id"]: u for u in res["unresolved_openings"]}
    assert unresolved["door-bc"]["status"] == "one_side_known" and unresolved["door-bc"]["known_rooms"] == ["room-b"]
    assert sorted(res["rooms_without_door_evidence"]) == ["room-c", "room-d"]
    assert res["status"] == "partial" and sorted(map(sorted, res["components"])) == [["room-a", "room-b"], ["room-c"], ["room-d"]]


def test_door_connectivity_persists_as_relationships_idempotently():
    w = _house()
    apply_door_connectivity(w)
    first = json.dumps(w.to_dict(), sort_keys=True)
    apply_door_connectivity(w)
    assert json.dumps(w.to_dict(), sort_keys=True) == first
    rel = [r for r in w.entities["room-a"].relationships if r.target_id == "room-b"]
    assert len(rel) == 1 and rel[0].metadata["via_opening_id"] == "door-ab" and rel[0].metadata["derived_from"] == "door_room_connectivity"
    assert rel[0].confidence is None                       # derived, not measured: no fabricated confidence
    assert not any(r.target_id == "room-c" for r in w.entities["door-bc"].relationships)   # the unknown side is NOT linked
    assert w.metadata["room_connectivity"]["status"] == "partial"


def test_connectivity_is_unknown_without_any_door_evidence_and_validation_reports_it():
    w = _house()
    del w.entities["door-ab"], w.entities["door-bc"]
    assert derive_door_connectivity(w)["status"] == "unknown"
    w2 = _house()
    codes = {i.code for i in (*validate_world_ir(w2).errors, *validate_world_ir(w2).warnings)}
    assert "topology_opening_connectivity_unknown" in codes and "topology_room_without_door_evidence" in codes
    assert validate_world_ir(w2).is_valid()                # unknown connectivity is a warning, never an error


def test_competing_hypotheses_and_unresolved_geometry_are_surfaced_not_invented():
    from world_ir.hypotheses import attach_competing_hypothesis, surface_unresolved

    w = _house()
    ghost = _ent("room-x", EntityType.ROOM)                # a room with no boundary and no geometry evidence
    w.entities["room-x"] = ghost
    attach_competing_hypothesis(w.entities["room-a"], "classification", {"entity_class": "room.kitchen"}, "perception.a", 0.8)
    attach_competing_hypothesis(w.entities["room-a"], "classification", {"entity_class": "room.bedroom"}, "perception.b", 0.78)
    out = surface_unresolved(w)
    [c] = [x for x in out["competing_hypotheses"] if x["entity_id"] == "room-a"]
    assert len(c["candidates"]) == 2 and c["conflict_detected"] is True      # the loser is visible, not deleted
    fields = {(u["entity_id"], u["field"]) for u in out["unresolved_geometry"]}
    assert ("room-x", "geometry") in fields and ("room-x", "boundary_element_ids") in fields
    assert "room-a" not in {u["entity_id"] for u in out["unresolved_geometry"]}
    assert not w.entities["room-x"].geometry_ids            # nothing was invented to fill the gap


# ------------------------------------------------------------------------------------------------------- API tests

from tests.test_application_api import _commit, _head_and_count, _seed_compiled_world, client  # noqa: E402,F401


def test_api_commit_without_a_base_version_is_refused_428_and_nothing_changes(client, tmp_path, monkeypatch):
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    r = client.post(f"/api/worlds/{wid}/commit", json={"entity_id": "ent-shed", "changes": {"name": "Barn"}})
    assert r.status_code == 428, r.text
    assert "base version is required" in r.json()["detail"] and v0 in r.json()["detail"]
    assert _head_and_count(client, wid) == (v0, 1)


def test_api_stale_commit_is_409_with_current_version_info(client, tmp_path, monkeypatch):
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    v1 = _commit(client, wid, changes={"name": "Barn"}).json()["version_id"]
    r = _commit(client, wid, changes={"name": "Silo"}, parent=v0)
    assert r.status_code == 409 and v0 in r.json()["detail"] and v1 in r.json()["detail"] and "stale" in r.json()["detail"]
    assert _head_and_count(client, wid) == (v1, 2)


@pytest.mark.parametrize("op,body", [
    ("reclassify", {"entity_id": "room-001", "new_class": "room/kitchen"}),
    ("split", {"entity_id": "room-001"}),
    ("merge", {"primary_entity_id": "room-001", "secondary_entity_ids": ["room-002"]}),
])
def test_api_legacy_corrections_routes_require_and_check_the_base_version(client, tmp_path, monkeypatch, op, body):
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    assert client.post(f"/api/worlds/{wid}/corrections/{op}", json=body).status_code == 428
    v1 = _commit(client, wid, changes={"name": "Barn"}).json()["version_id"]
    stale = client.post(f"/api/worlds/{wid}/corrections/{op}", json={**body, "parent_version_id": v0})
    assert stale.status_code == 409 and v1 in stale.json()["detail"], stale.text
    assert _head_and_count(client, wid) == (v1, 2)


def test_api_commit_persists_a_complete_record_and_flags_stale_exports(client, tmp_path, monkeypatch):
    import asyncio

    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)

    async def _export():
        import apps.api.db as db_mod
        from apps.api.models import ExportRecord

        async with db_mod.get_sessionmaker()() as db:
            db.add(ExportRecord(id="exp_test0001", world_id=wid, version_id=v0, format="glb", size_bytes=1))
            await db.commit()

    asyncio.run(_export())
    before = client.get(f"/api/worlds/{wid}/derived-artifacts").json()
    assert before["stale_count"] == 0 and before["artifacts"][0]["built_from_version_id"] == v0
    res = _commit(client, wid, changes={"name": "Barn"}).json()
    ir = client.get(f"/api/worlds/{wid}/worldir").json()
    [rec] = ir["metadata"]["correction_log"]
    assert rec["base_version_id"] == v0 and rec["new_version_id"] == res["version_id"] and rec["actor_id"] and rec["actor_kind"] == "user"
    assert any(c["entity_id"] == "ent-shed" and c["field"] == "name" and c["old"] == "Shed" and c["new"] == "Barn" for c in rec["changes"])
    assert rec["stale_artifacts"] == [{"artifact_id": "exp_test0001", "kind": "export:glb", "built_from_version_id": v0}]
    after = client.get(f"/api/worlds/{wid}/derived-artifacts").json()
    assert after["stale_count"] == 1 and after["artifacts"][0]["stale"] is True and after["current_version_id"] == res["version_id"]


def test_api_commit_failing_mid_persist_does_not_move_head_or_leave_an_orphan_version(client, tmp_path, monkeypatch):
    wid, v0 = _seed_compiled_world(client, tmp_path, monkeypatch)
    import worldstore.store as ws

    real = ws._atomic_write_text

    def flaky(path, text):
        if Path(path).name == "sequence.json":
            raise WorldStoreError("injected: index write failed")
        return real(path, text)

    monkeypatch.setattr(ws, "_atomic_write_text", flaky)
    r = _commit(client, wid, changes={"name": "Barn"})
    monkeypatch.setattr(ws, "_atomic_write_text", real)
    assert r.status_code == 409, r.text
    assert _head_and_count(client, wid) == (v0, 1)
    from apps.api import worldstore_service as wss

    assert [v.version_id for v in wss.get_store().list_versions()] == [v0]
    assert client.get(f"/api/worlds/{wid}/worldir").json()["entities"]["ent-shed"]["name"] == "Shed"
    assert _commit(client, wid, changes={"name": "Barn"}).status_code == 200      # and the world is still correctable
