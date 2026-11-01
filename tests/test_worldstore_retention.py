"""Retention: prune_quarantine keeps live lineage, removes only old quarantined records and unreferenced blobs."""

import os
import time

import pytest

from provenance import Provenance
from world_ir import Entity, EntityType
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore

DAY = 86400


def _world(label):
    w = WorldIR(id="w-1")
    w.entities["e1"] = Entity(id="e1", type=EntityType.STRUCTURE, name=label,
                              provenance=Provenance.RECONSTRUCTED, confidence=0.8)
    return w


def _age(path, days):
    t = time.time() - days * DAY
    os.utime(path, (t, t))


@pytest.fixture
def store(tmp_path):
    s = WorldStore(tmp_path)
    a = s.save_version(_world("a"), parent=None).version_id
    b = s.save_version(_world("b"), parent=a).version_id          # HEAD
    o1 = s.save_version(_world("orphan-old"), parent=b).version_id
    o2 = s.save_version(_world("orphan-new"), parent=b).version_id
    s.quarantine_version(o1)
    s.quarantine_version(o2)
    _age(tmp_path / "quarantine" / f"{o1}.json", 90)
    return s, a, b, o1, o2, tmp_path


def _blob(s, vid_dir_record):
    return s._store._path_for(vid_dir_record["artifact_hash"])


def test_prune_removes_only_old_quarantined_and_keeps_live_lineage(store):
    s, a, b, o1, o2, root = store
    old_blob_digest = s._store.put(b"orphan-only-bytes")[1]
    _age(s._store._path_for(old_blob_digest), 90)
    out = s.prune_quarantine(max_age=30 * DAY, keep_last=0, dry_run=False)
    assert out["quarantine_removed"] == [o1]               # o2 is recent
    assert old_blob_digest in out["blobs_removed"]
    assert s.quarantined_versions() == [o2]
    assert [v.version_id for v in s.list_versions()] == [a, b]
    assert not s.verify_version(a) and not s.verify_version(b)
    assert s.load_version(b) is not None


def test_dry_run_changes_nothing_and_is_the_default(store):
    s, a, b, o1, o2, root = store
    before = sorted(str(p) for p in root.rglob("*") if p.is_file())
    out = s.prune_quarantine(max_age=30 * DAY, keep_last=0)
    assert out["dry_run"] and out["quarantine_removed"] == [o1]
    assert sorted(str(p) for p in root.rglob("*") if p.is_file()) == before


def test_keep_last_protects_the_newest_quarantined(store):
    s, a, b, o1, o2, root = store
    _age(root / "quarantine" / f"{o2}.json", 90)
    out = s.prune_quarantine(max_age=30 * DAY, keep_last=1, dry_run=False)
    assert len(out["quarantine_removed"]) == 1 and len(s.quarantined_versions()) == 1


def test_referenced_blobs_survive_even_when_old(store):
    s, a, b, o1, o2, root = store
    for vid in (a, b):
        _age(_blob(s, s._record(vid)), 365)
    geom_digest = s._store.put(b"geometry-bytes")[1]
    _age(s._store._path_for(geom_digest), 365)
    # a retained world referencing the geometry blob through an artifact:// uri keeps it alive
    w = _world("with-geometry")
    w.metadata["note"] = f"artifact://{geom_digest}"
    c = s.save_version(w, parent=b).version_id
    _age(_blob(s, s._record(c)), 365)
    out = s.prune_quarantine(max_age=30 * DAY, keep_last=0, dry_run=False)
    assert geom_digest not in out["blobs_removed"]
    assert s._store._path_for(geom_digest).exists()
    assert s.load_version(c) is not None and s.load_version(a) is not None


def test_a_quarantined_version_that_is_a_live_parent_is_protected(store):
    s, a, b, o1, o2, root = store
    import json
    rec = s._record(b)
    rec["parent"] = o1                                   # simulate a legacy chain through a quarantined id
    (root / "versions" / f"{b}.json").write_text(json.dumps(rec))
    assert s.prune_quarantine(max_age=30 * DAY, keep_last=0, dry_run=False)["quarantine_removed"] == []


def test_idempotent(store):
    s, *_ = store
    first = s.prune_quarantine(max_age=30 * DAY, keep_last=0, dry_run=False)
    second = s.prune_quarantine(max_age=30 * DAY, keep_last=0, dry_run=False)
    assert first["quarantine_removed"] and second["quarantine_removed"] == [] and second["blobs_removed"] == []


def test_pending_intent_protects_a_quarantined_version(store):
    s, a, b, o1, o2, root = store
    (root / "adoption-intents").mkdir()
    (root / "adoption-intents" / f"{o1}.json").write_text("{}")
    assert s.prune_quarantine(max_age=30 * DAY, keep_last=0, dry_run=False)["quarantine_removed"] == []


def test_invalid_arguments_refused(store):
    with pytest.raises(Exception):
        store[0].prune_quarantine(max_age=-1)
