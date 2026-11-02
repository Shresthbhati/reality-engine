"""CE-09 failure modes at the WorldStore / adoption boundary: disk-full, corrupted artifact, and two parallel jobs
that started from the same HEAD (stale-candidate rebase).

Same harness and invariants as tests/test_adoption_transaction.py (real sqlite DB + real WorldStore + real WorldIR):
  * HEAD unchanged after a failed attempt, DB mirror == WorldStore lineage (no phantom / duplicate versions)
  * a clear error is surfaced, and a retry once the fault is gone converges on exactly one new version
"""

from __future__ import annotations

import asyncio
import errno
import os
from pathlib import Path

import pytest

from tests.test_adoption_transaction import WID, _boot, _state, _v1, _world, env, patch  # noqa: F401 -- harness


def _flip_bytes(store, version_id):
    rec = store._record(version_id)
    f = store._store._path_for(rec["artifact_hash"])
    good = f.read_bytes()
    f.write_bytes(b"CORRUPTED" + good[9:])
    return f, good


# ----------------------------------------------------------------------------------------------------- disk full


def test_disk_full_while_writing_the_artifact_fails_cleanly_and_the_retry_converges(env, patch):
    db_mod, ws, _ = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)
        root = ws.get_store()._root
        real = Path.write_bytes

        def half_then_enospc(self, data):
            if root in self.parents:                      # a partial write that then runs out of space
                real(self, data[: len(data) // 2])
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            return real(self, data)

        patch.setattr(Path, "write_bytes", half_then_enospc)
        async with maker() as db:
            with pytest.raises(OSError) as ei:
                await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)
        assert ei.value.errno == errno.ENOSPC
        patch.undo()
        head, rows, lineage, _ = await _state(ws, maker)
        assert (head, rows, lineage) == (v1, [v1], [v1]), "a failed write must not adopt or leave history behind"
        assert not list((root / "adoption-intents").glob("*.json")), "the failed attempt left an intent behind"

        # the fault is gone: the SAME content must now commit and read back clean (a half-written blob must
        # not poison the content-addressed store forever)
        async with maker() as db:
            v2 = (await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)).id
        head, rows, lineage, _ = await _state(ws, maker)
        assert head == v2 and rows == lineage == sorted([v1, v2])
        assert not ws.get_store().verify_version(v2) and ws.get_store().load_version(v2).entities["e1"].name == "V2"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------------------------- corrupted artifact


def test_a_corrupted_head_artifact_is_reported_blocks_new_commits_and_adopts_nothing_until_repaired(env):
    from worldstore.store import WorldStoreError

    db_mod, ws, _ = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)
        store = ws.get_store()
        f, good = _flip_bytes(store, v1)

        assert store.verify_version(v1)[0]["reason"].startswith("hash mismatch")
        async with maker() as db:
            with pytest.raises(WorldStoreError, match="hash mismatch"):
                await ws.get_current_worldir(db, WID)            # reads surface the corruption, never serve it
        async with maker() as db:
            with pytest.raises(WorldStoreError, match="hash mismatch"):
                await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)
        head, rows, lineage, _ = await _state(ws, maker)
        assert (head, rows, lineage) == (v1, [v1], [v1]), "a commit on a corrupt base must adopt nothing"
        assert not list((store._root / "adoption-intents").glob("*.json"))

        f.write_bytes(good)                                      # repaired (restored from backup)
        async with maker() as db:
            v2 = (await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)).id
        head, rows, lineage, _ = await _state(ws, maker)
        assert head == v2 and rows == lineage == sorted([v1, v2]) and store.parents(v2) == [v1]
        assert not store.verify_version(v1) and not store.verify_version(v2)

    asyncio.run(scenario())


def test_a_corrupted_old_ancestor_is_detected_without_blocking_work_on_a_healthy_head(env):
    db_mod, ws, _ = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)
        async with maker() as db:
            v2 = (await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)).id
        store = ws.get_store()
        _flip_bytes(store, v1)
        assert store.verify_version(v1) and not store.verify_version(v2)
        async with maker() as db:
            assert (await ws.get_current_worldir(db, WID)).entities["e1"].name == "V2"
            v3 = (await ws.commit_version(db, world_id=WID, world=_world("V3"), parent=v2, expect_parent=v2)).id
        head, rows, lineage, _ = await _state(ws, maker)
        assert head == v3 and rows == lineage == sorted([v1, v2, v3])

    asyncio.run(scenario())


def test_a_corrupted_blob_shared_by_identical_content_is_healed_by_the_next_save(env):
    """Content-addressed: a later version with the same bytes must not trust a rotten blob already on disk."""
    db_mod, ws, _ = env
    store = ws.get_store()
    a = store.save_version(_world("same"), parent=None).version_id
    _flip_bytes(store, a)
    b = store.save_version(_world("same"), parent=None).version_id
    assert not store.verify_version(b), "the new version points at the corrupt blob it never rewrote"


# ------------------------------------------------------------------- two parallel jobs from one HEAD (stale base)


def test_two_parallel_jobs_from_one_head_adopt_exactly_one_and_the_loser_rebases_onto_the_winner(env):
    db_mod, ws, _ = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)

        async def job(label):
            async with maker() as db:                              # both jobs read HEAD = V1 and build from it
                try:
                    return "ok", (await ws.commit_version(db, world_id=WID, world=_world(label), parent=v1,
                                                          expect_parent=v1)).id
                except ws.ConcurrentModificationError as exc:
                    return "stale", exc

        r1, r2 = await asyncio.gather(job("job-1"), job("job-2"))
        assert sorted([r1[0], r2[0]]) == ["ok", "stale"], f"exactly one parallel job may adopt: {r1}, {r2}"
        winner = (r1 if r1[0] == "ok" else r2)[1]
        loser = (r1 if r1[0] == "stale" else r2)[1]
        won_label = ws.get_store().load_version(winner).entities["e1"].name
        head, rows, lineage, quarantined = await _state(ws, maker)
        assert head == winner and rows == lineage == sorted([v1, winner]), "no phantom / duplicate version"
        assert quarantined == [loser.orphan_version_id] and loser.current_head == winner

        # the loser rebases: reload HEAD, rebuild on it -> chained onto the WINNER (not forked from V1), once
        lost_label = "job-2" if won_label == "job-1" else "job-1"
        async with maker() as db:
            v3 = (await ws.commit_version(db, world_id=WID, world=_world(lost_label), parent=winner,
                                          expect_parent=winner)).id
        head, rows, lineage, _ = await _state(ws, maker)
        assert head == v3 and rows == lineage == sorted([v1, winner, v3])
        assert ws.get_store().ancestors(v3) == [v1, winner]
        # a duplicate retry of the same rebased content is deduplicated, not minted again
        async with maker() as db:
            again = (await ws.commit_version(db, world_id=WID, world=_world(lost_label), parent=v3)).id
        assert again == v3 and (await _state(ws, maker))[1] == sorted([v1, winner, v3])

    asyncio.run(scenario())
