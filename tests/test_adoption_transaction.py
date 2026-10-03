"""The adoption boundary: WorldStore (files) and the application DB (HEAD pointer + version mirror) must agree.

Adoption writes two systems that cannot commit atomically. The protocol under test:

    intent recorded -> WorldStore version written -> version verified (hash read-back) -> DB HEAD + mirror row
    committed in ONE transaction -> intent cleared

and the three invariants that must hold after ANY fault at ANY of those steps (and after the recovery that runs at
startup):

    I1  the DB never names a version the WorldStore cannot reconstruct (verified BEFORE the pointer moves)
    I2  the WorldStore never keeps a version in its lineage that the DB did not adopt (such an orphan is
        QUARANTINED, not mirrored into the version list as a phantom, and not left to be duplicated by a retry)
    I3  a retry after a failed attempt adds exactly one version, chained onto the real HEAD

Real DB (sqlite file), real WorldStore, real WorldIR. A hard process kill is modelled by a BaseException the code
under test must not absorb (no cleanup runs), followed by the same recovery call the API runs at startup.
"""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

WID = "wld_adopt01"


class SimulatedKill(BaseException):
    """The process died. BaseException: no ``except Exception`` in the code under test may run cleanup."""


def _kill_at(stage):
    def hook(s):
        if s == stage:
            raise SimulatedKill(stage)
    return hook


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.worldstore_service as ws

    ws._stores.clear()
    yield db_mod, ws, tmp_path
    asyncio.run(db_mod.dispose_db())


@pytest.fixture()
def patch():
    """A MonkeyPatch whose ``undo()`` reverts ONLY the faults a test injected -- the ``env`` fixture's environment
    variables belong to the test's other MonkeyPatch and must survive it."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _world(label: str):
    from provenance import Provenance
    from world_ir import Entity, EntityType
    from world_ir.world_v1 import WorldIR

    wir = WorldIR(id=WID)
    wir.entities["e1"] = Entity(id="e1", type=EntityType.STRUCTURE, name=label,
                                provenance=Provenance.RECONSTRUCTED, confidence=0.8)
    return wir


async def _boot(db_mod):
    from apps.api.models import World

    await db_mod.init_db()
    maker = db_mod.get_sessionmaker()
    async with maker() as db:
        db.add(World(id=WID, name="adopt"))
        await db.commit()
    return maker


async def _v1(ws, maker):
    async with maker() as db:
        row = await ws.commit_version(db, world_id=WID, world=_world("V1"), parent=None)
        return row.id


async def _state(ws, maker):
    """(DB head, DB mirror rows, WorldStore lineage, quarantined) -- the two systems side by side."""
    from sqlalchemy import select

    from apps.api.models import World, WorldVersion

    async with maker() as db:
        head = (await db.get(World, WID)).current_version_id
        rows = sorted((await db.execute(select(WorldVersion.id).where(WorldVersion.world_id == WID))).scalars().all())
    store = ws.get_store()
    lineage = sorted(v.version_id for v in store.list_versions() if v.world_id == WID)
    return head, rows, lineage, sorted(store.quarantined_versions())


# ------------------------------------------------------------------------------------------------ hard kill


@pytest.mark.parametrize("where", ["between the WorldStore write and the DB commit", "inside the DB transaction"])
def test_a_hard_kill_during_adoption_leaves_head_untouched_and_recovery_quarantines_the_orphan(env, patch, where):
    db_mod, ws, _ = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)
        if where.startswith("between"):
            patch.setattr(ws, "_adoption_hook", _kill_at("stored"))
        else:
            patch.setattr(ws, "_mirror_row", lambda *a, **k: (_ for _ in ()).throw(SimulatedKill()))
        async with maker() as db:
            with pytest.raises(SimulatedKill):
                await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)
        patch.undo()
        head, rows, lineage, _ = await _state(ws, maker)
        assert head == v1 and rows == [v1], "HEAD / mirror moved although nothing was adopted"
        assert len(lineage) == 2, "the orphan exists on disk until recovery runs (that window is what recovery closes)"

        async with maker() as db:                                  # what the API does at startup
            report = await ws.reconcile_adoptions(db)
        assert report["quarantined"] and not report["errors"], report
        head, rows, lineage, quarantined = await _state(ws, maker)
        assert (head, rows, lineage) == (v1, [v1], [v1]), "I2: the lineage still holds a version the DB never adopted"
        assert len(quarantined) == 1
        async with maker() as db:                                  # and the listing route's reconcile does not resurrect it
            await ws.resync_versions(db, WID)
        assert (await _state(ws, maker))[1] == [v1]

        # I3: a retry adds exactly one version, chained onto V1
        async with maker() as db:
            v2 = (await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)).id
        head, rows, lineage, quarantined = await _state(ws, maker)
        assert head == v2 and rows == sorted([v1, v2]) and lineage == sorted([v1, v2]) and len(quarantined) == 1
        assert ws.get_store().parents(v2) == [v1]
        assert not ws.get_store().verify_version(v2) and not ws.get_store().verify_version(v1)

    asyncio.run(scenario())


def test_a_hard_kill_after_the_commit_keeps_the_adopted_version_and_recovery_only_clears_the_intent(env, patch):
    db_mod, ws, tmp = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)
        patch.setattr(ws, "_adoption_hook", _kill_at("committed"))
        async with maker() as db:
            with pytest.raises(SimulatedKill):
                await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)
        patch.undo()
        head, rows, lineage, _ = await _state(ws, maker)
        assert head != v1 and rows == sorted([v1, head]) == lineage, "the commit was durable: V2 is adopted"
        assert list((tmp / "ws" / "adoption-intents").glob("*.json")), "the stale intent is still on disk"
        async with maker() as db:
            report = await ws.reconcile_adoptions(db)
        assert report["cleared"] and not report["quarantined"], report
        assert (await _state(ws, maker))[:3] == (head, sorted([v1, head]), sorted([v1, head]))
        assert not list((tmp / "ws" / "adoption-intents").glob("*.json"))

    asyncio.run(scenario())


# ----------------------------------------------------------------------------- in-process faults (no kill)


class _Boom(RuntimeError):
    pass


@pytest.mark.parametrize("fault", ["mirror row", "db commit", "read-back verification", "version store write"])
def test_an_in_process_fault_never_adopts_and_never_leaves_an_orphan(env, patch, fault):
    db_mod, ws, _ = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)
        store = ws.get_store()
        async with maker() as db:
            if fault == "mirror row":
                patch.setattr(ws, "_mirror_row", lambda *a, **k: (_ for _ in ()).throw(_Boom("row")))
            elif fault == "db commit":
                async def bad_commit():
                    raise _Boom("commit")
                patch.setattr(db, "commit", bad_commit)
            elif fault == "read-back verification":
                patch.setattr(store.__class__, "verify_version", lambda self, vid: [{"reason": "hash mismatch"}])
            else:
                patch.setattr(store.__class__, "save_version", lambda *a, **k: (_ for _ in ()).throw(_Boom("disk")))
            with pytest.raises(Exception):
                await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)
            # The caller (the job runner) now commits its OWN failure bookkeeping on this very session. That must
            # not be able to persist a HEAD move that did not happen.
            patch.undo()
            try:
                await db.commit()
            except Exception:
                await db.rollback()
        head, rows, lineage, _ = await _state(ws, maker)
        assert head == v1 and rows == [v1] and lineage == [v1], (fault, head, rows, lineage)
        async with maker() as db:
            v2 = (await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)).id
        head, rows, lineage, _ = await _state(ws, maker)
        assert head == v2 and rows == lineage == sorted([v1, v2])

    asyncio.run(scenario())


def test_a_concurrent_head_move_orphans_nothing_and_the_error_still_names_the_loser(env):
    db_mod, ws, _ = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)
        async with maker() as db:
            v2 = (await ws.commit_version(db, world_id=WID, world=_world("winner"), parent=v1, expect_parent=v1)).id
        async with maker() as db:                                           # a slower writer still claims V1 as its base
            with pytest.raises(ws.ConcurrentModificationError) as ei:
                await ws.commit_version(db, world_id=WID, world=_world("loser"), parent=v1, expect_parent=v1)
        assert ei.value.current_head == v2 and ei.value.orphan_version_id
        head, rows, lineage, quarantined = await _state(ws, maker)
        assert head == v2 and rows == lineage == sorted([v1, v2]), "the loser must not stay in the lineage"
        assert quarantined == [ei.value.orphan_version_id]

    asyncio.run(scenario())


def test_a_version_written_outside_the_api_is_still_mirrored_it_was_never_an_adoption_attempt(env):
    """The CLI (``reality compile`` / ``store save``) writes straight into the store; that is legitimate history."""
    db_mod, ws, _ = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)
        outside = ws.get_store().save_version(_world("cli"), parent=v1).version_id
        async with maker() as db:
            report = await ws.reconcile_adoptions(db)
            await ws.resync_versions(db, WID)
        assert not report["quarantined"], report
        head, rows, lineage, quarantined = await _state(ws, maker)
        assert head == v1 and rows == lineage == sorted([v1, outside]) and not quarantined

    asyncio.run(scenario())


def test_reconciliation_is_idempotent_and_a_second_recovery_changes_nothing(env, patch):
    db_mod, ws, _ = env

    async def scenario():
        maker = await _boot(db_mod)
        v1 = await _v1(ws, maker)
        patch.setattr(ws, "_adoption_hook", _kill_at("stored"))
        async with maker() as db:
            with pytest.raises(SimulatedKill):
                await ws.commit_version(db, world_id=WID, world=_world("V2"), parent=v1, expect_parent=v1)
        patch.undo()
        async with maker() as db:
            first = await ws.reconcile_adoptions(db)
            second = await ws.reconcile_adoptions(db)
        assert len(first["quarantined"]) == 1 and second == {"quarantined": [], "cleared": [], "errors": []}
        assert (await _state(ws, maker))[:3] == (v1, [v1], [v1])

    asyncio.run(scenario())


def test_quarantine_moves_the_record_out_of_the_lineage_and_deletes_nothing(env):
    db_mod, ws, tmp = env
    store = ws.get_store()
    a = store.save_version(_world("a"), parent=None).version_id
    b = store.save_version(_world("b"), parent=a).version_id
    record_before = json.loads((tmp / "ws" / "versions" / f"{b}.json").read_text())
    store.quarantine_version(b)
    assert [v.version_id for v in store.list_versions()] == [a]
    assert store.quarantined_versions() == [b]
    assert json.loads((tmp / "ws" / "quarantine" / f"{b}.json").read_text()) == record_before
    with pytest.raises(Exception):
        store.load_version(b)
    assert not store.verify_version(a)
    store.quarantine_version(b)                                    # idempotent: already quarantined, no error
    with pytest.raises(Exception):
        store.quarantine_version("v-doesnotexist")
