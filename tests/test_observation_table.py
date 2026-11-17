"""B02: ObservationTable round-trips, indexes by image/entity, and exports a valid inventory."""
import numpy as np
import pytest

from benchmarks.reality_benchmark.inventory import validate_inventory
from world_ir.observation_table import ObservationTable
from world_ir.observations import DepthSource, Observation, Visibility


def _obs(i, img, pos=(1.0, 2.0, 3.0)):
    return Observation(f"o{i}", img, "furniture/storage/almirah", pos, 0.1, DepthSource.MVS, model="gdino@1",
                       confidence=0.7, bbox_px=(1, 2, 30, 40), visibility=Visibility.PARTIAL)


def _table():
    t = ObservationTable.from_observations([_obs(0, "a"), _obs(1, "a"), _obs(2, "b"), _obs(3, "b"), _obs(4, "c")])
    t.entity_row[:] = [0, 1, 0, 1, 2]
    t.reindex()
    return t


def test_save_load_roundtrip(tmp_path):
    t = _table(); t.save(tmp_path / "t.npz")
    u = ObservationTable.load(tmp_path / "t.npz")
    for k in ("obs_id", "image_id", "class_path", "bbox_px", "world_pos", "entity_row"):
        assert np.array_equal(getattr(t, k), getattr(u, k))


def test_indices():
    t = _table()
    assert t.by_image["a"].tolist() == [0, 1] and t.by_image["c"].tolist() == [4]
    assert t.by_entity[0].tolist() == [0, 2] and t.by_entity[2].tolist() == [4]


def test_observation_roundtrip_and_inventory_is_valid():
    t = _table()
    t.visibility[1] = 3  # Visibility.NOT_OBSERVED
    back = t.to_observations()
    assert back[2].image_id == "b" and back[0].visibility is Visibility.PARTIAL
    assert back[1].visibility is Visibility.NOT_OBSERVED
    assert validate_inventory(t.to_inventory()) == []


def test_unlifted_row_exports_none_with_reason():
    t = ObservationTable.from_observations([_obs(0, "a", pos=(float("nan"),) * 3)])
    (r,) = t.to_inventory()
    assert r["position_3d"] is None and "position_3d" in r["unavailable"]
    assert validate_inventory([r]) == []


def test_to_rows():
    t = _table()
    rows = t.to_rows()
    assert len(rows) == 5
    assert rows[0]["obs_id"] == "o0"
    assert rows[0]["image_id"] == "a"
    assert rows[0]["entity_row"] == 0
    assert rows[0]["world_pos"] == [1.0, 2.0, 3.0]


def test_observations_api_endpoint(tmp_path, monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    from apps.api.auth import Role, issue_scoped_token
    from apps.api.models import World, WorldVersion

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'obs.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_SIGNING_KEY", "obs-secret")

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    # Save table to storage
    from apps.api.storage import store_bytes
    t = _table()
    p = tmp_path / "obs_table.npz"
    t.save(p)
    sha, uri = store_bytes(p.read_bytes())
    art_uri = f"sha256://{sha}"

    # Create world and version in DB
    from apps.api.db import get_sessionmaker
    import asyncio
    maker = get_sessionmaker()

    async def _setup():
        async with maker() as db:
            w = World(id="wld_obs_test", name="Obs Test World", status="active", team_id="tenant-x")
            v = WorldVersion(id="ver_obs_1", world_id="wld_obs_test", report={"observations_artifact_uri": art_uri})
            db.add(w)
            db.add(v)
            await db.commit()

    token = issue_scoped_token("tenant-x", "user-x", Role.EDITOR, secret="obs-secret")
    foreign_token = issue_scoped_token("tenant-y", "user-y", Role.EDITOR, secret="obs-secret")

    with TestClient(main_mod.app) as c:
        asyncio.run(_setup())

        # Successful query
        r = c.get("/api/worlds/wld_obs_test/versions/ver_obs_1/observations", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 5
        assert len(data["items"]) == 5

        # Query filtered by image
        r_img = c.get("/api/worlds/wld_obs_test/versions/ver_obs_1/observations?image_id=a", headers={"Authorization": f"Bearer {token}"})
        assert r_img.status_code == 200
        assert r_img.json()["count"] == 2

        # Query filtered by entity
        r_ent = c.get("/api/worlds/wld_obs_test/versions/ver_obs_1/observations?entity_id=0", headers={"Authorization": f"Bearer {token}"})
        assert r_ent.status_code == 200
        assert r_ent.json()["count"] == 2

        # Cross-tenant query blocked with 404
        r_for = c.get("/api/worlds/wld_obs_test/versions/ver_obs_1/observations", headers={"Authorization": f"Bearer {foreign_token}"})
        assert r_for.status_code == 404
