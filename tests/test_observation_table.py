"""B02: ObservationTable round-trips, indexes by image/entity, and exports a valid inventory."""
import numpy as np

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
    back = t.to_observations()
    assert back[2].image_id == "b" and back[0].visibility is Visibility.PARTIAL
    assert validate_inventory(t.to_inventory()) == []


def test_unlifted_row_exports_none_with_reason():
    t = ObservationTable.from_observations([_obs(0, "a", pos=(float("nan"),) * 3)])
    (r,) = t.to_inventory()
    assert r["position_3d"] is None and "position_3d" in r["unavailable"]
    assert validate_inventory([r]) == []
