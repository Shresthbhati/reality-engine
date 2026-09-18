"""Cross-checks SpatialIndex's grid-accelerated nearest()/within_radius()
against a brute-force reference implementation over randomized worlds,
across several chunk_size choices (including ones that force multiple
grid-expansion rings and ones far larger/smaller than the data spread).

This is the correctness net for the city-scale spatial-index rewrite
(engine/scene_graph/spatial_index.py): the existing test_spatial_index.py
covers the API contract with small, hand-picked worlds; this file proves
the grid gives the *same answers* as an O(n) scan would, at a scale and
randomness that hand-picked cases can't cover.
"""

from __future__ import annotations

import math
import random

from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR

from engine.scene_graph.spatial_index import SpatialIndex


def _random_world(n: int, spread: float, seed: int) -> WorldIR:
    rng = random.Random(seed)
    world = WorldIR()
    for i in range(n):
        x, y, z = (rng.uniform(-spread, spread) for _ in range(3))
        world.entities[f"e{i}"] = Entity(
            id=f"e{i}",
            type=EntityType.DEBRIS if i % 2 == 0 else EntityType.WALL,
            transform={"position": {"x": x, "y": y, "z": z}},
        )
    return world


def _brute_nearest(world: WorldIR, point, k: int, predicate=None):
    candidates = []
    for entity in world.entities.values():
        if predicate is not None and not predicate(entity):
            continue
        p = entity.transform["position"]
        d = math.dist(point, (p["x"], p["y"], p["z"]))
        candidates.append((entity.id, d))
    candidates.sort(key=lambda pair: (pair[1], pair[0]))
    return candidates[:k]


def _brute_within_radius(world: WorldIR, point, radius: float, predicate=None):
    candidates = []
    for entity in world.entities.values():
        if predicate is not None and not predicate(entity):
            continue
        p = entity.transform["position"]
        d = math.dist(point, (p["x"], p["y"], p["z"]))
        if d <= radius:
            candidates.append((entity.id, d))
    candidates.sort(key=lambda pair: (pair[1], pair[0]))
    return candidates


def test_nearest_matches_brute_force_across_chunk_sizes():
    world = _random_world(n=300, spread=50.0, seed=1)
    point = (3.0, -7.0, 12.0)
    for chunk_size in (0.5, 5.0, 25.0, 500.0):
        index = SpatialIndex(world, chunk_size=chunk_size)
        for k in (1, 5, 50, 1000):  # 1000 > n exercises the "fewer than k exist" path
            got = [(e.id, d) for e, d in index.nearest(point, k=k)]
            want = _brute_nearest(world, point, k=k)
            assert [i for i, _ in got] == [i for i, _ in want], (chunk_size, k)
            for (_, gd), (_, wd) in zip(got, want):
                assert math.isclose(gd, wd, rel_tol=1e-9, abs_tol=1e-9)


def test_nearest_with_predicate_matches_brute_force():
    world = _random_world(n=200, spread=30.0, seed=2)
    point = (0.0, 0.0, 0.0)
    predicate = lambda e: e.type == EntityType.WALL
    for chunk_size in (1.0, 10.0, 100.0):
        index = SpatialIndex(world, chunk_size=chunk_size)
        got = [e.id for e, _ in index.nearest(point, k=10, predicate=predicate)]
        want = [i for i, _ in _brute_nearest(world, point, k=10, predicate=predicate)]
        assert got == want, chunk_size


def test_within_radius_matches_brute_force_across_chunk_sizes():
    world = _random_world(n=300, spread=50.0, seed=3)
    point = (5.0, 5.0, -5.0)
    for chunk_size in (0.5, 5.0, 25.0, 500.0):
        index = SpatialIndex(world, chunk_size=chunk_size)
        for radius in (0.1, 10.0, 60.0, 200.0):
            got = sorted(e.id for e, _ in index.within_radius(point, radius=radius))
            want = sorted(i for i, _ in _brute_within_radius(world, point, radius=radius))
            assert got == want, (chunk_size, radius)


def test_auto_chunk_size_agrees_with_explicit_default():
    # No explicit chunk_size -> must still match brute force.
    world = _random_world(n=150, spread=20.0, seed=4)
    point = (1.0, 1.0, 1.0)
    index = SpatialIndex(world)
    got = [e.id for e, _ in index.nearest(point, k=7)]
    want = [i for i, _ in _brute_nearest(world, point, k=7)]
    assert got == want


def test_clustered_world_forces_multi_ring_expansion():
    # Two tight clusters far apart -- querying near one cluster with a
    # small chunk_size forces nearest() to expand several rings before
    # it can prove correctness (points in the other cluster are decoys).
    world = WorldIR()
    rng = random.Random(5)
    for i in range(20):
        x, y, z = (rng.uniform(-0.5, 0.5) for _ in range(3))
        world.entities[f"near{i}"] = Entity(
            id=f"near{i}", transform={"position": {"x": x, "y": y, "z": z}},
        )
    for i in range(20):
        x, y, z = (1000.0 + rng.uniform(-0.5, 0.5) for _ in range(3))
        world.entities[f"far{i}"] = Entity(
            id=f"far{i}", transform={"position": {"x": x, "y": y, "z": z}},
        )
    index = SpatialIndex(world, chunk_size=0.1)
    got = [e.id for e, _ in index.nearest((0.0, 0.0, 0.0), k=5)]
    assert all(i.startswith("near") for i in got)
    want = [i for i, _ in _brute_nearest(world, (0.0, 0.0, 0.0), k=25)]
    assert set(got) <= set(want[:20])
