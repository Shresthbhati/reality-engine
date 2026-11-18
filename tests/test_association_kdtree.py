"""RE-PERF-000011: KD-tree candidate pruning must change nothing about the result, only how many entities are scanned."""
import random

import numpy as np

from perception.instances import association as A
from world_ir.observations import DepthSource, Observation


def _scene(seed: int, n_images: int = 12, per_image: int = 14):
    rng = random.Random(seed)
    objects = [(rng.uniform(0, 12), rng.uniform(0, 3), rng.uniform(0, 12), rng.choice(["chair", "door", "table"]))
               for _ in range(40)]
    obs = []
    for im in range(n_images):
        for k, (x, y, z, label) in enumerate(rng.sample(objects, per_image)):
            sigma = rng.choice([0.03, 0.05, 0.15])
            obs.append(Observation(f"o{im}-{k}", f"img{im}", label,
                                   (x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), z + rng.gauss(0, sigma)),
                                   sigma, DepthSource.METRIC))
    return obs


def _run(obs, kd_min):
    old, A.KD_MIN_ENTITIES = A.KD_MIN_ENTITIES, kd_min
    try:
        r = A.associate(obs)
    finally:
        A.KD_MIN_ENTITIES = old
    return r.assignment, [e.entity_id for e in r.entities], [d.state for d in r.decisions]


def test_pruned_and_full_scan_give_identical_results():
    for seed in (1, 2, 3):
        obs = _scene(seed)
        assert _run(obs, kd_min=0) == _run(obs, kd_min=10**9)                  # always-tree vs never-tree


def test_ball_query_is_a_superset_of_every_entity_that_passes_the_gate():
    obs = _scene(7)
    r = A.associate(obs)
    ents = list(r.entities)
    near = A._near_entities(ents) if len(ents) >= A.KD_MIN_ENTITIES else None
    assert near is not None, "scene should create enough entities to use the tree"
    for o in obs:
        passing = {j for j, e in enumerate(ents) if A._candidate(o, e, None) is not None}
        assert passing <= set(near(o))
