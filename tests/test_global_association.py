"""B08: strongest evidence merges first; one object never appears twice in one photo; input order is irrelevant."""
import itertools

import numpy as np

from perception.instances.global_association import cluster_observations


def test_five_views_of_one_socket_become_one_entity():
    imgs = [f"im{i}" for i in range(5)]
    pairs = [(i, j, 3.0) for i, j in itertools.combinations(range(5), 2)]
    assert len(set(cluster_observations(imgs, pairs).tolist())) == 1


def test_two_identical_sockets_apart_stay_two():
    imgs = ["a", "b", "a", "b"]                       # rows 0,1 = socket X ; rows 2,3 = socket Y
    ent = cluster_observations(imgs, [(0, 1, 5.0), (2, 3, 5.0), (0, 3, 0.5)])
    assert ent[0] == ent[1] and ent[2] == ent[3] and ent[0] != ent[2]


def test_same_image_rows_never_merge_even_with_high_score():
    ent = cluster_observations(["a", "a"], [(0, 1, 99.0)])
    assert ent[0] != ent[1]


def test_input_order_does_not_change_the_partition():
    imgs = ["a", "b", "c", "d", "a", "b"]
    pairs = [(0, 1, 4.0), (1, 2, 3.0), (2, 3, 2.0), (4, 5, 4.0), (3, 4, 0.2), (0, 5, 1.0)]
    base = cluster_observations(imgs, pairs)
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = [pairs[i] for i in rng.permutation(len(pairs))]
        e = cluster_observations(imgs, p)
        assert np.array_equal(e == e[:, None], base == base[:, None])


def test_a_part_cannot_join_a_part_of_a_different_whole():
    # rows 0,1 wholes (same object, two photos); rows 2,3 parts of whole 0 and whole 1 respectively
    ent = cluster_observations(["a", "b", "a", "b"], [(0, 1, 5.0), (2, 3, 4.0)], parent_obs=[-1, -1, 0, 1])
    assert ent[0] == ent[1] and ent[2] == ent[3]
    # parts whose wholes were never merged must stay apart
    ent2 = cluster_observations(["a", "b", "a", "b"], [(2, 3, 4.0)], parent_obs=[-1, -1, 0, 1])
    assert ent2[2] != ent2[3]
