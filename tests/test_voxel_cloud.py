"""B11: voxel fusion lowers variance, tracks sources, and treats missing voxels as not observed."""
import time

import numpy as np

from reconstruction.fusion.voxel_cloud import MIN_OBSERVED, MONO_DEPTH, SFM, VoxelCloud


def _plane(n, noise, seed):
    r = np.random.default_rng(seed)
    p = r.uniform(0, 0.02, (n, 3)); p[:, 2] = 1.0 + r.normal(0, noise, n)
    return p


def test_two_sources_fuse_to_lower_variance_and_set_bits():
    v = VoxelCloud(voxel_m=0.02)
    rgb = np.zeros((200, 3))
    v.integrate(_plane(200, 0.001, 1), rgb, 0.01, SFM)
    var_one = float(v.var_m2.min())
    v.integrate(_plane(200, 0.001, 2), rgb, 0.01, MONO_DEPTH)
    assert float(v.var_m2.min()) < var_one
    assert int(v.sources.max()) == SFM | MONO_DEPTH
    assert v.observed_mask().any() and int(v.count.max()) >= MIN_OBSERVED


def test_empty_space_has_no_voxel():
    v = VoxelCloud()
    v.integrate(_plane(50, 0.0, 3), np.zeros((50, 3)), 0.01, SFM)
    assert v.lookup(np.array([[0.01, 0.01, 1.0]]))[0] >= 0
    assert v.lookup(np.array([[5.0, 5.0, 5.0]]))[0] == -1
    assert VoxelCloud().lookup(np.zeros((2, 3))).tolist() == [-1, -1]


def test_a_million_points_integrate_quickly():
    p = np.random.default_rng(0).uniform(0, 5, (1_000_000, 3))
    t = time.time(); VoxelCloud().integrate(p, np.zeros((len(p), 3)), 0.01, SFM)
    assert time.time() - t < 20
