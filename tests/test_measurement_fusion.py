"""B14: measurements fuse by inverse variance, reject outliers, and count correlated sources once."""
import pytest

from reconstruction.fusion.measurements import fuse


def test_weighted_toward_the_precise_measurement():
    r = fuse([4.02, 4.18, 3.72], [0.02, 0.10, 0.30])
    assert r.value == pytest.approx(4.03, abs=0.02)
    assert r.rejected == () and r.used == (0, 1, 2)


def test_wild_outlier_is_rejected_and_listed():
    r = fuse([4.00, 4.01, 3.99, 9.0], [0.05, 0.05, 0.05, 0.05])
    assert r.rejected == (3,) and 3 not in r.used
    assert r.value == pytest.approx(4.0, abs=0.01)


def test_a_group_counts_as_one_measurement():
    independent = fuse([4.0, 4.0, 4.0], [0.1, 0.1, 0.1])
    grouped = fuse([4.0, 4.0, 4.0], [0.1, 0.1, 0.1], group=["m", "m", "m"])
    assert grouped.sigma > independent.sigma
    assert grouped.sigma == pytest.approx(0.1)


def test_unknown_sigma_is_refused_not_invented():
    with pytest.raises(ValueError):
        fuse([1.0, 2.0], [0.1, 0.0])
