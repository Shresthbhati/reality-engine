"""The pre-label draft tool: tile coverage, de-duplication and the small-fitting size sanity check (no model needed)."""

from __future__ import annotations

from perception.entities import taxonomy
from scripts.prelabel_images import SMALL_FITTING_MAX_SHARE, _nms, _tiles, plausible_size

SIZE = (1672, 941)


def _cls(prompt):
    return taxonomy.class_for_label(prompt)


def test_tiles_cover_the_whole_frame_with_overlap():
    tiles = _tiles(*SIZE)
    assert min(t[0] for t in tiles) == 0 and min(t[1] for t in tiles) == 0
    assert max(t[2] for t in tiles) == SIZE[0] and max(t[3] for t in tiles) == SIZE[1]
    assert all(t[2] > t[0] and t[3] > t[1] for t in tiles)
    xs = sorted({t[0] for t in tiles})
    assert all(b - a < 640 for a, b in zip(xs, xs[1:]))                                  # neighbouring tiles overlap


def test_a_socket_sized_box_is_kept_and_a_tile_sized_one_is_dropped():
    sock = _cls("wall socket")
    assert sock is not None and sock.small
    assert plausible_size(sock, (880, 340, 930, 380), SIZE)                              # ~0.1% of the frame
    big = (100, 100, 100 + 640, 100 + 640)                                               # a whole tile
    assert not plausible_size(sock, big, SIZE)
    assert (640 * 640) / (SIZE[0] * SIZE[1]) > SMALL_FITTING_MAX_SHARE


def test_large_things_are_never_size_filtered():
    for prompt in ("sofa", "stair tread"):
        cls = _cls(prompt)
        assert cls is not None
        assert plausible_size(cls, (0, 0, 900, 500), SIZE)


def test_nms_keeps_the_stronger_box_and_distinct_objects():
    a = ("c1", 0.9, (0, 0, 100, 100), "coarse")
    dup = ("c1", 0.5, (5, 5, 105, 105), "fine")                       # same class, overlapping: dropped
    other_class_same_pixels = ("c2", 0.4, (2, 2, 100, 100), "fine")   # one object with two names: weaker dropped
    apart = ("c1", 0.6, (300, 300, 380, 380), "fine")
    kept = _nms([dup, a, other_class_same_pixels, apart])
    assert [k[1] for k in kept] == [0.9, 0.6]
