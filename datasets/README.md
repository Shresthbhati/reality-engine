# datasets

Only one dataset is committed here: `south_building/`. Everything else under
`datasets/*/` is deliberately git-ignored (large, machine-specific captures;
see `.gitignore`).

## south_building — the real-photograph golden dataset

* 32 real photographs (1024 px derivatives of the public COLMAP "South
  Building" example set, UNC Chapel Hill; provided by Christopher Zach per the
  COLMAP datasets page). Source URL, original SHA-256, per-image SHA-256 and the
  derivative recipe are in `south_building/MANIFEST.json`; it is regenerable
  with `scripts/fetch_south_building.py`.
* Rights: public research data distributed by the COLMAP authors for exactly
  this purpose.
* Used by `tests/integration/test_progressive_product_journey.py` through fixed
  windows into the name-sorted image list, each chosen from the measured
  pairwise SIFT/RANSAC match matrix (see `docs/PROGRESSIVE_RECONSTRUCTION.md`):
  a single photo, six strongly overlapping views, +4 and +10 further views,
  and a wide-baseline set on which real COLMAP genuinely finds no initial pair.

## Not present (so not claimed)

There is **no** real photographic corridor dataset and **no** real photographic
room dataset in this repository. Any `room_capture`-style dataset referenced in
older notes is synthetic/rendered, is not committed, and backs no claim in the
progressive-reconstruction work.
