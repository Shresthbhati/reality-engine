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

## Room-fidelity fixtures (plan `docs/superpowers/plans/2026-10-30-room-reconstruction-fidelity.md`, T0.1)

Two fixtures feed `scripts/room_fidelity_report.py`. Only the first exists.

### 1. `fidelity_room` -- synthetic, exists, committed (NOT under `datasets/`: `datasets/*/` is git-ignored)

* Generator: `synthetic/indoor.py` -> `fidelity_room()`, rendered through the RGB-D sensor model by
  `observe(ind, colors="albedo")` (per-point colour). Deterministic and headless: no COLMAP, no images on disk.
* Ground truth: `synthetic/fidelity_room/ground_truth.json` (regenerate with
  `python -c "from synthetic.indoor import write_fidelity_fixture as w; w('synthetic/fidelity_room')"`;
  `tests/test_room_fidelity_fixture.py` fails if the file and the generator disagree).
* Content: a 5.0 x 4.0 x 2.6 m box room (Z up, metres), a door 0.9 x 2.05 m in the south wall, a window 1.2 x 1.1 m
  (sill 0.9) in the north wall, a table 1.2 x 0.8 x 0.75 m and a wardrobe 0.6 x 0.6 x 1.8 m, four distinctly coloured
  walls, floor and ceiling.
* What it does NOT cover: `synthetic/lroom.py` (the COLMAP-rendered L-room with a false ceiling) is still the only
  fixture with an L footprint and two ceiling layers, but it has no door and one uniform wall colour; running it needs
  COLMAP and minutes of rendering (`scripts/run_lroom_e2e.py`), so it is not part of the headless gate.

`ground_truth.json` schema `room_fidelity_ground_truth/1` (all lengths metres, Z up):

| key | meaning |
|---|---|
| `schema`, `fixture`, `source` (`synthetic`\|`real`), `units`, `up_axis`, `scale_basis` | provenance of the numbers |
| `room` | `id`, `length_m` (plan X), `width_m` (plan Y), `height_m`, `floor_z`, `ceiling_z`, `area_m2`, `min`/`max` (plan x,y) |
| `walls[]` | `id`, `side`, `start`/`end` (plan x,y), `length_m`, `height_m`, `color_rgb` |
| `floor`, `ceiling` | `color_rgb` |
| `openings[]` | `kind` (`door`\|`window`), `wall`, `width_m`, `height_m`, `sill_m`, `center` (x,y,z) |
| `furniture[]` | `ref`, `min`/`max` (x,y,z), `size_m`, `color_rgb` |

### 2. `real_room` -- NOT PRESENT. The user must supply it; it is not fabricated.

Needed from the user (all are measurements of one real room, no estimates):

1. **The photo set** actually used in testing (or a fresh capture of the same room): all frames, original files, with
   overlap between neighbouring shots. A video is fine if the frame extraction settings are stated.
2. **Tape-measured room length, width and height** (floor-level wall-to-wall, and floor to ceiling), in metres, noting
   where each was measured. If the ceiling has steps or a false ceiling, the height of each layer.
3. **Door and window sizes**: width, height and sill height of each, and which wall each is on.
4. **One camera-to-camera baseline**: the tape-measured distance between the positions of two of the photos (name the two
   files). This is the only way the world becomes METRIC (Decision 023); without it every cm figure is `model_units`.
5. Optionally: the dominant colour of each wall, the floor and the ceiling (a phone photo of a colour card is better
   than a guess), and the rough size of the larger furniture pieces.

Store it as `datasets/real_room/images/*` plus `datasets/real_room/ground_truth.json` in the same
`room_fidelity_ground_truth/1` schema with `"source": "real"` (the folder stays git-ignored: keep photos out of git).
Until that exists, `python scripts/room_fidelity_report.py --fixture real` prints every metric as `not available`.
