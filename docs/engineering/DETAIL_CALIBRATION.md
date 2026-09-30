# Detail-threshold calibration (P7-05) -- what was measured, what was decided, what is still unknown

Data: the one real photographic dataset in the repo (`datasets/south_building`, 32 photographs, 49,608 sparse points,
COLMAP's arbitrary scale). Script: `scripts/measure_detail_thresholds.py`; raw output:
`docs/engineering/detail_threshold_measurements.json` (run 2026-09-30, one-factor-at-a-time around the defaults).
There is no ground truth for "which details exist", so false positives and missed details are NOT scored.

## Measured

| factor (others at default) | values | ROIs | refined / refused | reading |
|---|---|---|---|---|
| voxel size (fraction of scene extent) | 0.01 / 0.02 / 0.04 / 0.08 | 14 / 8 / 1 / 1 | all refined, 0 refused | **cliff** between 0.02 and 0.04 |
| curvature threshold | 0.05 / 0.1 / 0.2 | 1 / 1 / 4 | all refined | 0.05 = 0.1 (plateau); 0.2 is a different regime |
| planarity threshold | 0.90 / 0.95 / 0.98 | 1 / 1 / 1 | all refined | **plateau** -- insensitive on this scene |

Refinement never refused on this data (0 refusals in 8 runs), and the median plane residual was 0.07%-0.7% of the scene
extent, tighter for smaller voxels. The evidence-quality report measured GSD 1.58 "mm/px" and tier `fine` -- but the
model is not metric, so that number is in model-unit-millimetres and the GSD->level bands (L4 <= 5, L2 <= 25, L1 <= 100
mm/px) are **not meaningful without a metric anchor**.

## Decisions

1. **Voxel size must be scene-relative, not 1.0 "metre".** The absolute default (1.0) silently assumes a metric world.
   On this scene (extent 67.9 units) 1.0 is 1.5% of the extent -- the dense-ROI regime -- while on a 3 m room it would be
   a third of the scene. Recommended band when scale is unknown: 1-2% of the scene extent; the cliff above 2% means a
   coarser voxel silently loses detail. (Not yet wired: `detail_voxel_size_m` stays an absolute option; deriving it from
   the measured extent is the recorded follow-up.)
2. **Curvature 0.1 and planarity 0.95 stay** as the initial thresholds: 0.1 sits on a plateau with 0.05; planarity
   shows no sensitivity here. These are *empirical initial thresholds*, not calibrated constants.
3. **GSD bands apply only to metric worlds.** For a relative-scale world the level mapping should be withheld
   (unsupported budget), not computed from model-unit millimetres. Recorded as a follow-up; today the stage does not
   check `scale_state`.

## Explicitly not established

Universal calibration needs more real datasets (different camera, scale, surface types, indoor scenes). With one
dataset the thresholds are **empirical initial thresholds; additional real datasets required for universal
calibration.** Nothing here validates them beyond South Building.
