# Detail-threshold calibration (P7-05) -- what was measured, what was decided, what is still unknown

Data: the ONE real photographic dataset in the repo (`datasets/south_building`, 32 photographs, 49,608 sparse points,
COLMAP's arbitrary scale). Script: `scripts/measure_detail_thresholds.py`; raw outputs in `docs/engineering/`.
There is no ground truth for "which details exist", so false positives and missed details are NOT scored.

## Correction to the 2026-09-30 measurement

The first run expressed voxel size as a fraction of the **bounding-box diagonal (67.9 units)**. That box is dominated by
a few far-flung tracks: the 1st-99th percentile diagonal of the same points is **6.3 units** (10x smaller). Its
"1-2% of extent" band was really 11-22% of the real footprint, and its "cliff between 2% and 4%" must not be read as a
property of the scene. `detail_threshold_measurements.json` is kept as the historical record
(`--extent bbox`); the product now uses the robust extent (`perception/detail/voxel.py`).

## Measured (2026-10-02, robust extent 6.320 units, others at default; `detail_threshold_measurements_robust_extent.json`)

| factor | values | ROIs | refined / refused | median plane residual (fraction of extent) |
|---|---|---|---|---|
| voxel (fraction of robust extent) | 0.05 / 0.10 / 0.16 / 0.25 / 0.40 | 40 / 20 / 10 / 4 / 2 | all refined, 0 refused | 0.0022 / 0.0042 / 0.0140 / 0.0065 / 0.0388 |
| curvature threshold | 0.05 / 0.1 / 0.2 | 11 / 10 / 8 | all refined | 0.018 / 0.014 / 0.008 |
| planarity threshold | 0.90 / 0.95 / 0.98 | 11 / 10 / 6 | all refined | 0.017 / 0.014 / 0.015 |

Reading (nothing here is a ground-truth claim):
* ROI count falls smoothly with voxel size (roughly with its inverse square): **there is no plateau or cliff in robust
  units**; the voxel is a resolution/compute trade-off, not a tuned constant. Residuals of finer voxels are smaller partly
  because their patches are smaller, so a lower residual is not evidence of better detail. Refinement never refused.
* Planarity is NOT insensitive here (0.98 keeps 6 of 10 ROIs); the earlier "plateau" does not survive the unit fix.
* Curvature 0.05 / 0.1 differ by one ROI: mild sensitivity.
* The measured GSD figure (1.58) is model-unit-millimetres. It is meaningless without a metric anchor.

## Decisions (now implemented)

1. **Voxel size.** Explicit size wins. Metric world: 1.0 m (the documented absolute default, unchanged). Relative or
   unknown scale: `RELATIVE_VOXEL_FRACTION` = **0.16 of the robust scene extent** (1st-99th percentile box diagonal).
   0.16 is chosen for CONTINUITY with the previous behaviour on this dataset (1.0 unit = 0.158 of 6.32), not because it
   was found optimal. If the extent cannot be measured (< 50 points) the stage refuses rather than assume a metre.
2. **GSD bands are metric-only.** With a non-metric scale the report withholds `gsd_mm_per_px`, the fine/medium/coarse
   tier (`scale_unavailable`) and the GSD->level bands; the budget has `gsd_basis="unavailable"` and its level comes from
   multi-view coverage alone (L2 if every observed point has >= 2 views, else L1) -- never L4, never a millimetre figure.
   The scale-free sampling measure (`gsd_model_units_per_px`) is still reported. Capture feedback tells the operator one
   measured distance enables metric levels. Consequence: in a relative world refinement runs the `standard` tier (plane
   backend) only; cylinder/sphere escalation needs a metric anchor.
3. **Curvature 0.1 / planarity 0.95** stay as *empirical initial thresholds*.

## What remains provisional

* Every threshold above is calibrated on ONE scene (South Building), one camera, outdoor facade. The 4 indoor photographs
  mentioned elsewhere have unverified provenance and were not used; no other real dataset is in the repo and none was
  downloaded. **Universal calibration needs further real datasets (indoor, other cameras and scales)** -- an external
  data requirement, not an engineering gap.
* The L2/L1 coverage rule for relative worlds, `RELATIVE_VOXEL_FRACTION`, and the frame-preservation tolerances in the
  candidate arbiter are provisional by the same standard.
