# Day 6 preparation: audit of the depth backends

Date: 2026-11-26. This is an audit and a checkpoint/convention probe, not a Day-6
implementation. Nothing below is a depth accuracy claim: the dataset has **no
ground-truth depth**, so no AbsRel / RMSE / δ / plane metric could be computed, and none
is reported.

## 1. Backends present in the tree

| backend | file | checkpoint / method | unit / scale claim | device |
|---|---|---|---|---|
| `DepthAnythingV2Backend` | `perception/depth/depth_anything_backend.py` | `depth-anything/Depth-Anything-V2-Small-hf` (Apache-2.0, ~25 M params) | `unit="relative"`, `quantity="inverse_depth"`, raw model values (no per-image normalisation) | CUDA (fp32) / CPU |
| `DepthAnythingV2MetricBackend` | same file | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | `unit="meters"`, `quantity="depth"`, `scale_state = "metric_estimated"` | CUDA / CPU |
| `HeuristicDepthPriorBackend` | `perception/depth/metric_depth_backend.py` | no model — luminance + vertical gradient scene statistics | `unit="relative"`, `quantity="depth"`, `scale_state="relative"` (arbitrary scale, not metres) | CPU |
| `depth_anything` adapter (stage 2) | `engine/adapters/depth_anything_v2.py` | Depth Anything V2 Small (same HF id) | `is_metric: False`, "relative inverse depth" | CUDA / CPU |
| `depth_midas_dpt` | `engine/adapters/depth_midas_dpt.py` | MiDaS/DPT | `is_metric: False`; `perception/depth/midAS_backend.py` still min–max normalises per image | CUDA / CPU |

`IDepthBackend.estimate_depth` (`perception/depth/interface.py`) requires a backend to
**skip** an item it cannot process rather than return a blank map; the production
`depth_anything` adapter behaved that way on all 32 real photographs
(`docs/audits/DAY4_REAL_GEOMETRY_MEASUREMENTS.md` §6: 32 observations, 3.37 s, 312.5 MB).

## 2. Checkpoint verification (2026-11-26)

`https://huggingface.co/api/models/<id>` returns **200** for all three probed ids:

* `depth-anything/Depth-Anything-V2-Small-hf`
* `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`
* `depth-anything/Depth-Anything-V2-Base-hf`

Both checkpoints referenced by code therefore exist; neither is a phantom name and both
loaded in this environment (transformers 5.x, torch 2.14.0+cu130).

## 3. Measured output conventions (two real south_building photographs, 1024×768)

| checkpoint | image | min | median | max | reading |
|---|---|---|---|---|---|
| Depth-Anything-V2-Small | P1180141.JPG | 0.0000 | 1.7367 | 5.9559 | **relative inverse depth** (1/depth up to ~6.4), not metres |
| Depth-Anything-V2-Small | P1180308.JPG | 0.0000 | 2.1535 | 6.4433 | same |
| Metric-Indoor-Small | P1180141.JPG | 4.4912 | 10.2740 | 19.5457 | metres |
| Metric-Indoor-Small | P1180308.JPG | 4.3670 | 9.9803 | 18.7538 | metres |

Masking, measured on the same two photographs after the fix: **217,119 / 786,432 (27.6%) and
234,098 / 786,432 (29.8%) pixels are holes**. The raw model output is exactly 0.0 on ~28%
of pixels, concentrated toward the image edges (70% of a left-edge column, 15% mid-image),
and as inverse depth 0 means infinite distance, so those are not measurements. Before the
fix they were carried silently as the map minimum (0 after normalisation); they are now NaN
with a False mask entry, and `metricize_relative_depth`/`depth_map_to_points` keep them out
of the geometry instead of inventing a surface there.

Consequences for the code as written:

1. The relative model's raw output is **inverse depth (disparity-like)**, which is why the
   production adapter's `"is_metric": False` note says "relative inverse depth". The
   wrapper used to **min–max normalise each image independently to [0, 1]**, which destroys
   both the scale *and* the offset of the underlying quantity: two views of the same
   surface got unrelated numbers, so "compare depth across overlapping views" was
   impossible with that output. Fixed 2026-11-26 (§4): the raw values are passed through,
   `quantity="inverse_depth"` states which way they run, and `metricize_relative_depth`
   fits its scale on them directly.
2. The metric checkpoint's output range (4.4–19.5 m, median ~10 m) is plausible for the
   capture and is in metres, but there is no ground truth here: it is an **in-domain
   estimate from a model trained on other indoor data**, so it may be reported as
   `metric_estimated` and must never be promoted to a calibrated metric scale.
3. `HeuristicDepthPriorBackend` returned values clipped to [0.3, 20] from luminance and a
   vertical gradient with `unit="meters"` — a **fabricated metric scale**: physical-looking
   numbers with nothing tying them to the scene. Fixed 2026-11-26: `unit="relative"`,
   `quantity="depth"`, `scale_state="relative"`, and `metricize_relative_depth` refuses it
   (it needs inverse depth, and inverting a depth-like prior would mirror the room).

## 4. Defects found, and what was fixed on 2026-11-26

| # | defect | consequence | state |
|---|---|---|---|
| 1 | `HeuristicDepthPriorBackend` labelled a scene-statistics prior `unit="meters"` | fabricated metric scale reached anything that trusted the unit | **fixed**: `unit="relative"`, `quantity="depth"`, `scale_state="relative"`; `metricize_relative_depth` raises on a depth-like relative map instead of inverting it |
| 2 | `DepthAnythingV2Backend` min–max normalised per image | cross-view comparison and depth fusion impossible | **fixed**: raw model values, `quantity="inverse_depth"`; a regression test pins that 2/4/8 stays 2/4/8 |
| 3 | `DepthMap` had no invalid-pixel representation | holes were silently averaged into geometry | **fixed**: `valid_mask` + NaN in `values` + `invalid_pixels` count, validated for shape and frozen like `values`; both Depth Anything backends and `metricize_relative_depth` populate it |
| 4 | Backends `except Exception: continue` without recording why | provenance lost the rejection reason the interface requires | **fixed**: `backend.skipped` (evidence id + reason, reset per call, logged) on both Depth Anything backends and the heuristic backend |
| 5 | No uncertainty model beyond fixed `confidence=0.92` (relative) / `0.65` (metric) / `0.50` (heuristic) | confidence numbers are not calibrated to anything measured | **open** |
| 6 | `perception/depth/midAS_backend.py` still min–max normalises per image (it is inverse depth, so `quantity="inverse_depth"` is now set, but the normalisation remains) | same cross-view incomparability as defect 2, on the MiDaS path that `engine/pipeline/single_image.py` uses | **open**, deliberately not touched here |

## 5. What is missing for a real depth benchmark

There is no ground-truth depth in this repository's datasets (`RELEASE.md`
`total_verified: 0`; the south_building manifest has no depth). A benchmark therefore needs
either (a) a dataset with sensor or rendered depth, or (b) a proxy built from the verified
pair geometry: the un-projected SfM points can score relative depth *up to a per-image
scale and shift* (scale-invariant and shift-invariant errors, e.g. SILog / δ over
log-depth), which is honest and does not require metric truth. Neither exists yet.

## 6. Status

Defects 1–4 fixed and covered by tests (`tests/test_depth_anything_backend.py`,
`tests/test_perception_interfaces.py`, `tests/test_depth_to_points.py`,
`tests/test_metric_depth_task4_4.py`); defects 5–6 remain open and are listed above. No
depth accuracy claim is made, and there is still no depth benchmark (no ground-truth depth
in the repository). `docs/status.yaml` S5-26 records this.
