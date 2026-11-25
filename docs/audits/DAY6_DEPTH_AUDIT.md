# Day 6 preparation: audit of the depth backends

Date: 2026-11-26. This is an audit and a checkpoint/convention probe, not a Day-6
implementation. Nothing below is a depth accuracy claim: the dataset has **no
ground-truth depth**, so no AbsRel / RMSE / δ / plane metric could be computed, and none
is reported.

## 1. Backends present in the tree

| backend | file | checkpoint / method | unit / scale claim | device |
|---|---|---|---|---|
| `DepthAnythingV2Backend` | `perception/depth/depth_anything_backend.py` | `depth-anything/Depth-Anything-V2-Small-hf` (Apache-2.0, ~25 M params) | `unit="relative"`, min–max normalised to [0, 1] per image | CUDA (fp32) / CPU |
| `DepthAnythingV2MetricBackend` | same file | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | `unit="meters"`, raw `predicted_depth`, `scale_state = "metric_estimated"` | CUDA / CPU |
| `HeuristicDepthPriorBackend` | `perception/depth/metric_depth_backend.py` | no model — luminance + vertical gradient scene statistics | `unit="meters"`, `scale_state="estimated"` | CPU |
| engine adapter `depth_anything` | `engine/adapters/depth_anything_v2.py` | Depth Anything V2 Small (same HF id) | `is_metric: False`, "relative inverse depth" | CUDA / CPU |
| `depth_midas_dpt` | `engine/adapters/depth_midas_dpt.py` | MiDaS/DPT | `is_metric: False` | CUDA / CPU |

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

Consequences for the code as written:

1. The relative model's raw output is **inverse depth (disparity-like)**, which is why the
   production adapter's `"is_metric": False` note says "relative inverse depth". The
   `DepthAnythingV2Backend` wrapper then **min–max normalises each image independently to
   [0, 1]**, which destroys both the scale *and* the offset of the underlying quantity:
   two views of the same surface get unrelated numbers, so "compare depth across
   overlapping views" cannot be done with that output. Any multi-view depth comparison
   has to use the raw inverse depth plus a per-image scale/shift fit, not the normalised
   array.
2. The metric checkpoint's output range (4.4–19.5 m, median ~10 m) is plausible for the
   capture and is in metres, but there is no ground truth here: it is an **in-domain
   estimate from a model trained on other indoor data**, so it may be reported as
   `metric_estimated` and must never be promoted to a calibrated metric scale.
3. `HeuristicDepthPriorBackend` returns values clipped to [0.3, 20] m from luminance and a
   vertical gradient, with `unit="meters"`. That is a **fabricated metric scale**: the
   numbers are physical-looking but nothing ties them to this scene. It is documented as a
   heuristic, but `unit="meters"` is what downstream code reads. This is the first defect
   to fix (either `unit="relative"`, or delete the backend), and it is called out here
   rather than silently changed because Day 6 implementation has not begun.

## 4. Defects found (Day-6 work list, none fixed yet)

| # | defect | consequence |
|---|---|---|
| 1 | `HeuristicDepthPriorBackend` labels a scene-statistics prior as `unit="meters"` | fabricated metric scale reaches anything that trusts the unit |
| 2 | `DepthAnythingV2Backend` min–max normalises per image | cross-view comparison and depth fusion impossible; the "sharpness"/confidence story is unrelated to depth accuracy |
| 3 | `DepthMap` has no invalid-pixel representation (no mask, no NaN contract) | sky/windows/specular holes are silently averaged into geometry |
| 4 | Backends `except Exception: continue` without recording why an item was skipped | provenance loses the rejection reason the interface's own docstring requires |
| 5 | No uncertainty model beyond a fixed `confidence=0.92` (relative) / `0.65` (metric) | confidence numbers are not calibrated to anything measured |

## 5. What is missing for a real depth benchmark

There is no ground-truth depth in this repository's datasets (`RELEASE.md`
`total_verified: 0`; the south_building manifest has no depth). A benchmark therefore needs
either (a) a dataset with sensor or rendered depth, or (b) a proxy built from the verified
pair geometry: the un-projected SfM points can score relative depth *up to a per-image
scale and shift* (scale-invariant and shift-invariant errors, e.g. SILog / δ over
log-depth), which is honest and does not require metric truth. Neither exists yet.

## 6. Status

Day 6 implementation: **not started**. Checkpoints verified, conventions measured, five
defects listed, no accuracy claim made. `docs/status.yaml` S5-26 records this.
