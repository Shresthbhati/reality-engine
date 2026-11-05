# Technology Gap Analysis

Ranked list of missing integrations, highest value first. Companion to `docs/audits/TECHNOLOGY_INTEGRATION_AUDIT.md` and `TECHNOLOGY_SELECTION_MATRIX.md` (2026-11-05).

"API verified" says how the entry point was checked:
- **binary** = flags/enums read from the installed `colmap.exe` (`-h`, `grep -a` on the executable).
- **installed source** = signature read from the package in the venv or torch-hub cache.
- **docs-only** = upstream package not installed here; entry point taken from upstream documentation or repo code and **not run**.

Sketches are illustrative, not tested, unless marked "ran".

| # | Gap | Unblocks | API verified |
|---|---|---|---|
| 1 | Working monocular depth (deps + honest probe) | metric depth, object lifting, detection chain | installed source (call sites), docs-only (transformers) |
| 2 | Learned features in SfM (COLMAP ALIKED + LightGlue) | registration on low-texture rooms, GPU extraction | binary |
| 3 | `colmap global_mapper` as a second mapper candidate | faster/more robust large sets | binary |
| 4 | Truthful matchers + real DeDoDe via kornia | trustworthy benchmarks | installed source |
| 5 | SAM ViT-B mask refinement from Mask R-CNN boxes | cleaner masks for 3-D lifting | installed source (torch-hub cache) |
| 6 | OWL-ViT open-vocabulary crop pass wired into the detail pipeline | small-object recall | repo code, docs-only (transformers) |
| 7 | Real measured benchmark harness | trust in every number | torch (installed) |
| 8 | Real kornia / trimesh calls in `engine/technologies` adapters | catalog honesty | ran |
| 9 | Import-time sidecar: one source of truth for "can this model run" | removes silent skips | installed source |

---

## 1. Working monocular depth

**State.** Default `depth_model="DPT_Hybrid"` fails with `ModuleNotFoundError: No module named 'timm'` (reproduced). Depth Anything V2 fails on missing `transformers`. Weights for both are cached. `perception/availability.py:_report_midas` reports "available" because it checks only `torch` and the checkpoint file. The failure becomes `status: skipped` at `vertical_slice.py:883-889`, which also disables Mask R-CNN lifting.

**Upstream entry point (docs-only, not installed):** `transformers.AutoModelForDepthEstimation.from_pretrained(repo)` -> `outputs.predicted_depth`; `transformers.AutoImageProcessor.from_pretrained(repo)`. Repo: `depth-anything/Depth-Anything-V2-Small-hf` (cached). Output is relative depth, which `reconstruction/depth_to_points.metricize_all` already scales against COLMAP.

**Hook.** `perception/depth/depth_anything_backend.py` `_ensure_model` (it calls `from_pretrained(self._checkpoint)` without `local_files_only=True`, so an offline box may still attempt network access); `perception/availability.py` (add an import probe); `pyproject.toml` `perception` extra (add `transformers`, `timm`); `vertical_slice.py:875` (choose Depth Anything when MiDaS is unavailable). Installing packages needs the user's approval.

```python
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
repo = "depth-anything/Depth-Anything-V2-Small-hf"
proc = AutoImageProcessor.from_pretrained(repo, local_files_only=True)
model = AutoModelForDepthEstimation.from_pretrained(repo, local_files_only=True).to("cuda").eval()
inputs = proc(images=pil_image, return_tensors="pt").to("cuda")
with torch.no_grad():
    rel = model(**inputs).predicted_depth                      # (1, H', W') relative depth
rel = torch.nn.functional.interpolate(rel[:, None], size=(h, w), mode="bicubic", align_corners=False)[0, 0]
```

Acceptance: `metricize_all` returns non-empty `alignments` on the South Building or synthetic-room capture; the probe returns "blocked" when `timm`/`transformers` are absent.

## 2. Learned features in SfM: COLMAP's own ALIKED and LightGlue

**State.** The pipeline runs CPU SIFT with affine shape and domain-size pooling (`colmap_backend.py:222-231,404-406`). The installed COLMAP 4.2.0 contains `ALIKED_N16ROT`, `ALIKED_N32`, `SIFT_LIGHTGLUE`, `ALIKED_BRUTEFORCE`, `ALIKED_LIGHTGLUE` (strings found in the executable).

**Entry point (binary).**
`colmap feature_extractor --FeatureExtraction.type ALIKED_N16ROT --AlikedExtraction.max_num_features N --AlikedExtraction.n16rot_model_path PATH --FeatureExtraction.use_gpu 1`
`colmap exhaustive_matcher --FeatureMatching.type ALIKED_LIGHTGLUE --AlikedMatching.lightglue_model_path PATH --FeatureMatching.use_gpu 1`.
The ONNX models default to `github.com/colmap/colmap/releases/download/3.13.0/*.onnx` and download on first use. **Not verified:** that a local filesystem path is accepted in place of the `url;name;sha256` default. Pre-fetching needs approval.

**Hook.** `reconstruction/backend/colmap_backend.py`: `reconstruct()` (~L455-460) and `_reconstruct_in_session.extract_and_match` (~L691-698). Do not combine with `--SiftExtraction.*` flags. Add it as an alternate candidate so `engine/pipeline/candidate_selection.WorldArbiter` (L195) judges it against SIFT as worlds rather than by assumption.

```python
LEARNED = ["--FeatureExtraction.type", "ALIKED_N16ROT",
           "--AlikedExtraction.max_num_features", "4096",
           "--FeatureExtraction.use_gpu", "1"]
run("feature_extractor", ["--database_path", db, "--image_path", imgs, *intr_args, *LEARNED])
run("exhaustive_matcher", ["--database_path", db,
                           "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
                           "--FeatureMatching.use_gpu", "1"])
run("mapper", ["--database_path", db, "--image_path", imgs, "--output_path", sparse])
```

Fallback if downloads are not approved: kornia's ALIKED + LightGlue already run offline here (weights cached; measured 1041 matches / 1040 inliers on a synthetic pair). Feeding them to COLMAP would use `colmap feature_importer --import_path` and `colmap matches_importer --match_type raw` (both flags verified in `-h`); the on-disk text format that `feature_importer` expects was **not** verified.

## 3. `colmap global_mapper`

**State.** Present in the installed binary (`colmap help`), unused (grep = 0).
**Entry point (binary):** `colmap global_mapper --database_path DB --image_path IMGS --output_path OUT` with options such as `--GlobalMapper.min_num_matches 15`, `--GlobalMapper.multiple_models 1`, `--GlobalMapper.skip_rotation_averaging`.
**Hook.** `colmap_backend.py`, immediately after the `mapper` call (~L477), writing to a separate `sparse_global/` directory; submit it to the same candidate arbiter as gap 2.

```python
glob_dir = workspace / "sparse_global"; glob_dir.mkdir()
run("global_mapper", ["--database_path", str(db_path), "--image_path", str(image_dir),
                      "--output_path", str(glob_dir), "--GlobalMapper.min_num_matches", "15"])
candidates.append(("global", self._parse_sparse_dir(glob_dir, evidence_id_by_name, n_input, env)))  # existing parser (colmap_backend.py:582)
```

Value is unproven. It needs a benchmark on real captures (registered cameras, reprojection error) before it becomes a default.

## 4. Truthful matchers and a real DeDoDe

**State.** `XFeatMatcher` and `DeDoDeMatcher` fall back silently to AKAZE / BRISK while reporting `xfeat@accelerated_features` / `dedode@v2`. `available_matchers()` hard-codes `"xfeat": "available"` and `"dedode": "available"` (`reconstruction/matching/__init__.py:49`).

**Entry point (installed source, kornia 0.9.0):** `kornia.feature.DeDoDe.from_pretrained(detector_weights='L-C4-v2', descriptor_weights='G-upright', amp_dtype=torch.float16)`; `forward(images, n=10000, apply_imagenet_normalization=True, pad_if_not_divisible=True) -> (keypoints, scores, descriptors)`; `kornia.feature.match_mnn(desc1, desc2) -> (distances, idxs)`. DeDoDe weights are not cached, so first use downloads from GitHub (needs approval).

**Hook.** `reconstruction/matching/dedode.py` (replace the `except Exception` fallback at L71-73 with `raise MatcherUnavailableError`), `xfeat.py:70`, and `matching/__init__.py:49` (derive availability from an actual import/weights check).

```python
import torch, kornia.feature as kf
dd = kf.DeDoDe.from_pretrained("L-C4-v2", "G-upright").cuda().eval()
with torch.no_grad():
    k0, s0, d0 = dd(img0, n=2048); k1, s1, d1 = dd(img1, n=2048)    # img: (1,3,H,W) in [0,1]
_, idx = kf.match_mnn(d0[0], d1[0])                                  # (M,2) mutual nearest neighbours
p0, p1 = k0[0][idx[:, 0]], k1[0][idx[:, 1]]                          # then cv2.findFundamentalMat as in lightglue.py
```

## 5. SAM ViT-B refinement of Mask R-CNN regions

**State.** `sam_vit_b_01ec64.pth` is cached; the `segment_anything` package is not installed (it is already in the `perception` extra); `sam_backend.py` exists but nothing in the vertical slice calls it.
**Entry point (installed source: `~/.cache/torch/hub/facebookresearch_segment-anything_main/segment_anything`):** `sam_model_registry["vit_b"](checkpoint=...)`, `SamPredictor.set_image(image, image_format="RGB")`, `SamPredictor.predict(point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True)`.
**Hook.** After the detector call at `vertical_slice.py:952-975`, before `lift_region_to_3d`; reuse `SAMSegmentationBackend` (`sam_backend.py:193-209`) for loading.

```python
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_b"](checkpoint=str(ckpt)).to("cuda")
pred = SamPredictor(sam); pred.set_image(rgb, image_format="RGB")
for i, region in enumerate(seg.regions):                             # SegmentedRegion is frozen; mask is List[List[bool]]
    ys, xs = np.nonzero(np.asarray(region.mask))                     # derive the prompt box from the Mask R-CNN mask
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    masks, scores, _ = pred.predict(box=box, multimask_output=False) # (1,H,W) bool
    seg.regions[i] = dataclasses.replace(region, mask=masks[0].tolist())
```

Gate it on `scores[0]` and keep the Mask R-CNN mask as the fallback. Whether SAM improves lifted geometry is unmeasured.

## 6. OWL-ViT open-vocabulary crop pass

**State.** `perception/detail/crop_pass.py` constructs `OpenVocabDetector`, but no non-test module imports `crop_pass`. `OwlVitDetector` is real code; it reports unavailable only because `transformers` is missing. Weights (`google/owlvit-base-patch32`) are cached.
**Entry point (repo, real):** `perception.detection.get_detector("owlvit").detect(image_rgb, queries) -> list[ImageDetection]` (backend uses `OwlViTProcessor` / `OwlViTForObjectDetection` with `local_files_only=True`).
**Hook.** `perception/detail/pipeline.py:run_detail_pipeline` (L94) where candidates are refined; `vertical_slice.py:776-790` already calls it.

```python
from perception.detection import get_detector
det = get_detector("owlvit")                     # NullDetector with a reason if unavailable
hits = det.detect(rgb_crop, ["door", "window", "socket", "radiator"])
obs = [h for h in hits if h.score >= 0.25]       # crop_pass converts these into Observations
```

Needs `transformers` (gap 1 installs it). Recall/precision are unmeasured.

## 7. Real benchmark harness

**State.** `engine/technologies/benchmark.py` reports constants as VRAM and memory (audit section 5).
**Entry point (installed torch):** `torch.cuda.reset_peak_memory_stats()`, `torch.cuda.synchronize()`, `torch.cuda.max_memory_allocated()`, `torch.cuda.Event(enable_timing=True)`.
**Hook.** `benchmark.py:235-243` and the seven `vram_mb = min(host_vram_mb, <const>)` lines; add a `try/except` around `adapter.execute` (L238) so failures count in `failure_rate_pct`; refuse `RECOMMENDED` unless `classification` is a real-upstream class.

```python
torch.cuda.reset_peak_memory_stats(); start, end = torch.cuda.Event(True), torch.cuda.Event(True)
start.record(); result = run_real_model(sample); end.record(); torch.cuda.synchronize()
vram_mb = torch.cuda.max_memory_allocated() / 2**20
latency_ms = start.elapsed_time(end)
```

This only means something once the benchmarked entry actually runs a real model (gaps 1, 4, 5, 6 provide candidates). Until then, benchmark output for simulated entries should say "not measured".

## 8. Real kornia / trimesh calls in the technology adapters

**State.** The `kornia` branch (`adapters.py:91-113`) calls a numpy module and hard-codes `score 0.99`; trimesh is installed but no adapter calls it.
**Entry points (ran):** `kornia.geometry.transform.warp_perspective(src, M, dsize, mode='bilinear', padding_mode='zeros', align_corners=True)`, `kornia.filters.sobel(x)`; `trimesh.Trimesh.is_watertight`, `.is_winding_consistent`, `trimesh.repair.fill_holes`. Verified on a 16x16 patch and a unit box (watertight True, winding consistent True).
**Hook.** `engine/technologies/adapters.py` `VisualPerceptionAdapter` kornia branch and `GeometryCadMeshAdapter` (add a `trimesh` branch).

```python
import torch, trimesh
from kornia.geometry.transform import warp_perspective
from kornia.filters import sobel
warped = warp_perspective(torch.as_tensor(img)[None, None].float(), torch.as_tensor(H)[None].float(), img.shape)
grad = sobel(warped)                                      # (1,1,H,W), measured, not a constant
m = trimesh.Trimesh(vertices=V, faces=F, process=False)
facts = {"is_watertight": m.is_watertight, "is_winding_consistent": m.is_winding_consistent}
```

## 9. One source of truth for "can this model run"

**State.** `perception/availability.py` (MiDaS "available" while broken), `matching.available_matchers()` (XFeat/DeDoDe "available" while mislabelled), and `engine/technologies/registry.py` (`INTERNAL_IMPLEMENTATION` for yolox/oneformer/xfeat/dedode) each answer differently, and all three disagree with reality in at least one case.
**Entry point (stdlib):** `importlib.util.find_spec(name)` plus a cheap import of the model graph module (for MiDaS: `torch.hub.load(<repo>, name, source="local", pretrained=False)` in a try/except, which is how the failure in gap 1 was reproduced), plus a file check on the weight path.
**Hook.** `perception/availability.py` (`_report_midas`, `_report_sam`, `_report_mask_rcnn`) as the shared probe; have `matching/__init__.py` and `engine/technologies/registry.py` call it.

```python
def can_run(import_name, weights=None, builder=None):
    if importlib.util.find_spec(import_name) is None: return False, f"{import_name} not installed"
    if weights is not None and not Path(weights).exists(): return False, f"missing {weights}"
    try: builder() if builder else None
    except Exception as e: return False, f"{type(e).__name__}: {e}"
    return True, "ok"
```

---

## Deliberately not listed
- Open3D, PyMeshLab, GTSAM, Ceres, GDAL, nerfstudio, gsplat, SLAM binaries, OCR, tracking upstreams: no pipeline stage in the audited path consumes their output, so integrating them would add dependencies without closing a measured gap. Revisit when a stage needs them.
- `colmap model_aligner` georegistration: `reconstruction/scale/georeferencing.py` already implements ECEF/ENU and Umeyama Sim(3); a comparison would be needed before replacing it.

## Not verified
- None of the sketches was executed except the kornia/trimesh calls in gap 8 and the LightGlue runs cited in the audit.
- Package installs (`transformers`, `timm`, `segment-anything`) and ONNX/DeDoDe weight downloads are not done and need the user's approval.
- Behaviour of COLMAP when given local ONNX paths, and the `feature_importer` text format, are unchecked.
