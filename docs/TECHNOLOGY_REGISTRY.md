# Technology Registry

Research only — no code in this pass, per the directive's own research-loop
rule (§30/§37) and `docs/REALITY_ENGINE_AUDIT.md` item 3: licensing/technology
evaluation happens before any dependency is installed, never after. This
mirrors `docs/RECONSTRUCTION_BACKEND_DECISION.md`'s format and rigor bar.

**What this document is:** an evaluation of candidate depth-estimation and
segmentation models that could eventually implement the two adapter
interfaces already committed with zero implementations:
`perception/depth/interface.py` (`IDepthBackend`) and
`perception/segmentation/interface.py` (`ISegmentationBackend`).

**What this document is NOT:** a decision. No dependency is installed, no
model is downloaded, no adapter code is written. Everything below is drawn
from general knowledge of these projects as of this session and should be
independently re-verified (repo README, actual `LICENSE` file, actual
checkpoint card) before any real integration begins — exactly the caveat
`RECONSTRUCTION_BACKEND_DECISION.md` didn't need because COLMAP's BSD license
is unambiguous, but these vision-model ecosystems are not.

---

## Depth estimation candidates

### Depth Anything / Depth Anything V2

| Field | Value |
|---|---|
| Name | Depth Anything (and successor Depth Anything V2) |
| Category | CORE COMPATIBLE (code) / CONDITIONAL — see checkpoint row |
| Repository | `LiheYoung/Depth-Anything` (V1); `DepthAnything/Depth-Anything-V2` (V2), both on GitHub, from the University of Hong Kong / TikTok research group |
| License (code) | Apache 2.0 for both versions, to my knowledge — permissive, no viral clause |
| License (model/checkpoint) | This is where it gets conditional, and I am **not fully certain**: Depth Anything V2 ships multiple checkpoint tiers — Small/Base/Large are, to my knowledge, released under Apache 2.0 or CC-BY, but there is also a "Giant" / metric-depth variant trained partly on data with more restrictive terms in some releases. Different checkpoints in the same repo can carry different licenses. **Do not assume all checkpoints share the code's Apache 2.0 license — verify per-checkpoint before use.** |
| Commercial use | Conditional — likely Yes for the standard Small/Base/Large relative-depth checkpoints, but must be confirmed per checkpoint file before any commercial claim is made |
| Platform support | PyTorch, Linux/Windows/macOS in principle; ONNX/TensorRT export exists in the community but is not first-party for every checkpoint |
| GPU/VRAM requirement | I do not have a precise, current number I'm confident in. Order-of-magnitude expectation: the Small (ViT-S) variant should run inference on a few GB of VRAM (potentially CPU-viable but slow), while the Large/Giant variants likely need a modern consumer or datacenter GPU with roughly 8+ GB VRAM for comfortable batch inference. **Treat these as rough guesses, not specs — benchmark before committing to a hardware requirement in any design doc.** |
| Reality Engine adapter status | NOT_STARTED |
| Strengths | Relative (and, in V2, better metric) monocular depth from a single RGB image; strong generalization across in-the-wild photos, which matches Reality Engine's "arbitrary user photo" evidence intake better than models trained mainly on driving/indoor benchmark distributions; active, recent (2024) research lineage with V2 explicitly improving fine detail and transparent/reflective surfaces — a known weak point for older monocular depth models and relevant to photogrammetry-adjacent imagery |
| Weaknesses | Monocular depth is *relative* by default (needs a metric-depth head or external scale reference to align with COLMAP's metric point cloud); not a drop-in geometric fusion with COLMAP's sparse SfM output — reconciling a dense-but-scaleless depth map against COLMAP's sparse-but-metric points is itself unbuilt adapter logic, not just a model swap; checkpoint licensing needs a per-file check, not a per-repo assumption |
| Confidence in this assessment | **Medium.** I'm reasonably confident on the code license (Apache 2.0) and general architecture/lineage. I'm **low-to-medium confidence** on exact checkpoint licensing terms per variant and **low confidence** on the VRAM figures — these should be verified against the live repo and checkpoint cards before any integration decision, not taken from this document as fact. |

### MiDaS

| Field | Value |
|---|---|
| Name | MiDaS (Mixing Datasets for Zero-shot Cross-dataset Transfer) |
| Category | CORE COMPATIBLE |
| Repository | `isl-org/MiDaS` (Intel ISL), GitHub |
| License (code) | MIT license, to my knowledge — permissive |
| License (model/checkpoint) | To my knowledge the released checkpoints are distributed under the same MIT terms as the repo, unlike Depth Anything's more fragmented checkpoint licensing — but this should still be verified against the current repo state, since license terms and checkpoint sets have changed across MiDaS versions (v2 vs v3.1) |
| Commercial use | Likely Yes, given MIT licensing — verify against the current LICENSE file before relying on this |
| Platform support | PyTorch native; ONNX, TensorFlow, and mobile (TFLite) export paths exist in the official repo, which is broader deployment-format coverage than Depth Anything typically offers out of the box |
| GPU/VRAM requirement | Older and generally smaller models than Depth Anything V2's larger tiers; the smaller MiDaS variants (e.g. MiDaS small) are commonly cited in the community as usable on modest GPUs or even CPU for single-image inference, but I do not have a verified current VRAM number and this should be benchmarked rather than assumed |
| Reality Engine adapter status | NOT_STARTED |
| Strengths | Long track record (multiple published papers, several successive versions), broad export/deployment tooling (ONNX/TFLite) which could matter if Reality Engine ever needs a non-Python or edge deployment path, permissive and comparatively simpler licensing story than Depth Anything's multi-checkpoint situation |
| Weaknesses | Generally superseded in accuracy benchmarks by newer models (Depth Anything V2, ZoeDepth) for zero-shot relative depth quality, to my knowledge — MiDaS is the safer/older choice, not the state-of-the-art one; still relative (not metric) depth by default, same fusion-with-COLMAP problem as Depth Anything |
| Confidence in this assessment | **Medium.** Licensing (MIT, code and checkpoints) is something I'm fairly confident about relative to the other candidates here, precisely because MiDaS has historically had a simpler single-license story — but "fairly confident" is not "verified," and the exact current state of the repo's LICENSE file should still be checked. |

### ZoeDepth

| Field | Value |
|---|---|
| Name | ZoeDepth |
| Category | CORE COMPATIBLE / CONDITIONAL — see below |
| Repository | `isl-org/ZoeDepth` (Intel ISL), GitHub |
| License (code) | MIT license, to my knowledge |
| License (model/checkpoint) | I am **not confident** here. ZoeDepth combines a MiDaS-style relative-depth backbone with metric-depth fine-tuning heads trained on datasets (NYU Depth v2, KITTI) that themselves carry their own research-use-oriented licenses/terms in some configurations. Whether the *released checkpoint weights* inherit any restriction from those training datasets, or are cleanly MIT like the code, is something I cannot state with confidence from memory. **This must be checked against the actual model card before any commercial use claim.** |
| Commercial use | Conditional / Unknown — flagged explicitly for a real license check, not assumed |
| Platform support | PyTorch; built directly on top of a MiDaS-family backbone, so tooling/export maturity is likely similar to MiDaS's but I have not confirmed ZoeDepth ships the same ONNX/TFLite export paths |
| GPU/VRAM requirement | No confident figure. ZoeDepth's backbone is comparable in scale to a mid-size MiDaS/Depth-Anything model, so expect single-digit-GB VRAM for inference as a rough order of magnitude, but this is an inference from architecture similarity, not a benchmarked number |
| Reality Engine adapter status | NOT_STARTED |
| Strengths | Purpose-built for **metric** (absolute-scale) monocular depth rather than only relative depth — this is the one candidate here whose core value proposition (metric depth from a single image) could plausibly align directly with COLMAP's metric point cloud without an extra scale-recovery step, which is the main integration friction the other two candidates have |
| Weaknesses | Smaller/less active community and ecosystem than Depth Anything V2 as of my knowledge; metric-depth generalization across arbitrary in-the-wild photos (as opposed to the NYU/KITTI-style domains it was tuned on) is a known harder problem than relative depth, so "metric" does not necessarily mean "accurate metric" on Reality Engine's kind of unconstrained photo input; the licensing uncertainty above is itself a real weakness for adoption |
| Confidence in this assessment | **Low-to-Medium**, specifically lower than MiDaS and Depth Anything on the licensing question. This is exactly the kind of candidate where "the user should verify before any real integration decision" is not boilerplate — the checkpoint license genuinely needs a real look before ZoeDepth is used for anything beyond a research spike. |

---

## Segmentation candidates

### SAM (Segment Anything, original)

| Field | Value |
|---|---|
| Name | SAM (Segment Anything Model) |
| Category | CORE COMPATIBLE |
| Repository | `facebookresearch/segment-anything`, GitHub (Meta AI) |
| License (code) | Apache 2.0, to my knowledge |
| License (model/checkpoint) | To my knowledge the released checkpoints (ViT-B/L/H) are also released under Apache 2.0, matching the code — SAM1's licensing story is comparatively clean and I recall this with more confidence than the depth-model checkpoint situations above, but it should still be confirmed against the current model card rather than taken purely from memory |
| Commercial use | Likely Yes under Apache 2.0 — verify current LICENSE/model card before relying on this for a real integration |
| Platform support | PyTorch; ONNX export path is officially supported for the mask decoder, enabling lighter-weight/web deployment of at least part of the pipeline |
| GPU/VRAM requirement | I do not have a confident current figure. Rough expectation from the architecture (ViT-H is the largest/default-quality checkpoint): likely needs a GPU with several GB of VRAM (possibly 8+ GB) for the ViT-H encoder at reasonable resolution/batch size; the smaller ViT-B checkpoint should be meaningfully lighter. **Benchmark, don't assume, before sizing hardware requirements around this number.** |
| Reality Engine adapter status | NOT_STARTED |
| Strengths | Promptable segmentation (point/box/mask prompts) rather than fixed-class semantic segmentation — this maps naturally onto an interactive Reality Studio workflow where a user or the Outliner selects a region of an image and asks "segment this into an entity," which fits the evidence-grounded/human-in-the-loop pattern the rest of the codebase (Session/Dataset, provenance) already uses; large, well-established ecosystem and community tooling |
| Weaknesses | Image-only, no native video/multi-frame temporal consistency (that's what SAM 2 adds — see below); class-agnostic masks only, no semantic labels out of the box, so downstream entity-type inference (mapping a mask to Reality Engine's `EntityType` enum) is still unbuilt logic on top of SAM regardless of which SAM version is chosen; heavier ViT-H encoder can be a real latency/hardware cost for anything approaching interactive use |
| Confidence in this assessment | **Medium-High** on license/commercial-use (Apache 2.0 is a comparatively well-known and simple case for SAM1), **Low** on the specific VRAM figure — flagged explicitly. |

### SAM 2

| Field | Value |
|---|---|
| Name | SAM 2 (Segment Anything Model 2) |
| Category | CORE COMPATIBLE |
| Repository | `facebookresearch/sam2`, GitHub (Meta AI) |
| License (code) | Apache 2.0, to my knowledge, same lineage as SAM1 |
| License (model/checkpoint) | To my knowledge SAM 2's checkpoints are also released under a permissive license consistent with Apache 2.0 (Meta moved away from the more restrictive non-commercial-style licenses used for some other Meta research releases, e.g. Llama's community license, and SAM/SAM2 have generally been the more permissively-licensed branch of Meta's vision research) — but "to my knowledge" is doing real work in that sentence and this is exactly the kind of claim that needs a direct check against the current repo's `LICENSE` file, not this document, before being relied on |
| Commercial use | Likely Yes — same caveat as above |
| Platform support | PyTorch; Meta has published mobile/edge-oriented variants and demos, suggesting broader deployment-target ambition than SAM1, but I do not have confident specifics on which export formats (ONNX/CoreML/etc.) are officially first-party vs. community-maintained |
| GPU/VRAM requirement | No confident figure, same caveat as SAM1: expect it to be in a similar or somewhat higher range than SAM1 given the added temporal/memory-attention machinery for video, but this is architectural inference, not a benchmark |
| Reality Engine adapter status | NOT_STARTED |
| **What's different from SAM 1** | SAM 2 extends promptable segmentation from single images to **video**, adding a streaming memory mechanism so a mask prompted on one frame propagates and stays consistent across subsequent frames without re-prompting every frame. For Reality Engine specifically, this matters if photo/video evidence ever includes video walkthroughs of a scene (not just still photos) — SAM 2's per-object memory could track a segmented entity (e.g. a piece of debris, a vehicle) across a video capture the way COLMAP tracks feature points across still frames for SfM. For a still-photo-only evidence pipeline, SAM 2's video capability is unused overhead relative to SAM1 |
| Strengths | Everything SAM1 has, plus native video segmentation/tracking without extra tooling; reportedly improved image-segmentation quality and speed over SAM1 as a side effect of the architecture update, to my knowledge, though I don't have a confident quantified benchmark number to cite here |
| Weaknesses | Newer and so has had less time to accumulate the breadth of community tooling/tutorials/forks that SAM1 has; the video/memory machinery is added complexity that's wasted if Reality Engine's evidence intake stays photo-only (per `RECONSTRUCTION_BACKEND_DECISION.md`, current evidence intake is COLMAP-style multi-photo, not video); same class-agnostic-mask-to-EntityType gap as SAM1 |
| Confidence in this assessment | **Medium** on licensing (I believe it follows SAM1's permissive pattern but have not verified the exact current license text), **Low** on the specific hardware/VRAM numbers and on any precise quality-improvement percentage over SAM1 — none of those specific numbers should be treated as established fact from this document. |

---

## Recommendation

Modeled on how `RECONSTRUCTION_BACKEND_DECISION.md` reached the COLMAP call:
narrow the field on license clarity and integration-cost grounds first, then
name one candidate per category as **worth evaluating further with a real
license check and benchmark** — not a final decision, since nothing here has
been installed, run, or measured.

**Depth: ZoeDepth is the most interesting candidate to evaluate next, but MiDaS
is the safer fallback if ZoeDepth's checkpoint licensing doesn't clear.**
ZoeDepth's metric-depth output is the only one of the three that could
plausibly align with COLMAP's metric point-cloud scale without an extra
scale-recovery step, which is the single biggest integration cost every
monocular-depth candidate here shares. But its checkpoint licensing is the
least certain of the three (flagged above as Low-to-Medium confidence), so
the realistic next step is: (1) pull the current ZoeDepth model card and
confirm checkpoint licensing in writing, and if that fails commercial-use
requirements, (2) fall back to MiDaS, whose MIT code-and-checkpoint story is
the cleanest of the three even though it's the oldest/lowest-benchmark
option. Depth Anything V2 is likely the best raw quality but its
per-checkpoint licensing fragmentation makes it the highest-diligence-cost
option of the three — worth revisiting once the license story is resolved
via the repo's own documentation, not this one.

**Segmentation: SAM (original) over SAM 2, for now.** Reality Engine's
current evidence intake (per `RECONSTRUCTION_BACKEND_DECISION.md` and the
audit) is multi-photo, COLMAP-style — not video. SAM 2's headline
improvement over SAM1 is video/temporal tracking, which is unused capability
for a photo-only pipeline today. SAM1 is more mature, has more community
tooling, and (to my knowledge) has the cleaner, longer-established licensing
story of the two. If Reality Engine's evidence model ever grows to ingest
video walkthroughs rather than discrete photos, SAM 2 becomes the obvious
re-evaluation, since its memory mechanism would map onto tracking a segmented
entity across frames the same way COLMAP tracks feature points — but that is
a future trigger condition, not today's recommendation.

**Neither recommendation is a decision.** Per this document's own confidence
flags, both candidates need: (1) a direct read of the live repository's
`LICENSE` file and model card, not this document's memory-based summary; (2)
an actual benchmark run against Reality Engine's real hardware target, since
every VRAM/performance figure above is an order-of-magnitude guess, not a
measurement; and (3) a look at how the chosen model's output type (relative
vs. metric depth; class-agnostic mask) actually reconciles with WorldIR's
existing `Provenance`/`Geometry`/`Entity` schema before any adapter code is
written against `IDepthBackend` or `ISegmentationBackend`.
