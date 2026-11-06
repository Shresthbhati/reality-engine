# Scene-Complete Reconstruction — detect and rebuild everything in a room

**Status:** MISSING as a pipeline (pieces exist, see §1). Plan written 2026-11-06.
**Registry entry:** `docs/implementation/PENDING_IMPLEMENTATION.md` → P2.20.

Goal: from one or a few room photos (including AI-generated ones), produce a World
that contains the room shell **and** every visible object — furniture, appliances,
fixed infrastructure (switches, sockets, vents, lights, radiators, pipes), decor —
each as an entity with class, mask, 3D placement, size, shape and honest confidence.

---

## 1. Why the current output has no furniture (diagnosis, from code)

| # | Finding | Evidence |
|---|---|---|
| D1 | AI-generated "views of the same room" are not geometrically consistent, so SfM cannot register them. Each image falls back to the single-image path, and `fuse_single_views` puts them in **separate frames**. | `engine/pipeline/single_image.py:482-526` |
| D2 | The single-image path runs **no object detection at all**: depth → RANSAC planes → wall/floor/ceiling → UNKNOWN. | `engine/pipeline/single_image.py:1-28`, `156-475` |
| D3 | It uses MiDaS `DPT_Hybrid` (2021, relative inverse depth). A Depth Anything V2 backend exists but is only used by the multi-view slice. | `single_image.py:144-161`; `vertical_slice.py:880` |
| D4 | Plane RANSAC runs over **all** pixels, so furniture pixels pollute wall/floor fits. Wall/floor/ceiling segmentation (OneFormer backend) is not used to pick plane support. | `single_image.py:231+`; `perception/segmentation/oneformer_backend.py` |
| D5 | The only working detector is COCO Mask R-CNN (80 classes: no wardrobe, no switch, no radiator, no rug, no curtain…). It runs only in the multi-view slice. | `vertical_slice.py:955`; `PENDING_IMPLEMENTATION.md` P0.6/P0.7 |
| D6 | Open-vocabulary detection is written but has never reached a World. The lifecycle doc says `transformers` is missing, but on 2026-11-06 the machine's default Python had `transformers` 5.16.1, torch 2.14 + CUDA and `segment_anything` (`sam2` absent). The API's own environment is not verified. Detector choice (OWLv2 vs Grounding DINO) is still an open decision. | `perception/detection/open_vocab.py`; lifecycle doc step 16 |
| D7 | A floor-contact furniture placer exists and is tested, but is explicitly **not wired**. | `perception/instances/furniture_from_detections.py:15` |
| D8 | Infrastructure modules exist (`entities/switchboard.py`, `infrastructure/utility_network.py`, `detail/crop_pass.py` small-object crop pass) but have no detector feeding them in this path. | files listed |

So the user-visible failure is mainly **integration**, not missing research. Wiring the
existing pieces (D2, D3, D4, D7) is the first milestone; new models come second.

## 2. What "accuracy" can mean for AI-generated images

An AI image is not a photo of a real place. It has no true metric size and its
"other views" depict a different room. So, for AI input:

* **Achievable:** the World re-renders to match the image. Every visible object is
  present, in the right place, at plausible size, resting on the right surface.
* **Not achievable:** true metric dimensions, or one consistent room built from several
  AI images that disagree with each other. The engine must say so, not hide it.
  Multi-image AI sets are treated as **independent hypotheses**, unless their
  registration really succeeds.

For real photos and video, the same pipeline feeds the existing multi-view path. Its
multi-view fusion (P0.10 / P2.14) then supplies metric size and consistency.

## 3. Target pipeline (single image; reused per view in multi-view)

```
image
 ├─ S1 camera + depth      FOV/gravity estimate → metric-ish depth
 ├─ S2 scene parsing       panoptic "stuff" (wall/floor/ceiling/window/door/rug…)
 ├─ S3 detect everything   tagger → open-vocab boxes → SAM masks, tiered ontology
 ├─ S4 room shell          planes fitted ONLY on wall/floor/ceiling pixels + vanishing points
 ├─ S5 lift objects        mask ∩ depth → gravity-aligned oriented box, floor/wall/support contact
 ├─ S6 object shape        box proxy → retrieved CAD → generated mesh (each labelled)
 ├─ S7 scene consistency   support, no interpenetration, wall snap, size priors
 └─ S8 verify              re-render vs image: mask IoU, depth error, recall → confidence
```

| Stage | Reuse in repo | New dependency (candidate; licence to verify before adoption) |
|---|---|---|
| S1 | `perception/depth/depth_anything_backend.py`, `metric_depth_backend.py` | Depth Anything V2 **Small** (Apache-2.0; Base/Large are CC-BY-NC), or Depth Pro / Metric3D v2 for metric depth + focal. GeoCalib (Apache-2.0) for FOV + gravity |
| S2 | `perception/segmentation/oneformer_backend.py` (ADE20K, 150 classes) | — |
| S3 | `perception/detection/open_vocab.py`, `perception/ontology.py`, `entities/taxonomy.py`, `detail/crop_pass.py`, `segmentation/sam_backend.py` | `transformers`; Grounding DINO or OWLv2 (both Apache-2.0); RAM++ tagger (Apache-2.0) to propose the vocabulary per image |
| S4 | `perception/geometry/planes.py`, `architecture/room_layout.py`, `engine/pipeline/openings.py` | vanishing-point/line detection (MISSING per lifecycle step 28; OpenCV LSD suffices) |
| S5 | `instances/furniture_from_detections.py`, `instances/object_geometry.py`, `instances/bounding.py`, `instances/lifting.py` | — |
| S6 | `procedural/` (proxies) | tier B: CAD retrieval from a licence-clean library; tier C: image-to-3D (TripoSR MIT / SF3D fits 6 GB; TRELLIS needs more VRAM) — **decision required** |
| S7 | `perception/spatial_reasoning.py`, `entities/attributes.py` | — |
| S8 | `reconstruction/evaluation.py`, `uncertainty/` | a lightweight rasteriser (existing viewer/export path or `pyrender`) |

### Tiered ontology (S3) — "everything you see"

| Tier | Examples | Placement rule |
|---|---|---|
| Architecture | wall, floor, ceiling, door, window, stair, column, beam | S2 + S4 |
| Fixed infrastructure | switch, socket, light fixture, ceiling fan, AC unit, vent, radiator, sprinkler, smoke detector, pipe, conduit, sink, toilet, tap | wall/ceiling-mounted: ray → wall/ceiling plane; small → `crop_pass` second pass at higher res |
| Furniture | sofa, chair, table, bed, wardrobe, shelf, desk, cabinet, counter | floor contact (`furniture_from_detections`) |
| Appliances / electronics | TV, fridge, oven, washer, monitor, lamp | floor or support contact |
| Soft / decor | rug, curtain, cushion, painting, mirror, plant, clock | rug = floor-coplanar; painting/mirror = wall-coplanar; mirror flags specular (roadmap 3.4) |
| Small objects | books, cups, bottles, remotes | on-support; may stay 2D-only + UNKNOWN extent |

Anything detected but not liftable is kept as a 2D-only entity with a reason. Nothing is dropped silently.

## 4. Milestones and acceptance

**M1 — Wire what exists (no new models).**
In `single_image.py`: switch depth to Depth Anything V2, fit planes only on OneFormer
wall/floor/ceiling masks, run Mask R-CNN, lift detections via `furniture_from_detections`
using the solved floor plane, and append the furniture entities.
*Accept:* on a labelled set of ≥ 20 room images (real + AI), COCO-class furniture recall ≥ 70 %;
every furniture entity rests on the floor (contact error < 5 % of room height); the wall-plane
fit residual improves vs the current all-pixel RANSAC; tests are green.

**M2 — Open vocabulary + infrastructure.**
Confirm `transformers` in the API environment, then run the chosen open-vocab detector with the tiered ontology plus the
RAM++ tag proposal, SAM masks, and `crop_pass` for small wall fixtures.
*Accept:* recall ≥ 70 % on the full ontology (≥ 50 % on small infrastructure); precision ≥ 70 %;
wall-mounted items lie on a wall plane.

**M3 — Shape and consistency.**
S6 tier A (oriented boxes / procedural proxies) for every class, plus S7 constraints.
Tier B/C follow the decision in §5.
*Accept:* no interpenetrating objects; re-rendered object-mask IoU vs the image ≥ 0.6 mean.

**M4 — Verification as confidence.**
S8 re-render comparison becomes each entity's confidence input, and the Studio shows it.
*Accept:* every entity has a re-render score; regressions are gated in CI on the eval set.

**Evaluation set (prerequisite for M1).** Add `benchmarks/scene_complete/` with ~20 AI-generated
and ~20 real room photos, hand-labelled (class, box, support surface). Add synthetic rooms with exact
ground truth from `synthetic/`. Metrics: per-tier recall/precision, mask IoU, floor-contact error,
and depth AbsRel where ground truth exists.

## 5. Decisions required (product owner)

| Decision | Options | Recommendation |
|---|---|---|
| Open-vocab detector | OWLv2 · Grounding DINO | Grounding DINO (Tiny): stronger on indoor long-tail classes, fits 6 GB; keep OWLv2 behind the same interface |
| Generated geometry | proxies only · CAD retrieval · image-to-3D generation | proxies now; image-to-3D only as an opt-in layer labelled GENERATED (never "observed") |
| Metric depth model | DA-V2 Small (relative) · Depth Pro · Metric3D v2 | DA-V2 Small now (licence-clean, small); evaluate metric models on the M1 set |
| GPU budget | 6 GiB shared with the API | models load per stage and unload; API idle VRAM fix (roadmap P0-4) is a prerequisite for M2 |

## 6. Failure modes to design for

* AI images with impossible geometry (non-planar walls, warped perspective): S8 must lower confidence, not "fix" the image.
* Detector hallucinations on textured surfaces: they need a mask, a depth-consistent segment and a support surface, or they stay 2D-only.
* Occluded objects: visible extent only; hidden extent marked UNKNOWN.
* Mirrors/windows: reflected objects must not be lifted (mirror mask → exclude).
