# 37 — Open-Source Reuse Map

Companion to `34_MASTER_PLAN.md`. Replaces `20_REUSE_MAP.md` (stale).
Rule: nothing below enters a production run until its licence is recorded in `perception/model_registry.py` and passes the gate from card T009.

**Licence column:** ✓ = confirmed from the project's own page on 2026-11-10 · ◇ = recalled from memory, **confirm before adopting** · weights and code are often licensed separately — check both.
**Verdict:** ADOPT (use as a dependency) · WRAP (call as an external backend behind our interface) · LEARN (read for method, write our own) · EVAL (benchmark/reference only, never shipped) · AVOID.

---

## 1. Already in the repo [V]

Registered in `perception/model_registry.py` or declared in `pyproject.toml`: COLMAP (external binary) · LightGlue + ALIKED · DeDoDe · XFeat · MiDaS · Depth Anything V2 · Mask R-CNN · SAM (ViT-B/H) · SAM 2 (tiny…large) · OWL-ViT / OWLv2 · Grounding DINO tiny · MASt3R ViT-L (**CC-BY-NC — must stay out of production, T009**) · IfcOpenShell · ezdxf · trimesh · networkx · boto3 · OpenCV · kornia · timm · transformers.

Use these first. Every row below is an addition, and needs a reason the existing set does not cover.

---

## 2. Camera poses and multi-view geometry

| Repo | Licence | Verdict | What we take | Card |
|---|---|---|---|---|
| `colmap/colmap` | BSD ✓ | keep | Incremental SfM today. GLOMAP was archived in March 2026 and its global SfM now ships inside COLMAP — use COLMAP's global mapper as a second mapping mode before adding anything else. | P107 |
| `colmap/glomap` | BSD-3 ✓ (archived) | AVOID as a separate dependency | superseded by COLMAP itself | — |
| `facebookresearch/map-anything` | Apache-2.0 code ✓; `facebook/map-anything-apache` checkpoint Apache-2.0 ✓ (default checkpoint is CC-BY-NC) | **WRAP — first choice for D6** | Metric multi-view reconstruction that accepts optional intrinsics, poses and depth as inputs — fits photos-only today and ARKit priors later. | G301–G304 |
| `ByteDance-Seed/Depth-Anything-3` | Apache-2.0 repo ✓ (confirm per checkpoint) | WRAP — alternative | Multi-view depth + pose from unposed images; smaller variants for the 6 GB card. | G301 |
| `facebookresearch/vggt` | default weights non-commercial; a commercial checkpoint is reported ◇ | EVAL unless the commercial terms are confirmed | reference point for G301 | G301 |
| `naver/dust3r`, `naver/mast3r`, MASt3R-SLAM | CC-BY-NC-SA ◇ | EVAL | accuracy reference only | — |
| `cvg/Hierarchical-Localization` (hloc) | Apache-2.0 ◇ (SuperPoint/SuperGlue weights are non-commercial) | LEARN | retrieval + matching pipeline shape for cross-session registration; use our LightGlue + ALIKED | W4xx |
| `cvg/pixel-perfect-sfm` | Apache-2.0 ◇ | LEARN | featuremetric refinement idea if dimensional error stalls | P7 |

## 3. Monocular depth, normals, planes

| Repo | Licence | Verdict | What we take | Card |
|---|---|---|---|---|
| `microsoft/MoGe` (MoGe-2) | MIT code ✓ (bundled DINOv2 Apache-2.0) | WRAP | metric point map + normals from one image; blank-wall plane priors; single-photo path | G309 |
| `YvanYin/Metric3D` (v2) | BSD-2 ✓ | WRAP — alternative | metric depth + surface normals | G309 |
| `DepthAnything/Depth-Anything-V2` | small Apache-2.0, larger CC-BY-NC ◇ | keep small only | already registered; confirm which size the registry points to | T009 |
| `lpiccinelli-eth/UniDepth` | CC-BY-NC ◇ | EVAL | — | — |

## 4. Structured rooms: walls, openings, floor plans

| Repo | Licence | Verdict | What we take | Card |
|---|---|---|---|---|
| `manycore-research/SpatialLM` | weights CC-BY-NC ✓ (1.0 via SceneScript encoder, 1.1 via Sonata); Pointcept-based code Apache-2.0 ✓ | **EVAL** | Point cloud → walls / doors / windows / oriented boxes. Run it on the six measured rooms as an outside reference for our own layout stage. Do not ship. Its output format (typed layout script) is worth copying as an interchange format. | E210 side-run |
| `facebookresearch/scenescript` | non-commercial ◇ | LEARN | structured-language scene representation | — |
| `ywyue/RoomFormer` | licence not confirmed ◇ | LEARN | density-map → room polygons; method for non-rectangular rooms | P7 |
| `sunset1995/HorizonNet` | MIT ◇ | LEARN / WRAP | layout from a single panorama — relevant because 360 ingestion already exists | P7 |
| `CubiCasa/CubiCasa5k` | dataset CC-BY-NC ◇ | EVAL | floor-plan symbol vocabulary | — |
| ScanNet++, ARKitScenes, Structured3D | research / non-commercial terms ◇ | EVAL | public benchmarks alongside our six rooms; never training data for a shipped model without legal review | E2xx |

## 5. Detection, segmentation, open vocabulary

| Repo | Licence | Verdict | What we take | Card |
|---|---|---|---|---|
| `facebookresearch/sam2` | Apache-2.0 ◇ (registry says Apache-2.0 [V]) | keep | masks for windows, doors, fixtures; video propagation for identity across frames | G306 |
| `IDEA-Research/GroundingDINO` | Apache-2.0 ◇ (registry [V]) | keep | text-prompted boxes: "window", "glass door", "socket", "light switch" | G306 |
| OWLv2 (`google-research/scenic`, via `transformers`) | Apache-2.0 ◇ (registry [V]) | keep | second detector for agreement voting | G306 |
| `microsoft/Florence-2` (via `transformers`) | MIT ◇ | WRAP if recall is short | dense region captions for fine components | P7 |
| `concept-graphs/concept-graphs`, `hovsg/HOV-SG` | MIT ◇ | LEARN | open-vocabulary 3D scene-graph construction: how to lift 2D masks to persistent 3D objects with relations | W4xx, P7 |
| `Pointcept/Pointcept` | MIT ◇ | LEARN | point-cloud segmentation baselines | P7 |

## 6. Capture and device tracking (D5)

| Repo | Licence | Verdict | What we take | Card |
|---|---|---|---|---|
| ARKit / ARCore (platform SDKs) | platform terms | ADOPT via a thin native wrapper | poses, intrinsics, depth, IMU, tracking state — the phone's own VIO and loop closure | C601–C606 |
| `strayrobots/scanner` (Stray Scanner) | MIT ◇ | ADOPT its export layout as our first import format | RGB + ARKit odometry + LiDAR depth + IMU in plain files; lets C602 ship before any native code | C602 |
| `jc211/NeRFCapture` | MIT ◇ | LEARN | minimal ARKit pose-streaming app | C6xx |
| `ionic-team/capacitor` | MIT ◇ | ADOPT when the wrapper is built | keep the existing web capture UI, add a native plugin for ARKit/ARCore | after C606 |
| `VladyslavUsenko/basalt` | BSD-3 ◇ | fallback only | the one permissively licensed VIO, if device tracking is unavailable | — |
| `UZ-SLAMLab/ORB_SLAM3`, `rpng/open_vins`, `HKUST-Aerial-Robotics/VINS-Fusion` | GPL-3 ◇ | **AVOID** | remove from `trajectories/backend/selection.py` after D5 is approved | C604 |
| `tus/tusd` + `tus/tus-js-client` | MIT ◇ | consider | standard resumable uploads, only if the existing chunked upload (`apps/api/storage.py: stage_chunk`) fails the S709 tests | S709 |

## 7. Surfaces, meshes, geometry checks

| Repo | Licence | Verdict | What we take | Card |
|---|---|---|---|---|
| `mikedh/trimesh` | MIT ◇ | keep | watertight / manifold / components checks | P110 |
| `elalish/manifold` (`manifold3d`) | Apache-2.0 ◇ | ADOPT when booleans are needed | robust opening subtraction from wall solids | P7 |
| `isl-org/Open3D` | MIT ◇ | ADOPT optional extra | TSDF fusion, plane segmentation, ICP for cross-session registration | W4xx |
| `PDAL/PDAL`, `laspy` | BSD ◇ | ADOPT optional | LAS/LAZ/E57 import for scanner customers | P7 |
| `gumyr/build123d` / `CadQuery` (OpenCascade) | Apache-2.0 / LGPL kernel ◇ | LEARN now | B-rep for parametric walls and openings | P7 |
| `cdcseacave/openMVS` | AGPL-3 ◇ | AVOID | — | — |
| `cnr-isti-vclab/PyMeshLab` | GPL-3 ◇ | AVOID in product | offline tooling only | — |

## 8. Gaussian splats and large-world rendering (P7)

| Repo | Licence | Verdict | What we take |
|---|---|---|---|
| `nerfstudio-project/gsplat` | Apache-2.0 ✓ | WRAP | photo-based 3DGS training from our COLMAP/feed-forward poses — the missing half of the splat feature |
| `graphdeco-inria/gaussian-splatting`, 2DGS, SuGaR, PGSR | non-commercial ◇ | EVAL | quality reference |
| `sparkjsdev/spark` | licence not confirmed ◇ | candidate | three.js splat renderer that mixes with meshes; compare with the current overlay |
| `mkkellogg/GaussianSplats3D` | MIT ◇ | candidate | simpler three.js splat viewer |
| `NASA-AMMOS/3DTilesRendererJS` | Apache-2.0 ✓ | ADOPT | camera-driven tile streaming in three.js; pair with `worldstore/tiles.py` |
| `gkjohnson/three-mesh-bvh` | MIT ◇ | ADOPT | fast picking and measurement raycasts on large meshes |
| `CesiumGS/3d-tiles-tools` | Apache-2.0 ◇ | ADOPT (tooling) | validate the tiles we write |

## 9. BIM, CAD and exchange

| Repo | Licence | Verdict | What we take | Card |
|---|---|---|---|---|
| `IfcOpenShell/IfcOpenShell` | LGPL-3 ◇ | keep | IFC write + its validation tools; Bonsai (BlenderBIM) as the free external check | E211 |
| `mozman/ezdxf` | MIT ◇ | keep | DXF write and audit (`ezdxf.audit`) as an automatic pre-check | E212 |
| `ThatOpen/engine_web-ifc`, `ThatOpen/engine_components` | MPL-2.0 ◇ | consider | in-browser IFC round-trip check of our own exports | P7 |
| `buildingSMART/IDS` + `ifctester` | ◇ | ADOPT (tooling) | machine-checkable export requirements per customer | E211 |
| `xeokit/xeokit-sdk` | AGPL-3 ◇ | AVOID | — | — |

## 10. Jobs, storage, observability

| Repo | Licence | Verdict | What we take |
|---|---|---|---|
| (existing) `apps/api/jobs.py` queue | ours | **keep (D8)** | add stage table, resource classes, tenant caps |
| `procrastinate-org/procrastinate` | MIT ✓ | fallback | Postgres-native queue with locks and retries, if S705 shows our claim logic is unsound |
| `hatchet-dev/hatchet` | MIT ✓ | fallback | durable DAG execution on Postgres, if stage orchestration outgrows the runner |
| `temporalio/temporal` | MIT ◇ | not now | only for multi-day, multi-service workflows |
| `open-telemetry/opentelemetry-python` | Apache-2.0 ◇ | ADOPT in P6 | trace id from request → job → stage |
| `rerun-io/rerun` | MIT / Apache-2.0 ◇ | ADOPT (dev tooling) | inspect cameras, points and stage outputs when a wall goes missing |
| `postgis/postgis` | GPL-2 (server) ◇ | ADOPT in production | site/geo queries; a separate service, not linked code |
| `minio/minio` | AGPL-3 ◇ | dev only | local S3; production uses a managed S3-compatible store |

## 11. Frontend

| Repo | Licence | Verdict | What we take | Card |
|---|---|---|---|---|
| `pmndrs/zustand` | MIT ◇ | keep (installed) | the Studio store | U506 |
| `radix-ui/primitives` | MIT ◇ | consider, narrowly | accessible Tabs / Dialog / Popover if hand-rolled ones fail `U518` | U507, U512 |
| `dequelabs/axe-core` (`@axe-core/playwright`) | MPL-2.0 ◇ | ADOPT (dev) | accessibility gate | U518 |
| `yomotsu/camera-controls` | MIT ◇ | consider | smooth orbit ↔ walk ↔ top transitions | P7 |
| `pmndrs/react-three-fiber` | MIT ◇ | AVOID for now | the viewport is plain three.js; a rewrite buys nothing |
| TanStack Query | MIT ◇ | AVOID for now | `frontend/src/lib/api/hooks.ts` exists; add de-duplication there before adding a library |

---

## 12. Shortlist — the eight that change the product

1. **MapAnything (Apache checkpoint)** — poses and metric geometry where SfM fails.
2. **ARKit/ARCore tracking via the Stray Scanner layout** — indoor trajectory, scale and depth without building a VIO.
3. **MoGe-2 or Metric3D v2** — normals and planes for blank walls.
4. **Grounding DINO + SAM 2 (already registered)** — closed windows from appearance, not gaps.
5. **SpatialLM as an outside referee** — an independent layout to compare against, never shipped.
6. **gsplat** — real photo-trained splats (P7).
7. **3DTilesRendererJS + three-mesh-bvh** — streaming and picking at building scale (P7).
8. **IfcOpenShell validation + ezdxf audit + IDS** — automatic export checks before a human opens CAD.

## 13. Before adopting anything

- [ ] Read the LICENSE file and the model card; record both in the registry entry.
- [ ] Confirm the weights' training-data terms allow commercial use.
- [ ] Pin a commit or version and a sha256.
- [ ] Measure VRAM and seconds on the 6 GB dev card (G301 pattern).
- [ ] Wrap behind an existing interface; the absence of the dependency must produce an explicit "unavailable", never an empty success.
