# Repository Inventory — Reality Engine Technology Investigation

**Parent directory:** `C:\Users\shres\.cline\worktrees\fed0c\`
**Reality Engine:** `C:\Users\shres\.cline\worktrees\fed0c\reality-engine` (branch `cline/fed0c`, commit `f53af6e`, working tree clean)
**Codes (sibling):** `C:\Users\shres\.cline\worktrees\fed0c\codes`
**Environment:** Windows, Python 3.14.7, Node.js v26.8.1 / npm 11.19.0, torch 2.14.0+cpu, torchvision 0.29.0+cpu, numpy 2.5.3, scipy 1.18.1, opencv 5.0.0, pillow 12.3.0, segment-anything installed. COLMAP binary at `C:\Users\shres\tools\colmap-extracted\bin\colmap.exe`. PyPI (files.pythonhosted.org) unreachable — `pip install` fails, so no additional Python packages or model weights can be fetched. GPU/CUDA not available.

## Cloned candidate repositories

| # | Repository | Official URL | Local path | Commit SHA | Branch | License | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | REA (Reverse Engineer Anything) | https://github.com/morluto/rea | `codes/rea` | `b22b0ef22a34ce983d12d015d2895f70263f457a` | main | MIT | **Cloned OK (shallow, `--depth 1`)** | Node.js (v22+/v24+/v26+) MCP + CLI reverse-engineering tool. Deep-analysis providers (Ghidra, IDA, Hopper, Frida, Kuna) are bring-your-own commercial tools — none installed here. Only the JS/asar static-analysis path and source inspection are usable offline. |
| 2 | COLMAP | https://github.com/colmap/colmap | `codes/colmap` | `fa089062a56261d11b070aaea0c767cc3730526f` | main (shallow) | New BSD | **Cloned OK (shallow)** | SfM + MVS + meshing. BSD = permissive, safe to wrap as subprocess. A prebuilt `colmap.exe` is already on this machine and on PATH; Reality Engine's `ColmapReconstructionBackend` shells out to it. |
| 3 | Open3D | https://github.com/isl-org/Open3D | `codes/Open3D` | `b6c5e196384ad71e75b6e6f9c5da22d046221f1d` | main (shallow) | MIT | **Cloned OK (shallow)** | Point-cloud / geometry processing (ICP, colored ICP, RANSAC global registration, voxel downsampling, outlier removal, meshing). **Cannot be used at runtime here:** `pip install open3d` fails (PyPI unreachable) and no prebuilt wheel is present. Inspectable as research only. |

## Not yet cloned (deferred / researched by reputation only)

The following candidates from the mission scope were intentionally NOT cloned, to avoid indiscriminate repository collection. Each is recorded with a reason; they can be cloned later if a concrete capability gap justifies it.

| Repository | Official URL | Reason deferred |
|---|---|---|
| ORB_SLAM3 | https://github.com/UZ-SLAMLab/ORB_SLAM3 | GPLv3 + heavy C++/Pangolin/OpenCV build; SLAM not yet a defined RE core use case; deferred |
| VINS-Fusion | https://github.com/HKUST-Aerial-Robotics/VINS-Fusion | GPLv3 + ROS/catkin build; VIO not yet a defined RE core use case; deferred |
| GTSAM | https://github.com/borglab/gtsam | BSD pose-graph/slam factor graph library; potentially useful for Layer 4 pose-graph optimization; deferred pending a registration-gap analysis |
| OpenVINS | https://github.com/rpng/open_vins | GPLv3 + ROS; deferred |
| segment-anything | https://github.com/facebookresearch/segment-anything | `segment-anything` pip package is already installed in this env but SAM checkpoints are not cached and cannot be downloaded (PyPI blocked); repo clone adds nothing testable right now |
| Depth-Anything-V2 | https://github.com/DepthAnything/Depth-Anything-V2 | Needs PyTorch + checkpoint weights (not cached, cannot download); per-checkpoint licensing uncertain; deferred |
| Detectron2 | https://github.com/facebookresearch/detectron2 | Needs torchvision + COCO weights (not cached); deferred |
| PyTorch3D | https://github.com/facebookresearch/pytorch3d | Needs CUDA build; deferred |
| CesiumJS | https://github.com/CesiumGS/cesium | Apache 2.0; 3D tiles / geospatial viewer; deferred until a geospatial visualization slice is prioritized |
| 3d-tiles | https://github.com/CesiumGS/3d-tiles | Apache 2.0 spec repo; deferred |
| MapLibre GL JS | https://github.com/maplibre/maplibre-gl-js | BSD; deferred |
| GDAL | https://github.com/OSGeo/gdal | C++ geospatial translation library; deferred |
| PROJ | https://github.com/OSGeo/PROJ | MIT coordinate transformations; deferred |
| IfcOpenShell | https://github.com/IfcOpenShell/IfcOpenShell | LGPL/MIT mixed; BIM import/export; deferred |
| OpenUSD | https://github.com/PixarAnimationStudios/OpenUSD | Apache 2.0; deferred |
| Blender | https://github.com/blender/blender | GPLv2; deferred |
| CityJSON / CityGML | https://github.com/cityjson/cityjson.io | MIT-ish; deferred |
| ROS 2 / Gazebo / Habitat-Sim / SUMO | various | Simulation/robotics adapters — evaluated only after core architecture verified; deferred |

## Verification notes

- Every cloned repo URL was verified against the project's stated official homepage before cloning.
- Commit SHAs are pinned (shallow clones record the checked-out commit; `git rev-parse HEAD` was recorded).
- No uncommitted changes exist in any candidate repository (all freshly cloned).
- Reality Engine git history is kept separate from candidate histories (candidates live in `codes/`, a sibling directory, never inside `reality-engine/`).
- No model weights, datasets, or large binaries were downloaded.
- `REA` requires Node.js >= 22.23.4 / 24.22.1 / >=26.12.1; this machine has Node 26.8.1 which does **not** satisfy `>=26.12.1` nor the `^24.22.1`/`^22.23.4` ranges, so a full `npm ci && build` of REA's compiled `dist/` runtime was not attempted; REA is used here as source-level research material, not a runtime tool.
- Blockers are recorded explicitly rather than assumed away (see `KNOWN_LIMITATIONS.md`).
