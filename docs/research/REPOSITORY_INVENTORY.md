# Repository inventory (research clones)

Location: `C:\Users\shres\codes-research\` (outside OneDrive and outside every tracked source tree).
Deviation from the brief: the brief asked for a sibling `codes` directory, but `...\Desktop\codes` is already the parent
of reality-engine and ~20 unrelated projects, and sits inside OneDrive (syncing thousands of cloned files). Shallow
clones (`--depth 1`). Manifest: `codes-research/MANIFEST.csv`.

| Repo | Commit | License file | Status |
|---|---|---|---|
| morluto/rea | cc7cc8155a245d537ead2b391255bc213a4a1a0d | LICENSE (README badge: MIT) | cloned; NOT installed or run |
| colmap/colmap | fa089062a56261d11b070aaea0c767cc3730526f | LICENSE | cloned |
| isl-org/Open3D | b6c5e196384ad71e75b6e6f9c5da22d046221f1d | LICENSE | cloned |
| borglab/gtsam | fd4a42eef956168a4d0703c2f80bdcdc4e173308 | LICENSE | cloned |
| nerfstudio-project/nerfstudio | 50e0e3c70c775e89333256213363badbf074f29d | LICENSE | cloned |
| IfcOpenShell/IfcOpenShell | 7a644b1b2150dbe50f2b6874ccc5074795cf0685 | none at root listing | cloned |
| CesiumGS/3d-tiles | b4eedf8fa441beb7360d25d30903a6672cff5f1e | none at root listing | cloned; schema used as test oracle |
| PixarAnimationStudios/OpenUSD | 84d8ed1ae0337ce30dca8d91a3fbe7f04eebc358 | LICENSE.txt | cloned |
| UZ-SLAMLab/ORB_SLAM3 | 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4 | LICENSE (GPL family: do not link) | cloned |
| facebookresearch/segment-anything | dca509fe793f601edb92606367a655c15ac00fdf | LICENSE | cloned |
| DepthAnything/Depth-Anything-V2 | a561b849ebae10a6f5ef49e26c83cbbcd36c71bf | LICENSE | cloned |
| cityjson/specs | 877ee49e16c82a4bc7ab598926efcb9ba1075374 | see repo | cloned (spec; `cityjson.io` named in the brief does not exist) |
| cityjson/cjval | c85df3f8aa985a8adadb7b8a4703c75f703a994a | see repo | cloned (official validator; candidate oracle for exporters/cityjson) |

Not cloned (reason): Meshroom/AliceVision/ODM, VINS-Fusion, OpenVINS, Detectron2, PyTorch3D, CesiumJS, MapLibre, GDAL,
PROJ, Blender, ROS2, Gazebo, Habitat, SUMO. Not yet justified by a specific gap; clone when a capability needs it.
License text of every repo above is UNREVIEWED; only file presence was recorded. No weights or datasets downloaded.
