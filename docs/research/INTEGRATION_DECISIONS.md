# Integration decisions

| Capability | Candidate | Decision | Why (evidence) |
|---|---|---|---|
| Tiled world delivery | CesiumGS/3d-tiles | REIMPLEMENT (done, box-only) | spec used only as schema oracle; no code copied |
| Pose-graph optimisation | borglab/gtsam | REIMPLEMENT on SciPy now, GTSAM later behind same API | no cp314 wheel (pip: "no matching distribution"); local C++ build not attempted |
| SfM | colmap/colmap | RETAIN | already wrapped; CPU build blocks dense MVS |
| Segmentation | facebookresearch/segment-anything | RETAIN + FIX (done) | works; defects were download default and silent failures |
| Depth | DepthAnything/Depth-Anything-V2 | DEFER | needs same-input benchmark vs MiDaS and per-weight licence review; torch is CPU-only here |
| Point-cloud ops | isl-org/Open3D | DEFER | not installed; check cp314 wheel before planning around it |
| CityJSON validity | cityjson/cjval | NEXT | official validator = oracle for exporters/cityjson (Rust; needs build or binary) |
| IFC | IfcOpenShell | RETAIN | exporters/ifc_bridge.py already optional-imports it; LGPL -> keep isolated/optional |
| SLAM | ORB_SLAM3 | REJECT for linking | GPL-family licence; research reading only |
| Binary analysis | morluto/rea | USE AS TOOL, ON APPROVAL | analyses built artifacts, not source; `setup` modifies agent config |
Licences were identified by file presence only; texts are unreviewed.
