# Reality Engine — Target Model Ecosystem Catalogue (125 Slots)

## 1. Executive Summary

This specification establishes the official **125-Slot Target Model Ecosystem Catalogue** for Reality Engine. To reconstruct complete environments (individual rooms, furnished apartments, multi-storey buildings, street infrastructure, and entire cities) with physical accuracy, Reality Engine eliminates single-model dependency.

The catalogue is strictly partitioned into 10 computer vision, photogrammetry, geometry, and perception domains:

| # | Domain | Slots | Primary Function |
|---|---|---|---|
| 1 | **Geometry & 3D Reconstruction** | 15 | Multi-view stereo, Structure-from-Motion, neural surfaces, radiance fields |
| 2 | **Depth & Metric Geometry** | 15 | Monocular/stereo metric depth, relative depth, normal-guided depth |
| 3 | **Detection & Recognition** | 20 | Open-vocabulary 2D/3D bounding boxes, multi-modal LLM perception |
| 4 | **Segmentation** | 15 | Promptable masks, panoptic segmentation, video object segmentation |
| 5 | **Feature Matching & Tracking** | 12 | Dense/sparse keypoints, descriptor matching, optical flow, point tracking |
| 6 | **Point-Cloud 3D Understanding** | 12 | 3D object detection, semantic point segmentation, primitive fitting |
| 7 | **Room & Architectural Understanding** | 12 | Floorplan extraction, Manhattan/Atlanta planes, structural framing, openings |
| 8 | **Material, Surface Normal & Lighting** | 8 | PBR material decomposition, surface normal fields, HDR environment maps |
| 9 | **Mesh & Topology** | 8 | Surface meshing, manifold cleanup, remeshing, Poisson reconstruction |
| 10 | **Image Enhancement & Restoration** | 8 | Super-resolution, deblurring, HDR tone-mapping, reflection removal |
| **Total** | | **125** | Complete Universal Reconstruction Catalogue |

---

## 2. Licensing Separation & Commercial Viability Policy

Reality Engine enforces an explicit distinction between code licenses and model weight licenses:

1. **Permissive Commercial Viability**: Models with Apache-2.0, MIT, BSD, or commercial-permissive weights (e.g. COLMAP BSD, Grounding DINO Apache-2.0, Florence-2 MIT, Segment Anything 2 Apache-2.0) are cleared for default production pipelines.
2. **Research / Non-Commercial Segregation**: Models bearing CC-BY-NC-4.0, GPL-3.0 copyleft, or custom research-only licenses (e.g. DUSt3R code GPL-3.0 / weights CC-BY-NC-4.0; MASt3R weights CC-BY-NC-4.0; Depth Anything V2 metric indoor weights CC-BY-NC-4.0) are strictly gated behind the `allow_unestablished_licence=True` parameter flag.
3. **No Silent Upgrades or Downgrades**: If a non-commercial or proprietary model is requested but unlicensed or missing, Reality Engine records an explicit blocked state rather than substituting another model silently.

---

## 3. Detailed 125-Slot Domain Directory

### Domain 1: Geometry & 3D Reconstruction (15 Slots)
1. `colmap_sfm` — Sparse SfM, bundle adjustment, camera pose estimation (BSD-3-Clause).
2. `colmap_mvs` — Dense Multi-View Stereo, patch-match depth, dense point cloud fusion (BSD-3-Clause).
3. `glomap` — Global Structure-from-Motion, scalable rotation averaging (BSD-3-Clause).
4. `openmvg` — Open Multiple View Geometry, rigorous geometric calibration (MPL-2.0).
5. `openmvs` — Dense surface reconstruction, mesh texturing (AGPL-3.0 / Commercial).
6. `dust3r` — Dense Unconstrained Stereo 3D Reconstruction (Code GPL-3.0 / Weights CC-BY-NC-4.0).
7. `mast3r` — Matching and Stereo 3D Reconstruction with 3D point tracking (Code GPL-3.0 / Weights CC-BY-NC-4.0).
8. `vggt` — Visual Geometry Grounded Transformer, feedforward 3D scenes (MIT).
9. `vggt_omega` — Extended long-sequence VGGT for building-scale capture (MIT).
10. `mapanything` — Open-world spatial mapper and multi-view 3D integrator (Apache-2.0).
11. `gaussian_splatting` — 3D Gaussian Splatting for novel-view radiance synthesis (Inria Research License).
12. `nerfacto` — Real-time neural radiance fields for appearance synthesis (Apache-2.0).
13. `2d_gaussian_splatting` — Ray-aligned 2D surface surfels for sharp planar boundaries (Apache-2.0).
14. `neus2` — Fast neural surface reconstruction via signed distance functions (GPL-3.0).
15. `instant_ngp` — Instant Neural Graphics Primitives with multi-resolution hash encodings (NVidia Source Code License).

### Domain 2: Depth & Metric Geometry (15 Slots)
16. `depth_anything_v2_vits` — Lightweight relative monocular depth estimation (Apache-2.0).
17. `depth_anything_v2_vitb` — Balanced monocular depth (Apache-2.0).
18. `depth_anything_v2_vitl` — High-fidelity monocular depth for high-res images (Apache-2.0).
19. `depth_anything_v2_metric_indoor` — Metric absolute depth for interior rooms (CC-BY-NC-4.0).
20. `depth_anything_v2_metric_outdoor` — Metric absolute depth for urban scenes (CC-BY-NC-4.0).
21. `depth_anything_v3_metric` — Unified relative and zero-shot metric depth (Apache-2.0).
22. `zoedepth_nk` — Metric depth combining relative encoder with metric bins (MIT).
23. `zoedepth_k` — Outdoor metric depth calibrated for automotive/street scenes (MIT).
24. `marigold_v1` — Diffusion-based monocular depth for fine surface micro-details (Apache-2.0).
25. `marigold_normals` — Diffusion-based surface normal field predictor (Apache-2.0).
26. `metric3d_v2_vitl` — Zero-shot metric depth and surface normals (Apache-2.0).
27. `leres` — Learning recover depth with surface normal and edge constraints (Apache-2.0).
28. `midas_v31` — Robust multi-dataset monocular relative depth (MIT).
29. `unidepth_v1` — Metric depth and camera intrinsics estimation from a single image (Apache-2.0).
30. `geowizard` — Multi-task geometry foundation model for depth and normals (Apache-2.0).

### Domain 3: Detection & Recognition (20 Slots)
31. `grounding_dino_swint` — Open-vocabulary zero-shot 2D object detection (Apache-2.0).
32. `grounding_dino_swinb` — Medium open-vocabulary detector (Apache-2.0).
33. `grounding_dino_swinl` — High-accuracy open-vocabulary object detector (Apache-2.0).
34. `florence2_base` — Multi-task visual vision-language model for dense captions and region tags (MIT).
35. `florence2_large` — High-capacity dense captioning and open-vocabulary bounding boxes (MIT).
36. `yolo_world_v2_x` — Real-time zero-shot open-vocabulary object detector (GPL-3.0).
37. `owl_vit_v2` — Open-world localization transformer for long-tail items (Apache-2.0).
38. `glip_large` — Grounded language-image pretraining object detector (Apache-2.0).
39. `cube_rcnn` — Single-view 3D oriented bounding box detection (Apache-2.0).
40. `omni3d` — Large-scale 3D bounding box predictor across indoor and outdoor datasets (Apache-2.0).
41. `internvl2_8b` — Multi-modal vision-language assistant for complex room reasoning (Apache-2.0).
42. `qwen2_vl_7b` — Vision-language fine-grained object attribute and material identifier (Apache-2.0).
43. `detr_resnet101` — End-to-end transformer object detector for standard indoor categories (Apache-2.0).
44. `co_detr` — Collaborative DE-DETR for architectural fixture detection (Apache-2.0).
45. `dwt_detector` — Door and window opening detector for architectural elevations (MIT).
46. `ocr_paddle_v4` — High-accuracy scene text recognition for appliance labels and room signage (Apache-2.0).
47. `trocr_large` — Transformer OCR for printed and handwritten architectural annotations (Apache-2.0).
48. `mep_fixture_detector` — Specialist electrical sockets, switches, and HVAC vent detector (Apache-2.0).
49. `urban_infrastructure_det` — Street lamps, traffic lights, fire hydrants, and bollard detector (Apache-2.0).
50. `facade_element_detector` — Balcony, cornice, column, and lintel detector for urban facades (Apache-2.0).

### Domain 4: Segmentation (15 Slots)
51. `sam2_hiera_t` — Segment Anything 2 Tiny for fast interactive masking (Apache-2.0).
52. `sam2_hiera_s` — Segment Anything 2 Small (Apache-2.0).
53. `sam2_hiera_b` — Segment Anything 2 Base (Apache-2.0).
54. `sam2_hiera_l` — Segment Anything 2 Large for fine object and part boundary masks (Apache-2.0).
55. `sam2_video_tracker` — Spatio-temporal mask propagation across video capture sequences (Apache-2.0).
56. `sam3_prospect` — Advanced multi-scale segmentation architecture (Apache-2.0).
57. `mask2former_swinl` — Universal panoptic, instance, and semantic segmentation (Apache-2.0).
58. `oneformer_ade20k` — Universal segmentation model trained on ADE20K 150 indoor categories (Apache-2.0).
59. `seem_large` — Segment Everything Everywhere with visual and audio prompts (Apache-2.0).
60. `entity_part_segmenter` — Sub-object part segmenter (chair legs, drawer handles, appliance doors) (Apache-2.0).
61. `grounded_sam2` — Grounding DINO + SAM2 zero-shot pipeline (Apache-2.0).
62. `cutler_cascade` — Unsupervised mask discovery for unknown novel objects (Apache-2.0).
63. `semantic_sam` — Multi-granularity part-to-whole segmentation (Apache-2.0).
64. `hq_sam` — High-Quality SAM for intricate boundaries, cables, and railings (Apache-2.0).
65. `fast_sam` — Real-time mobile-friendly CNN segmentation pipeline (Apache-2.0).

### Domain 5: Feature Matching & Tracking (12 Slots)
66. `superpoint` — Self-supervised interest point detector and descriptor (BSD-3-Clause).
67. `lightglue_superpoint` — Fast neural feature matcher for SuperPoint (Apache-2.0).
68. `disk` — Differentiable feature detector and descriptor for difficult illumination (Apache-2.0).
69. `lightglue_disk` — LightGlue matcher adapted for DISK features (Apache-2.0).
70. `loftr_outdoor` — Detector-free local feature matching with transformers (Apache-2.0).
71. `loftr_indoor` — Dense transformer matching across low-texture walls and floors (Apache-2.0).
72. `cotr` — Functional correspondence matching for extreme viewpoint variations (Apache-2.0).
73. `rosettastone_roma` — Robust dense optical and geometric matcher (MIT).
74. `cotracker_v2` — Dense long-term video point tracking for moving cameras (CC-BY-NC-4.0).
75. `tapir` — Tracking Any Point with per-frame refinement (Apache-2.0).
76. `omnimotion` — Quasi-3D representation for dense video correspondence (Apache-2.0).
77. `raft_optical_flow` — Recurrent all-pairs field transforms for high-speed motion estimation (BSD-3-Clause).

### Domain 6: Point-Cloud 3D Understanding (12 Slots)
78. `pointnet2_msg` — Multi-scale grouping point cloud semantic segmentation (MIT).
79. `votenet` — Deep Hough voting for 3D oriented bounding box detection (Apache-2.0).
80. `groupfree3d` — Transformer-based 3D object detection from point clouds (Apache-2.0).
81. `pointgroup` — Dual-set point grouping for 3D instance segmentation (MIT).
82. `spconv_minkowski` — Sparse 3D convolutional network for billion-point building clouds (Apache-2.0).
83. `farthest_point_sampler` — Deterministic geometric subsampling for dense point clouds (MIT).
84. `ransac_plane_fitter` — Multi-model robust architectural plane segmenter (MIT).
85. `fpfh_registration` — Fast Point Feature Histograms for global point cloud registration (BSD-3-Clause).
86. `colored_icp` — Photometric and geometric Iterative Closest Point fine registration (BSD-3-Clause).
87. `pointcept_ptv3` — Point Transformer V3 for large-scale urban infrastructure LiDAR/clouds (Apache-2.0).
88. `segment3d` — 3D promptable point-cloud instance segmenter (Apache-2.0).
89. `supervoxel_oversegmenter` — Geometrically faithful point cloud supervoxel partitioning (BSD-3-Clause).

### Domain 7: Room & Architectural Understanding (12 Slots)
90. `roomformer` — Transformer for complete interior room layout prediction (MIT).
91. `horizon_net` — 1D representation for 360-degree room layout reconstruction (Apache-2.0).
92. `manhattan_frame_estimator` — Orthogonal vanishing point detector and Manhattan world aligner (MIT).
93. `atlanta_frame_estimator` — Multi-axis non-orthogonal vertical plane alignment (MIT).
94. `wall_plane_extractor` — RANSAC + alpha-shape polygonal wall boundary generator (MIT).
95. `opening_detector_3d` — 3D window and door void bounding extractor in wall planes (MIT).
96. `ceiling_height_calibrator` — Robust vertical interval estimator using floor-ceiling normal pairs (MIT).
97. `multi_room_graph_builder` — Adjacency and portal graph constructor between rooms (MIT).
98. `structural_framing_inferer` — Post-and-beam structural grid inferer from load-bearing elements (MIT).
99. `corridor_tracker` — Long-trajectory corridor geometry and turn estimator (MIT).
100. `storey_separator` — Multi-storey vertical elevation splitter for high-rise buildings (MIT).
101. `ifc_wall_generator` — Direct IFC / STEP compliant boundary representation exporter (MIT).

### Domain 8: Material, Surface Normal & Lighting (8 Slots)
102. `material_net_pbr` — Albedo, roughness, metallic, and normal decomposition (Apache-2.0).
103. `surface_normal_dsine` — High-accuracy deep surface normal estimation from single images (Apache-2.0).
104. `omnidata_normals` — Multi-task surface normal estimation on indoor benchmark (Apache-2.0).
105. `intrinsic_decomposer` — Shading, reflectance, and ambient occlusion decomposition (MIT).
106. `hdr_envmap_estimator` — Spatially varying HDR spherical lighting environment estimator (Apache-2.0).
107. `specular_remover` — Specular highlight and reflection suppressor for photogrammetry (MIT).
108. `roughness_calibrator` — Micro-roughness estimator for physical materials (MIT).
109. `glass_transparency_detector` — Glass pane and reflective surface segmenter (Apache-2.0).

### Domain 9: Mesh & Topology (8 Slots)
110. `screened_poisson_recon` — Screened Poisson Surface Reconstruction for watertight meshes (MIT).
111. `alpha_shape_mesher` — Non-convex boundary mesh generator from unoriented point clouds (BSD-3-Clause).
112. `marching_cubes_tsdf` — Truncated Signed Distance Function volumetric zero-crossing mesher (MIT).
113. `ball_pivoting_recon` — High-precision ball pivoting for preservation of sharp corners (BSD-3-Clause).
114. `manifold_cleanup` — Non-manifold edge, vertex, and self-intersection repair engine (MIT).
115. `quad_remesher` — Architectural quad-dominant surface remeshing for CAD/IFC alignment (Apache-2.0).
116. `mesh_decimator_qslim` — Quadric error metric mesh decimation preserving planar boundaries (MIT).
117. `uv_atlas_unwrapper` — Conformal parameterization and UV atlas packing for textured meshes (MIT).

### Domain 10: Image Enhancement & Restoration (8 Slots)
118. `real_esrgan_x4` — 4x blind super-resolution for distant architectural and text details (BSD-3-Clause).
119. `restormer` — Transformer-based image restoration for motion deblurring (Apache-2.0).
120. `nafnet_deblur` — Non-linear activation free network for high-speed camera deblurring (Apache-2.0).
121. `exposure_fusion_hdr` — Mertens exposure fusion for high-contrast window and indoor shots (BSD-3-Clause).
122. `shadow_remover` — Hard shadow detector and illumination normalizer (Apache-2.0).
123. `lens_distortion_corrector` — Brown-Conrady radial and tangential distortion unwrapper (MIT).
124. `rolling_shutter_rectifier` — Velocity-compensated scanline rolling shutter undistorter (MIT).
125. `color_checker_calibrator` — Macbeth chart automatic color constancy and white-balance aligner (MIT).

---

## 4. Hardware Profiles & Memory Footprint

To ensure execution across resource-constrained devices, models are mapped to three hardware tier envelopes:

- **Tier 1 (High VRAM / Multi-GPU, 24GB+)**: Gaussian Splatting, NeRF, VGGT-Omega, Florence-2 Large, Grounding DINO Swin-L, Point Transformer V3.
- **Tier 2 (Consumer GPU, 8GB - 16GB)**: Depth Anything V2 Metric, SAM2 Large, MapAnything, Real-ESRGAN, Mask2Former.
- **Tier 3 (Edge / CPU Fallback, < 8GB)**: COLMAP CPU, LightGlue, SuperPoint, FastSAM, RANSAC plane fitting, Screened Poisson, HorizonNet.

Under the Mandatory Sequential Execution Policy, **GPU VRAM is released (`torch.cuda.empty_cache()` and garbage collection)** between model invocations, allowing devices with 8GB–16GB VRAM to run Tier 1/2 models sequentially without crashing.
