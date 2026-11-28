# Advanced Perception Roadmap

## Completed Foundations
1. **Ontology & Small-Object Tiling (PR #417, #419):** Canonical `config/ontology.yaml` taxonomy, `ground_label()` resolution, `tile_scale=2` tiled Grounding DINO detection.
2. **Multi-View Identity & Scene Graph (PR #420):** Cross-view association and structural plausibility checks.
3. **Real-Data & 6 GB GPU Optimization (PR #421, PR-A):** LightGlue keypoint rescaling, orchestrator `max` VRAM reporting + inter-engine cache cleanup, Mask R-CNN `MaskStore` persistence, strict offline depth loading.
4. **Evidence-Backed Attributes & Observed/Inferred Scene Graph (PR-B):** `perception/attributes/recognizer.py` and `perception/reasoning/scene_graph_builder.py`.

## Next Milestones
1. **PR-C (Ground-Truth Dataset & Benchmark Runner):** COCO/identity/relationship benchmark suite computing AP50, AP50:95, per-class AP, small-object recall, mask IoU, and scene-graph relation accuracy.
2. **PR-D (Difficult-Image Failure Tests & Perception-to-WorldIR End-to-End):** Stress suite for low-light, blur, glare, glass/mirrors, occlusion, and full `Observation -> MaskStore -> Association -> Entity -> Relationship -> WorldIR -> WorldStore` validation.
3. **SAM2 6 GB Tiling / Downscale (Separate Owner):** Reduce `segmentation_adapter.py` peak allocation below 6 GB on 12 MP indoor captures.
