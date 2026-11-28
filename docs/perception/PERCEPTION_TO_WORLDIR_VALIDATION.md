# Perception-to-WorldIR End-to-End Validation

## 1. End-to-End Pipeline Contract
Verified data flow from raw perception outputs to persistent `WorldStore` state:
1. **`ObservationRecord` (`perception/engine_evidence.py`)**: Carries `source_image_id`, `bbox_px`, `mask`, `confidence`, `calibration_quality`, and ontology grounding (`ontology_path`).
2. **`MaskStore` (`perception/mask_persistence.py`, `world_ir/mask_store.py`)**: `persist_masks()` encodes in-memory boolean masks as run-length `.npz` entries and stamps `ObservationRecord.mask_ref`.
3. **Attributes (`perception/attributes/recognizer.py`)**: `extract_observed_attributes()` attaches pixel/depth/geometry-backed attributes (`provenance="observed"`) without fabricating priors.
4. **Scene Graph (`perception/reasoning/scene_graph_builder.py`)**: `build_scene_graph_from_observations()` + `to_world_ir_relationships()` produce `world_ir.schema_v1.Relationship`s with explicit `Provenance.OBSERVED` vs `Provenance.INFERRED`.
5. **`WorldIR` -> `WorldStore` (`worldstore/store.py`)**: `WorldStore(root).save_version(world, parent=None)` persists immutable, content-verified `WorldIR` snapshots that reload identically across process restarts.

## 2. Difficult-Image Failure Regimes (`tests/test_difficult_images_and_worldir.py`)
Automated stress coverage across 10 adversarial/degraded conditions:
- Low light (`is_underexposed`), glare/overexposure (`is_overexposed`), directional motion blur (`is_motion_blurred`), low-resolution small-object starvation, textureless drywall (`TEXTURELESS_FEATURE_STARVATION`), reflective/glass depth inversion (`DEPTH_INVERTED_OR_FLAT`), low-contrast segmentation leakage (`SEGMENTATION_BOUNDARY_LEAKAGE`), wide-baseline SfM disconnection (`SFM_LOW_OVERLAP_DISCONNECTION`), symmetrical furniture embedding confusion (`EMBEDDING_SYMMETRY_CONFUSION`), and malformed/NaN observation metadata.
