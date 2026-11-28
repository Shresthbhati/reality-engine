# Scene Understanding & Attribute Validation

## 1. Evidence-Backed Attribute Recognition (`perception/attributes/recognizer.py`)
- **Observed-only pixel & depth extraction:**
  - `dominant_color`, `surface_brightness`, `surface_texture` are computed exclusively from `ObservationRecord.mask` (preferred) or `ObservationRecord.bbox_px` crops when `image_rgb` is provided.
  - `relative_depth_band` (`foreground` / `midground` / `background`) is computed exclusively when a 2D `depth_map` is provided.
  - `2d_orientation` and `apparent_pixel_scale` are derived from `bbox_px`.
  - Material/operational state lexical cues (`wood`, `glass`, `metal`, `open`, `closed`) are emitted only when explicitly present in model label text.
- **No fabricated priors:** When `image_rgb=None` or `depth_map=None`, pixel and depth attributes are omitted (`n_inferred == 0`). Ontology priors are isolated in `infer_prior_attributes(..., provenance="inferred")`.

## 2. Observed vs Inferred Scene Graph (`perception/reasoning/scene_graph_builder.py`)
- **Observed relationships (`Provenance.OBSERVED`):** Emitted when same-image bbox containment `>= 0.70` pairs a component/fixture with a host (`wall`, `door`, `window`, `switchboard`, `cabinet`, `floor`, `table`) and (when `depth_maps` are available) relative depth difference is within `<= 35%`.
- **Inferred priors (`Provenance.INFERRED`):** Emitted only when a fixture (`switch`, `socket`, `ceiling_fan`, `light_fixture`, etc.) has no co-detected host in the view, targeting `inferred:<image_id>:<host>` at confidence `0.35`.

## 3. Automated Verification
- Verified by `tests/test_attributes_and_scene_graph.py` (`test_attributes_never_fabricate_color_or_depth_without_evidence`, `test_attributes_extract_mask_color_depth_and_lexical_cues`, `test_scene_graph_separates_observed_containment_from_inferred_priors`).
