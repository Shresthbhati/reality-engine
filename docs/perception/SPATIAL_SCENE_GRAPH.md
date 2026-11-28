# Spatial Scene Graph (Phase 6)

**Core Implementation:**
- `world_ir/relations.py` (`derive_relations`, `validate_scene_graph_consistency`, `Box`, `RELATION_KINDS`)
- `engine/multiengine/spatial_intelligence/relationships.py` & `scene_graph.py`
- `perception/architecture/space_graph.py` (`InteriorSpaceGraph`)

## Relationship & Consistency Rules

1. **Coordinate-Frame-Aware Derivation:** `derive_relations(boxes, up_axis=1|2)` supports both `Frame.Y_UP` (`up_axis=1`, horizontal `(0, 2)`) and `Frame.Z_UP` (`up_axis=2`, horizontal `(0, 1)`).
2. **Evidence-Backed Edges:** Relationships (`on`, `attached_to`, `inside`, `contains`, `adjacent_to`, `hosted_by`, `part_of`, `connects`, `supports`) are derived from 3D bounding geometry and supporting observations — never from 2D image proximity alone.
3. **Graph Consistency Validation:** `validate_scene_graph_consistency()` verifies that `attached_to`/`hosted_by` entities are within `max_attach_gap_m`, flags dangling references, and rejects contradictory `part_of` multi-parent edges.
