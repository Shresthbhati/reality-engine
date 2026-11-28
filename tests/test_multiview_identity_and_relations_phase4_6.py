"""Phase 4 & 6 tests: multi-view identity, entity re-keying, and Y-up spatial scene graph."""
from __future__ import annotations

from engine.multiengine.association import MERGED, REFUSED, Candidate, associate
from engine.multiengine.multi_view_association import (
    MultiViewEntityObservation,
    compute_cross_view_pair_score,
    reconcile_associated_geometry,
)
from world_ir.entity_reid import rekey_entities
from world_ir.relations import Box, derive_relations, validate_scene_graph_consistency
from world_ir.schema_v1 import Entity, EntityType, Relationship, RelationshipKind
from world_ir.world_v1 import WorldIR


def test_derive_relations_supports_y_up_and_z_up() -> None:
    # Y-up: floor at y=[0, 0.1], chair resting at y=[0.1, 0.9]
    floor_y = Box("floor", (0.0, 0.0, 0.0), (5.0, 0.1, 5.0))
    chair_y = Box("chair", (1.0, 0.1, 1.0), (1.6, 0.9, 1.6))
    rels_y = derive_relations([floor_y, chair_y], up_axis=1)
    assert ("chair", "on", "floor") in rels_y

    # Z-up: floor at z=[0, 0.1], chair resting at z=[0.1, 0.9]
    floor_z = Box("floor", (0.0, 0.0, 0.0), (5.0, 5.0, 0.1))
    chair_z = Box("chair", (1.0, 1.0, 0.1), (1.6, 1.6, 0.9))
    rels_z = derive_relations([floor_z, chair_z], up_axis=2)
    assert ("chair", "on", "floor") in rels_z


def test_scene_graph_consistency_flags_far_attachment_and_multi_parent() -> None:
    wall = Box("wall_1", (0.0, 0.0, 0.0), (0.2, 3.0, 4.0), is_wall=True)
    switch_far = Box("sw_1", (2.5, 1.0, 1.0), (2.6, 1.1, 1.1))
    bad = validate_scene_graph_consistency(
        [wall, switch_far],
        [("sw_1", "attached_to", "wall_1"), ("sw_1", "part_of", "wall_1"), ("sw_1", "part_of", "other")],
    )
    assert len(bad) >= 2


def test_rekey_entities_preserves_parent_and_boundary_links() -> None:
    w = WorldIR()
    w.entities["cand_wall"] = Entity(id="cand_wall", name="Wall", type=EntityType.WALL)
    w.entities["cand_room"] = Entity(
        id="cand_room",
        name="Room",
        type=EntityType.ROOM,
        relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="cand_wall")],
        custom_properties={"boundary_element_ids": ["cand_wall"], "host_wall_id": "cand_wall"},
    )
    w.metadata["association_decisions"] = [{"entity_id": "cand_wall", "state": "matched"}]

    applied = rekey_entities(w, {"cand_wall": "prev_wall"})
    assert applied == {"cand_wall": "prev_wall"}
    assert "prev_wall" in w.entities
    room = w.entities["cand_room"]
    assert room.relationships[0].target_id == "prev_wall"
    assert room.custom_properties["boundary_element_ids"] == ["prev_wall"]
    assert room.custom_properties["host_wall_id"] == "prev_wall"
    assert w.metadata["association_decisions"][0]["entity_id"] == "prev_wall"


def test_multiview_identity_merges_same_object_and_keeps_neighbors_distinct() -> None:
    obs = [
        # Same chair seen from view 1 and view 2
        MultiViewEntityObservation(
            observation_id="o1", source_id="img1.jpg", class_name="chair",
            position=(1.0, 0.45, 1.0), dimensions=(0.55, 0.55, 0.9),
            scale_units="METRES", is_physically_metric=True,
            position_uncertainty=(0.05, 0.05, 0.05), confidence=0.9,
        ),
        MultiViewEntityObservation(
            observation_id="o2", source_id="img2.jpg", class_name="chair",
            position=(1.04, 0.46, 1.02), dimensions=(0.56, 0.54, 0.89),
            scale_units="METRES", is_physically_metric=True,
            position_uncertainty=(0.05, 0.05, 0.05), confidence=0.88,
        ),
        # Neighboring similar chair 1.5m away in img1 (must NOT merge with o1 or o2)
        MultiViewEntityObservation(
            observation_id="o3", source_id="img1.jpg", class_name="chair",
            position=(2.5, 0.45, 1.0), dimensions=(0.55, 0.55, 0.9),
            scale_units="METRES", is_physically_metric=True,
            position_uncertainty=(0.05, 0.05, 0.05), confidence=0.85,
        ),
    ]
    obs_map = {o.observation_id: o for o in obs}
    cands = [
        Candidate(
            observation_id=o.observation_id,
            source_id=o.source_id,
            label=o.class_name,
        )
        for o in obs
    ]
    pair_scores = []
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            pair_scores.append(compute_cross_view_pair_score(obs[i], obs[j]))
    res = associate(cands, pair_scores)
    entities = [reconcile_associated_geometry(ent, obs_map) for ent in res.entities]
    assert len(entities) == 2
    assert any(d.outcome == MERGED for d in res.decisions)
    assert any(d.outcome == REFUSED for d in res.decisions)
