"""Sprint 7 — Scale-Safety Adversarial Tests.

Verifies that:
1. ``RelationshipEngine.evaluate_relationships`` marks metric-distance-dependent
   relationships as ``"hypothesized"`` (not ``"validated"``) when one or both
   entities are NOT metric-anchored.
2. Relative-depth entities never receive ``MOUNTED_ON`` or ``SUPPORTED_BY``
   relationships with ``validation_status="validated"``.
3. Entities in different coordinate frames produce ``"hypothesized"`` status.
4. Scale-invariant relationships (PART_OF from explicit parent/child,
   ALIGNED_WITH from surface normals) always remain ``"validated"``.
5. Metric-anchored entities with the same coordinate frame do receive
   ``"validated"`` distance-based relationships.
6. ``ObjectGeometryEvidence`` carries scale_state and is_physically_metric.
7. ``GeometryValidationResult.metric_check_applicable`` is False when evidence
   is not metric.
8. Cross-scale comparison is explicitly flagged (entity A is metric, B is not).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from engine.multiengine.spatial_intelligence.relationships import (
    EvidenceRelationship,
    RelationshipEngine,
    RelationshipType,
    _entity_is_metric,
    _entities_share_frame,
    _metric_validation_status,
)
from engine.multiengine.spatial_intelligence.geometry_validator import (
    GeometryValidationResult,
    ObjectGeometryEvidence,
    _evidence_is_metric,
    validate_object_geometry,
)


# ---------------------------------------------------------------------------
# Minimal entity stubs
# ---------------------------------------------------------------------------

@dataclass
class _Entity:
    """Minimal entity stub used to drive the relationship engine."""
    entity_id: str
    class_name: str
    position_m: tuple[float, float, float] | None = None
    surface_normal: tuple[float, float, float] | None = None
    bounds_min: tuple[float, float, float] | None = None
    bounds_max: tuple[float, float, float] | None = None
    parent_id: str | None = None
    child_ids: tuple[str, ...] = ()
    scale_state: str = "relative_depth"
    is_physically_metric: bool = False
    coordinate_frame: str = "world:canonical"


def _metric_entity(entity_id: str, class_name: str, pos: tuple, **kw) -> _Entity:
    return _Entity(
        entity_id=entity_id,
        class_name=class_name,
        position_m=pos,
        scale_state="metric_anchored",
        is_physically_metric=True,
        coordinate_frame="world:canonical",
        **kw,
    )


def _relative_entity(entity_id: str, class_name: str, pos: tuple, **kw) -> _Entity:
    return _Entity(
        entity_id=entity_id,
        class_name=class_name,
        position_m=pos,
        scale_state="relative_depth",
        is_physically_metric=False,
        coordinate_frame="world:canonical",
        **kw,
    )


def _sfm_entity(entity_id: str, class_name: str, pos: tuple, **kw) -> _Entity:
    return _Entity(
        entity_id=entity_id,
        class_name=class_name,
        position_m=pos,
        scale_state="sfm_gauge",
        is_physically_metric=False,
        coordinate_frame="sfm:reconstruction_0",
        **kw,
    )


# ---------------------------------------------------------------------------
# 1. entity_is_metric helper
# ---------------------------------------------------------------------------

class TestEntityIsMetric:
    def test_metric_anchored_with_flag(self):
        ent = _metric_entity("a", "socket", (0, 0, 1))
        assert _entity_is_metric(ent) is True

    def test_relative_depth(self):
        ent = _relative_entity("a", "chair", (0, 0, 0))
        assert _entity_is_metric(ent) is False

    def test_sfm_gauge(self):
        ent = _sfm_entity("a", "table", (0, 0, 0))
        assert _entity_is_metric(ent) is False

    def test_metric_state_without_physical_flag(self):
        ent = _Entity(
            entity_id="a",
            class_name="door",
            position_m=(0, 0, 0),
            scale_state="metric_anchored",
            is_physically_metric=False,  # flag not set
        )
        assert _entity_is_metric(ent) is False

    def test_none_scale_state(self):
        ent = _Entity(entity_id="a", class_name="wall", position_m=(0, 0, 0), scale_state=None)
        assert _entity_is_metric(ent) is False


# ---------------------------------------------------------------------------
# 2. _entities_share_frame
# ---------------------------------------------------------------------------

class TestEntitiesShareFrame:
    def test_same_frame(self):
        a = _metric_entity("a", "socket", (0, 0, 1))
        b = _metric_entity("b", "wall", (0, 0, 0), surface_normal=(1, 0, 0))
        assert _entities_share_frame(a, b) is True

    def test_different_frames(self):
        a = _metric_entity("a", "table", (0, 0, 0))
        b = _sfm_entity("b", "chair", (0, 0.5, 0))
        assert _entities_share_frame(a, b) is False

    def test_none_frame_treated_as_compatible(self):
        a = _Entity(entity_id="a", class_name="lamp", position_m=(0, 0, 2), coordinate_frame=None)
        b = _Entity(entity_id="b", class_name="ceiling", position_m=(0, 0, 2.2), coordinate_frame=None)
        assert _entities_share_frame(a, b) is True


# ---------------------------------------------------------------------------
# 3. RelationshipEngine — relative-depth entities never get "validated" metric rels
# ---------------------------------------------------------------------------

class TestRelationshipEngineScaleSafety:

    def _run(self, entities):
        engine = RelationshipEngine()
        return engine.evaluate_relationships(entities)

    def test_relative_depth_wall_mounting_is_hypothesized(self):
        """Socket on wall: both relative-depth → mounting relationship must be hypothesized."""
        socket = _relative_entity("s1", "socket", (1.0, 0.05, 1.2), surface_normal=(0, 1, 0))
        wall = _relative_entity("w1", "wall", (1.0, 0.0, 1.0), surface_normal=(0, 1, 0))
        rels = self._run([socket, wall])
        mounted = [r for r in rels if r.relation_type == RelationshipType.MOUNTED_ON
                   and r.source_id == "s1" and r.target_id == "w1"]
        assert mounted, "MOUNTED_ON relation should still be inferred"
        for r in mounted:
            assert r.validation_status == "hypothesized", (
                f"Non-metric mounting must be 'hypothesized', got '{r.validation_status}'"
            )

    def test_metric_wall_mounting_is_validated(self):
        """Socket on wall: both metric and same frame → mounting should be validated."""
        socket = _metric_entity("s1", "socket", (1.0, 0.05, 1.2))
        socket.surface_normal = None
        wall = _metric_entity("w1", "wall", (1.0, 0.0, 1.0))
        # Give wall a horizontal normal pointing in y-direction
        wall_with_normal = _Entity(
            entity_id="w1",
            class_name="wall",
            position_m=(1.0, 0.0, 1.0),
            surface_normal=(0.0, 1.0, 0.0),
            scale_state="metric_anchored",
            is_physically_metric=True,
            coordinate_frame="world:canonical",
        )
        socket_ent = _Entity(
            entity_id="s1",
            class_name="socket",
            position_m=(1.0, 0.05, 1.2),
            scale_state="metric_anchored",
            is_physically_metric=True,
            coordinate_frame="world:canonical",
        )
        rels = self._run([socket_ent, wall_with_normal])
        mounted = [r for r in rels if r.relation_type == RelationshipType.MOUNTED_ON
                   and r.source_id == "s1" and r.target_id == "w1"]
        assert mounted, "MOUNTED_ON relation should be inferred for metric entities"
        for r in mounted:
            assert r.validation_status == "validated", (
                f"Metric-anchored mounting on same frame must be 'validated', got '{r.validation_status}'"
            )

    def test_sfm_gauge_support_is_hypothesized(self):
        """Chair on floor: SfM-gauge entities → SUPPORTED_BY must be hypothesized."""
        chair = _sfm_entity("c1", "chair", (1.0, 1.0, 0.45),
                             bounds_min=(0.9, 0.9, 0.0), bounds_max=(1.1, 1.1, 0.45))
        floor = _sfm_entity("f1", "floor", (0.0, 0.0, 0.0),
                             bounds_min=(-10.0, -10.0, 0.0), bounds_max=(10.0, 10.0, 0.05))
        rels = self._run([chair, floor])
        supported = [r for r in rels if r.relation_type == RelationshipType.SUPPORTED_BY
                     and r.source_id == "c1"]
        if supported:
            for r in supported:
                assert r.validation_status == "hypothesized", (
                    f"SfM-gauge support must be 'hypothesized', got '{r.validation_status}'"
                )

    def test_cross_frame_relationship_is_hypothesized(self):
        """Entities from different coordinate frames → all distance-based rels hypothesized."""
        metric_ent = _metric_entity("m1", "table", (0.0, 0.0, 0.0))
        sfm_ent = _sfm_entity("s1", "chair", (0.5, 0.0, 0.0))
        rels = self._run([metric_ent, sfm_ent])
        distance_rels = [r for r in rels if r.relation_type in {
            RelationshipType.ADJACENT_TO,
            RelationshipType.SUPPORTED_BY,
            RelationshipType.MOUNTED_ON,
            RelationshipType.ABOVE,
            RelationshipType.BELOW,
        }]
        for r in distance_rels:
            assert r.validation_status == "hypothesized", (
                f"Cross-frame distance-based relation must be 'hypothesized', got '{r.validation_status}'"
            )

    def test_explicit_part_of_always_validated(self):
        """PART_OF from explicit parent/child IDs is scale-invariant and must be 'validated'."""
        child = _relative_entity("handle1", "door_handle", (1.0, 0.0, 1.5))
        child.parent_id = "door1"
        parent = _relative_entity("door1", "door", (1.0, 0.0, 1.0))
        # patch parent_id via object attribute (dataclass)
        import dataclasses
        child2 = dataclasses.replace(child, parent_id="door1")
        rels = self._run([child2, parent])
        part_of = [r for r in rels if r.relation_type == RelationshipType.PART_OF
                   and r.source_id == "handle1" and r.target_id == "door1"]
        assert part_of, "PART_OF should be inferred from explicit parent/child"
        for r in part_of:
            assert r.validation_status == "validated", (
                f"Structural PART_OF must be 'validated', got '{r.validation_status}'"
            )

    def test_aligned_with_always_validated(self):
        """ALIGNED_WITH from surface normals is scale-invariant and must be 'validated'."""
        a = _relative_entity("a1", "wall", (0.0, 0.0, 1.0), surface_normal=(1.0, 0.0, 0.0))
        b = _relative_entity("b1", "wall", (3.0, 0.0, 1.0), surface_normal=(1.0, 0.0, 0.0))
        rels = self._run([a, b])
        aligned = [r for r in rels if r.relation_type == RelationshipType.ALIGNED_WITH]
        assert aligned, "ALIGNED_WITH should be inferred from parallel normals"
        for r in aligned:
            assert r.validation_status == "validated", (
                f"Normal-based ALIGNED_WITH must be 'validated', got '{r.validation_status}'"
            )

    def test_above_below_relative_depth_hypothesized(self):
        """ABOVE/BELOW using z-coordinates of non-metric entities must be hypothesized."""
        high = _relative_entity("h1", "lamp", (0.0, 0.0, 2.5))
        low = _relative_entity("l1", "table", (0.0, 0.0, 0.8))
        rels = self._run([high, low])
        above = [r for r in rels if r.relation_type == RelationshipType.ABOVE
                 and r.source_id == "h1"]
        assert above, "ABOVE should be inferred from coordinate difference"
        for r in above:
            assert r.validation_status == "hypothesized", (
                f"Non-metric ABOVE must be 'hypothesized', got '{r.validation_status}'"
            )

    def test_adjacency_non_metric_hypothesized(self):
        """ADJACENT_TO with non-metric entities must be hypothesized."""
        a = _relative_entity("a1", "sofa", (0.0, 0.0, 0.5))
        b = _relative_entity("b1", "table", (0.8, 0.0, 0.5))
        rels = self._run([a, b])
        adj = [r for r in rels if r.relation_type == RelationshipType.ADJACENT_TO]
        assert adj, "ADJACENT_TO should be inferred"
        for r in adj:
            assert r.validation_status == "hypothesized", (
                f"Non-metric ADJACENT_TO must be 'hypothesized', got '{r.validation_status}'"
            )

    def test_geometric_evidence_carries_threshold(self):
        """geometric_evidence dict must include threshold_m for metric-dependent rels."""
        socket_ent = _Entity(
            entity_id="s1",
            class_name="socket",
            position_m=(1.0, 0.05, 1.2),
            scale_state="relative_depth",
            is_physically_metric=False,
            coordinate_frame="world:canonical",
        )
        wall_ent = _Entity(
            entity_id="w1",
            class_name="wall",
            position_m=(1.0, 0.0, 1.0),
            surface_normal=(0.0, 1.0, 0.0),
            scale_state="relative_depth",
            is_physically_metric=False,
            coordinate_frame="world:canonical",
        )
        rels = self._run([socket_ent, wall_ent])
        mounted = [r for r in rels if r.relation_type == RelationshipType.MOUNTED_ON]
        assert mounted
        for r in mounted:
            assert "threshold_m" in r.geometric_evidence, (
                "geometric_evidence must carry 'threshold_m' for metric-dependent relationship"
            )


# ---------------------------------------------------------------------------
# 4. ObjectGeometryEvidence — scale_state fields
# ---------------------------------------------------------------------------

class TestObjectGeometryEvidenceScaleFields:

    def test_evidence_has_scale_state_field(self):
        ev = ObjectGeometryEvidence(
            box_2d=(0.0, 0.0, 100.0, 200.0),
            class_name="door",
            scale_state="relative_depth",
            is_physically_metric=False,
        )
        assert ev.scale_state == "relative_depth"
        assert ev.is_physically_metric is False

    def test_evidence_has_coordinate_frame_field(self):
        ev = ObjectGeometryEvidence(
            box_2d=(0.0, 0.0, 100.0, 200.0),
            class_name="window",
            coordinate_frame="sfm:reconstruction_0",
        )
        assert ev.coordinate_frame == "sfm:reconstruction_0"

    def test_evidence_is_metric_true(self):
        ev = ObjectGeometryEvidence(
            box_2d=(0.0, 0.0, 100.0, 200.0),
            class_name="door",
            scale_state="metric_anchored",
            is_physically_metric=True,
            extent_3d=(0.9, 0.1, 2.1),
        )
        assert _evidence_is_metric(ev) is True

    def test_evidence_is_metric_false_for_relative(self):
        ev = ObjectGeometryEvidence(
            box_2d=(0.0, 0.0, 100.0, 200.0),
            class_name="door",
            scale_state="relative_depth",
            is_physically_metric=False,
            extent_3d=(0.9, 0.1, 2.1),
        )
        assert _evidence_is_metric(ev) is False


# ---------------------------------------------------------------------------
# 5. GeometryValidationResult — metric_check_applicable
# ---------------------------------------------------------------------------

class TestGeometryValidatorMetricCheckApplicable:

    def test_door_non_metric_metric_check_not_applicable(self):
        ev = ObjectGeometryEvidence(
            box_2d=(10.0, 20.0, 110.0, 250.0),
            class_name="door",
            scale_state="relative_depth",
            is_physically_metric=False,
        )
        result = validate_object_geometry(ev)
        assert result.metric_check_applicable is False, (
            "Door with relative-depth evidence must not apply metric bounds"
        )

    def test_door_metric_check_applicable(self):
        ev = ObjectGeometryEvidence(
            box_2d=(10.0, 20.0, 110.0, 250.0),
            class_name="door",
            scale_state="metric_anchored",
            is_physically_metric=True,
            extent_3d=(0.9, 0.05, 2.1),
        )
        result = validate_object_geometry(ev)
        assert result.metric_check_applicable is True

    def test_door_non_metric_not_marked_invalid_for_metric_check(self):
        """Non-metric door must NOT be invalid solely because metric height check is inapplicable."""
        ev = ObjectGeometryEvidence(
            box_2d=(10.0, 20.0, 110.0, 250.0),  # tall door aspect ratio
            class_name="door",
            scale_state="relative_depth",
            is_physically_metric=False,
        )
        result = validate_object_geometry(ev)
        # With good aspect ratio and no metric claims, should be valid
        assert result.is_valid is True, (
            "Non-metric door with valid aspect ratio must not be marked invalid"
        )

    def test_window_non_metric_not_applicable(self):
        ev = ObjectGeometryEvidence(
            box_2d=(50.0, 50.0, 200.0, 200.0),
            class_name="window",
            scale_state="sfm_gauge",
            is_physically_metric=False,
        )
        result = validate_object_geometry(ev)
        assert result.metric_check_applicable is False

    def test_validation_result_to_dict_includes_metric_check(self):
        ev = ObjectGeometryEvidence(
            box_2d=(10.0, 20.0, 110.0, 250.0),
            class_name="door",
            scale_state="relative_depth",
            is_physically_metric=False,
        )
        result = validate_object_geometry(ev)
        d = result.to_dict()
        assert "metric_check_applicable" in d, (
            "to_dict() must expose metric_check_applicable"
        )
        assert d["metric_check_applicable"] is False


# ---------------------------------------------------------------------------
# 6. No relationships are silently validated for incompatible frames
# ---------------------------------------------------------------------------

class TestNoSilentCrossFrameValidation:

    def test_all_distance_rels_hypothesized_across_frames(self):
        """When entities live in incompatible coordinate frames, no distance-based
        relationship should be validation_status='validated'."""
        engine = RelationshipEngine()
        metric_ent = _Entity(
            entity_id="m1",
            class_name="table",
            position_m=(1.0, 1.0, 0.75),
            scale_state="metric_anchored",
            is_physically_metric=True,
            coordinate_frame="world:canonical",
        )
        sfm_ent = _Entity(
            entity_id="s1",
            class_name="chair",
            position_m=(1.5, 1.0, 0.45),
            scale_state="sfm_gauge",
            is_physically_metric=False,
            coordinate_frame="sfm:reconstruction_0",
        )
        rels = engine.evaluate_relationships([metric_ent, sfm_ent])
        distance_dependent = {
            RelationshipType.ADJACENT_TO,
            RelationshipType.SUPPORTED_BY,
            RelationshipType.MOUNTED_ON,
            RelationshipType.EMBEDDED_IN,
            RelationshipType.ATTACHED_TO,
            RelationshipType.ABOVE,
            RelationshipType.BELOW,
        }
        for r in rels:
            if r.relation_type in distance_dependent:
                assert r.validation_status != "validated", (
                    f"Cross-frame {r.relation_type.value} must not be 'validated', "
                    f"got '{r.validation_status}' for ({r.source_id} → {r.target_id})"
                )


# ---------------------------------------------------------------------------
# 7. Sprint 7 Relationship Types: OPENS_INTO, CONNECTED_TO, INTERSECTS
# ---------------------------------------------------------------------------

class TestSprint7RelationshipTypes:

    def test_door_opens_into_corridor(self):
        engine = RelationshipEngine()
        door = _metric_entity("d1", "door", (0.0, 0.0, 1.0))
        corridor = _metric_entity("c1", "corridor", (0.0, 1.5, 1.0))
        rels = engine.evaluate_relationships([door, corridor])
        opens = [r for r in rels if r.relation_type == RelationshipType.OPENS_INTO
                 and r.source_id == "d1" and r.target_id == "c1"]
        assert opens, "Door must have OPENS_INTO relationship with corridor"
        assert opens[0].validation_status == "validated"

    def test_pipe_connected_to_sink(self):
        engine = RelationshipEngine()
        pipe = _metric_entity("p1", "pipe", (1.0, 1.0, 0.40))
        sink = _metric_entity("s1", "sink", (1.0, 1.0, 0.85))
        rels = engine.evaluate_relationships([pipe, sink])
        connected = [r for r in rels if r.relation_type == RelationshipType.CONNECTED_TO
                     and r.source_id == "p1" and r.target_id == "s1"]
        assert connected, "Plumbing pipe must have CONNECTED_TO relationship with sink"
        assert connected[0].validation_status == "validated"

    def test_orthogonal_walls_intersect(self):
        engine = RelationshipEngine()
        wall_north = _metric_entity("w_n", "wall", (0.0, 2.0, 1.2), surface_normal=(0.0, 1.0, 0.0))
        wall_east = _metric_entity("w_e", "wall", (2.0, 0.0, 1.2), surface_normal=(1.0, 0.0, 0.0))
        rels = engine.evaluate_relationships([wall_north, wall_east])
        intersecting = [r for r in rels if r.relation_type == RelationshipType.INTERSECTS]
        assert intersecting, "Orthogonal walls within junction proximity must INTERSECT"
        assert intersecting[0].validation_status == "validated"


# ---------------------------------------------------------------------------
# 8. Sprint 7 Multi-View Identity: Host and subparts must not be merged
# ---------------------------------------------------------------------------

class TestMultiViewIdentityParentPartExclusion:

    def test_door_and_door_handle_not_semantically_compatible_for_identity(self):
        from engine.multiengine.spatial_intelligence.taxonomy import are_semantically_compatible
        # General compatibility allows parent-part
        assert are_semantically_compatible("door", "door_handle", allow_parent_part=True) is True
        # Identity association strictly forbids parent-part false merges
        assert are_semantically_compatible("door", "door_handle", allow_parent_part=False) is False
        assert are_semantically_compatible("door_handle", "door", allow_parent_part=False) is False
        assert are_semantically_compatible("sink", "faucet", allow_parent_part=False) is False
        assert are_semantically_compatible("faucet", "sink", allow_parent_part=False) is False

    def test_same_class_and_synonyms_are_compatible_for_identity(self):
        from engine.multiengine.spatial_intelligence.taxonomy import are_semantically_compatible
        assert are_semantically_compatible("door", "door", allow_parent_part=False) is True
        assert are_semantically_compatible("socket", "outlet", allow_parent_part=False) is True
        assert are_semantically_compatible("table", "desk", allow_parent_part=False) is True

