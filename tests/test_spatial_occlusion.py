"""Tests for object-level occlusion reasoning and object permanence.

These tests verify that ``perception.spatial_intelligence.occlusion`` builds a
directed object->object occlusion graph from per-view box + depth evidence,
computes transitive occlusion chains, derives per-object occluded regions, and
judges object permanence (an object persists even while hidden).
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.spatial_intelligence.occlusion import (
    OccludedRegion,
    OcclusionGraph,
    OcclusionRelation,
    ObjectPermanence,
    compute_object_occlusions,
    merge_graphs,
    object_permanence_in_view,
    occluded_region_of,
    transitive_occlusion_chain,
)


def _box(x0, y0, x1, y1):
    return (float(x0), float(y0), float(x1), float(y1))


# ------------------------------------------------------------------ occlusion graph construction

class TestOcclusionGraphConstruction:
    def test_nearer_object_occludes_farther(self):
        # Box A (front, closer) overlaps Box B (farther). Depth says A is nearer.
        depth = np.full((100, 100), 5.0, dtype=float)
        depth[10:50, 10:50] = 2.0   # front nearer
        depth[30:70, 30:70] = 5.0   # back farther
        graph = compute_object_occlusions(
            objects={
                "front": _box(10, 10, 50, 50),
                "back": _box(30, 30, 70, 70),
            },
            camera_id="cam1",
            depth_map=depth,
            depth_metric=True,
        )
        # front occludes back (front is nearer).
        assert graph.directly_occludes("front") == ("back",)
        assert graph.directly_occludes("back") == ()
        rel = graph.by_camera["cam1"][0]
        assert rel.occluder_id == "front"
        assert rel.occluded_id == "back"
        assert rel.metric is True
        assert rel.depth_separation_m is not None
        assert rel.depth_separation_m > 0

    def test_no_overlap_no_relation(self):
        graph = compute_object_occlusions(
            objects={
                "left": _box(0, 0, 10, 10),
                "right": _box(100, 100, 110, 110),
            },
            camera_id="cam1",
        )
        assert graph.relations == ()
        assert graph.directly_occludes("left") == ()

    def test_no_depth_weak_2d_only(self):
        graph = compute_object_occlusions(
            objects={
                "a": _box(0, 0, 50, 50),
                "b": _box(25, 25, 75, 75),
            },
            camera_id="cam1",
        )
        # Without depth, both directions recorded as potential (weak, non-metric).
        assert len(graph.relations) == 2
        for r in graph.relations:
            assert r.metric is False
            assert r.depth_separation_m is None
            assert r.confidence < 0.5

    def test_identical_depth_no_definite_occlusion(self):
        # Same depth -> ambiguous, both directions weak.
        depth = np.full((100, 100), 3.0, dtype=float)
        graph = compute_object_occlusions(
            objects={
                "a": _box(0, 0, 50, 50),
                "b": _box(25, 25, 75, 75),
            },
            camera_id="cam1",
            depth_map=depth,
            depth_metric=True,
        )
        assert len(graph.relations) == 2
        for r in graph.relations:
            assert r.metric is False
            assert r.confidence < 0.5

    def test_depth_separation_recorded(self):
        depth = np.full((100, 100), 10.0, dtype=float)
        depth[10:60, 10:60] = 2.0   # object a much closer than b
        graph = compute_object_occlusions(
            objects={
                "a": _box(0, 0, 60, 60),
                "b": _box(30, 30, 90, 90),
            },
            camera_id="cam1",
            depth_map=depth,
            depth_metric=True,
        )
        rel = graph.by_camera["cam1"][0]
        assert rel.occluder_id == "a"
        assert rel.depth_separation_m == pytest.approx(8.0)
        assert rel.metric is True


# ------------------------------------------------------------------ transitive chains

class TestTransitiveOcclusion:
    def test_chain_a_occludes_b_occludes_c(self):
        # g1: a nearer than b (cam1).
        depth1 = np.full((100, 100), 5.0, dtype=float)
        depth1[0:40, 0:40] = 2.0   # a nearer
        g1 = compute_object_occlusions(
            objects={"a": _box(0, 0, 40, 40), "b": _box(20, 20, 60, 60)},
            camera_id="cam1",
            depth_map=depth1,
            depth_metric=True,
        )
        # g2: b nearer than c (cam2).
        depth2 = np.full((100, 100), 5.0, dtype=float)
        depth2[20:60, 20:60] = 2.0   # b nearer
        g2 = compute_object_occlusions(
            objects={"b": _box(20, 20, 60, 60), "c": _box(40, 40, 80, 80)},
            camera_id="cam2",
            depth_map=depth2,
            depth_metric=True,
        )
        graph = merge_graphs(g1, g2)
        chain = transitive_occlusion_chain(graph, "a")
        assert "b" in chain
        assert "c" in chain  # a -> b -> c transitively

    def test_chain_excludes_start(self):
        depth = np.full((100, 100), 5.0, dtype=float)
        depth[0:40, 0:40] = 2.0   # a nearer
        g = compute_object_occlusions(
            objects={"a": _box(0, 0, 40, 40), "b": _box(20, 20, 60, 60)},
            camera_id="cam1",
            depth_map=depth,
            depth_metric=True,
        )
        assert transitive_occlusion_chain(g, "a") == ("b",)
        assert transitive_occlusion_chain(g, "b") == ()


# ------------------------------------------------------------------ merge graphs (multi-camera)

class TestMergeGraphs:
    def test_union_of_cameras(self):
        # g1: a nearer than b (cam1).
        depth1 = np.full((100, 100), 5.0, dtype=float)
        depth1[0:40, 0:40] = 2.0   # a nearer
        g1 = compute_object_occlusions(
            objects={"a": _box(0, 0, 40, 40), "b": _box(20, 20, 60, 60)},
            camera_id="cam1",
            depth_map=depth1,
            depth_metric=True,
        )
        # g2: b nearer than c (cam2).
        depth2 = np.full((100, 100), 5.0, dtype=float)
        depth2[20:60, 20:60] = 2.0   # b nearer
        g2 = compute_object_occlusions(
            objects={"b": _box(20, 20, 60, 60), "c": _box(40, 40, 80, 80)},
            camera_id="cam2",
            depth_map=depth2,
            depth_metric=True,
        )
        graph = merge_graphs(g1, g2)
        assert "cam1" in graph.by_camera
        assert "cam2" in graph.by_camera
        # adjacency is union over cameras
        assert set(graph.directly_occludes("a")) == {"b"}
        assert set(graph.directly_occludes("b")) == {"c"}

    def test_duplicate_triples_collapsed(self):
        depth = np.full((100, 100), 5.0, dtype=float)
        depth[0:40, 0:40] = 2.0
        g = compute_object_occlusions(
            objects={"a": _box(0, 0, 40, 40), "b": _box(20, 20, 60, 60)},
            camera_id="cam1",
            depth_map=depth,
            depth_metric=True,
        )
        merged = merge_graphs(g, g)
        assert len(merged.relations) == len(g.relations)


# ------------------------------------------------------------------ occluded regions

class TestOccludedRegions:
    def test_hidden_region_computed(self):
        depth = np.full((100, 100), 5.0, dtype=float)
        depth[10:60, 10:60] = 2.0   # a nearer than b
        graph = compute_object_occlusions(
            objects={"a": _box(0, 0, 60, 60), "b": _box(30, 30, 90, 90)},
            camera_id="cam1",
            depth_map=depth,
            depth_metric=True,
        )
        boxes = {"a": _box(0, 0, 60, 60), "b": _box(30, 30, 90, 90)}
        regions = occluded_region_of(graph, "b", boxes)
        assert len(regions) == 1
        assert regions[0].hidden_by == "a"
        assert regions[0].camera_id == "cam1"
        assert 0.0 < regions[0].hidden_fraction <= 1.0
        hx0, hy0, hx1, hy1 = regions[0].hidden_box_px
        assert hx1 > hx0 and hy1 > hy0


# ------------------------------------------------------------------ object permanence

class TestObjectPermanence:
    def test_visible_object_is_present(self):
        graph = compute_object_occlusions(
            objects={"a": _box(0, 0, 50, 50), "b": _box(60, 60, 100, 100)},
            camera_id="cam1",
        )
        boxes = {"a": _box(0, 0, 50, 50), "b": _box(60, 60, 100, 100)}
        perm = object_permanence_in_view(graph, "a", "cam1", boxes)
        assert perm.present is True
        assert perm.visible is True
        assert perm.occluded_by == ()

    def test_fully_occluded_still_present(self):
        # a fully covers b, and a is nearer -> b present but hidden.
        depth = np.full((100, 100), 10.0, dtype=float)
        depth[0:100, 0:100] = 2.0   # a everywhere closer
        graph = compute_object_occlusions(
            objects={"a": _box(0, 0, 100, 100), "b": _box(20, 20, 80, 80)},
            camera_id="cam1",
            depth_map=depth,
            depth_metric=True,
        )
        boxes = {"a": _box(0, 0, 100, 100), "b": _box(20, 20, 80, 80)}
        perm = object_permanence_in_view(graph, "b", "cam1", boxes)
        assert perm.present is True   # object permanence: hidden but present
        assert perm.visible is False
        assert "a" in perm.occluded_by
        assert perm.confidence > 0.0

    def test_no_evidence_not_present(self):
        graph = compute_object_occlusions(
            objects={"a": _box(0, 0, 50, 50)}, camera_id="cam1",
        )
        boxes = {"a": _box(0, 0, 50, 50)}
        perm = object_permanence_in_view(graph, "missing", "cam1", boxes)
        assert perm.present is False
        assert perm.visible is False
        assert perm.confidence == 0.0
