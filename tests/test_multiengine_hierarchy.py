"""Spatial hierarchy: structure rules, transform composition, and the measured float32 precision argument for local frames."""

from __future__ import annotations

import numpy as np
import pytest

from engine.multiengine.alignment import Frame, FrameGraph, Sim3
from engine.multiengine.spatial_hierarchy import HierarchyError, SpatialHierarchy, local_precision_m

I3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _shift(x, y, z, scale=1.0):
    return Sim3(scale, I3, (x, y, z))


def _city():
    g = FrameGraph()
    for f in ("city", "bldg", "storey2", "roomA"):
        g.add_frame(Frame(f, scale_state="metric"))
    h = SpatialHierarchy(g)
    h.add("city", "city", "city")
    h.add("b1", "building", "bldg", parent_id="city", to_parent=_shift(5000.0, 3000.0, 0.0),
          source="survey benchmark (synthetic)")
    h.add("s2", "storey", "storey2", parent_id="b1", to_parent=_shift(0, 0, 6.0), source="floor plan (synthetic)")
    h.add("rA", "room", "roomA", parent_id="s2", to_parent=_shift(4.0, 2.0, 0.0), source="floor plan (synthetic)")
    return h


def test_point_composes_room_to_city_through_each_level():
    h = _city()
    p = h.to_root([2.0, 1.0, 0.0], "rA")
    assert np.allclose(p, [5006.0, 3003.0, 6.0])
    assert [a.node_id for a in h.ancestors("rA")] == ["s2", "b1", "city"]
    assert h.root_of("rA").node_id == "city"
    assert np.allclose(h.convert([2.0, 1.0, 0.0], "rA", "s2"), [6.0, 3.0, 0.0])
    assert [c.node_id for c in h.children("b1")] == ["s2"]


def test_level_order_parent_and_transform_rules():
    g = FrameGraph()
    for f in ("c", "r", "b", "x"):
        g.add_frame(Frame(f, scale_state="metric"))
    h = SpatialHierarchy(g)
    h.add("c", "city", "c")
    h.add("r", "room", "r", parent_id="c", to_parent=_shift(0, 0, 0), source="s")  # skipping levels is allowed
    with pytest.raises(HierarchyError, match="cannot sit inside"):
        h.add("b", "building", "b", parent_id="r", to_parent=_shift(0, 0, 0), source="s")  # inversion is not
    with pytest.raises(HierarchyError, match="needs its transform"):
        h.add("b", "building", "b", parent_id="c")
    with pytest.raises(HierarchyError, match="unknown parent"):
        h.add("b", "building", "b", parent_id="ghost", to_parent=_shift(0, 0, 0), source="s")
    with pytest.raises(HierarchyError, match="unknown level"):
        h.add("x", "galaxy", "x")
    with pytest.raises(HierarchyError, match="already exists"):
        h.add("c", "city", "c")
    with pytest.raises(HierarchyError, match="root node has no parent transform"):
        h.add("x", "city", "x", to_parent=_shift(0, 0, 0))


def test_undeclared_frame_is_refused():
    h = SpatialHierarchy(FrameGraph())
    with pytest.raises(Exception, match="unknown coordinate frame"):
        h.add("c", "city", "nope")


def test_local_frames_keep_precision_that_a_city_origin_frame_loses_in_float32():
    h = _city()
    local = local_precision_m(2.0)  # the chair stored in its room frame
    at_city = local_precision_m(5006.0)  # the same chair stored in the city frame
    assert local < 5e-7 and at_city > 4e-4  # sub-micron vs ~0.5 mm spacing
    # the loss is real, not just a bound: nudge the point by 0.1 mm and it cannot be represented in the city frame
    base = np.float32(h.to_root([2.0, 1.0, 0.0], "rA")[0, 0])
    nudged = np.float32(h.to_root([2.0001, 1.0, 0.0], "rA")[0, 0])
    assert abs(float(nudged) - float(base)) in (0.0, at_city)  # either swallowed, or snapped to a whole spacing step
    assert abs(float(nudged - base) - 1e-4) > 1e-5  # never the true 0.1 mm
    # while the local coordinate survives in float32 to well under a millimetre
    assert abs(float(np.float32(2.0001)) - 2.0001) < 1e-6


def test_readiness_never_claims_city_scale_and_reports_missing_levels():
    r = _city().readiness()
    assert r["city_scale_claimable"] is False and "no real" in r["why"]
    assert r["room_to_building_chain_tested_structurally"] is True
    assert r["levels_missing"] == ["district", "block", "element", "part"]
    assert r["structural_problems"] == []


def test_validate_reports_a_frame_whose_path_to_its_parent_was_removed():
    h = _city()
    h.graph._edges.clear()  # simulate a graph edited behind the hierarchy's back
    problems = h.validate()
    assert len(problems) == 3 and all("no known transform" in p for p in problems)
    assert h.readiness()["structural_problems"] == problems
