"""Task 5.1: contract hierarchy in WorldIR (additive, optional; old worlds load unchanged)."""
import json
from pathlib import Path

import pytest

from world_ir import WorldIR
from world_ir.v2_extensions import Hierarchy, HierarchyNode

FIXTURE = Path(__file__).parent / "fixtures" / "worldir_v1_pre_hierarchy.json"


def _hier() -> Hierarchy:
    h = Hierarchy()
    h.add(HierarchyNode("b0", "building", evidence_state="INFERRED"))
    h.add(HierarchyNode("f0", "floor", "b0", entity_id="storey-0", evidence_state="OBSERVED"))
    h.add(HierarchyNode("s-a", "space", "f0", entity_id="room-a", kind="room", evidence_state="OBSERVED", confidence=0.8))
    h.add(HierarchyNode("sf-1", "surface", "s-a", entity_id="wall-1", kind="wall", evidence_state="OBSERVED", coverage=0.6))
    h.add(HierarchyNode("op-1", "opening", "sf-1", entity_id="door-1", kind="door", evidence_state="OBSERVED"))
    h.entity_space["wall-1"] = "s-a"
    return h


def test_old_fixture_loads_unchanged():
    raw = FIXTURE.read_text()
    w = WorldIR.from_json(raw)
    assert not w.hierarchy and w.hierarchy.nodes == {}
    assert w.validate() == []
    assert "hierarchy" not in w.to_dict()
    assert json.loads(w.to_json()) == json.loads(raw)  # byte-for-byte semantic identity


def test_roundtrip_with_hierarchy():
    w = WorldIR.from_json(FIXTURE.read_text())
    w.hierarchy = _hier()
    assert w.validate() == []
    w2 = WorldIR.from_json(w.to_json())
    assert w2.hierarchy.nodes == w.hierarchy.nodes
    assert w2.hierarchy.entity_space == {"wall-1": "s-a"}
    assert w2.to_json() == w.to_json()
    assert [n.id for n in w2.hierarchy.children("f0")] == ["s-a"]
    assert w2.hierarchy.nodes["f0"].confidence is None  # no invented values


def test_problems_are_reported_not_repaired():
    w = WorldIR.from_json(FIXTURE.read_text())
    h = w.hierarchy
    h.add(HierarchyNode("f1", "floor", "missing"))
    h.add(HierarchyNode("s1", "space", "f1", entity_id="nope"))
    h.add(HierarchyNode("op", "opening", "s1"))  # opening must hang off a surface
    msgs = w.validate()
    assert any("missing building parent" in m for m in msgs)
    assert any("unknown entity 'nope'" in m for m in msgs)
    assert any("opening parent must be a surface" in m for m in msgs)
    with pytest.raises(ValueError):
        h.add(HierarchyNode("f1", "floor"))
