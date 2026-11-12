"""P111: architectural constraint checks surfaced by the ``compile_validate`` stage."""
from __future__ import annotations

from pathlib import Path

from engine.pipeline.stages.base import StageContext
from engine.pipeline.stages.registry import SegmentReport
from engine.pipeline.stages.s11_compile_validate import CompileValidate
from provenance import Provenance
from world_ir import WorldIR
from world_ir.constraints import (
    ConstraintConfig,
    ConstraintViolation,
    check_architectural_constraints,
    violations_to_segment_fields,
)
from world_ir.schema_v1 import (
    Entity, EntityType, Geometry, GeometryType, Relationship, RelationshipKind, Vector3,
)


def _box(w: WorldIR, gid: str, lo, hi) -> str:
    w.geometries[gid] = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(*lo), bounds_max=Vector3(*hi))
    return gid


def _ent(w: WorldIR, eid: str, etype: EntityType, lo=None, hi=None, rels=(), **props) -> Entity:
    gids = [_box(w, f"g_{eid}", lo, hi)] if lo is not None else []
    e = Entity(id=eid, name=eid, type=etype, provenance=Provenance.OBSERVED, geometry_ids=gids,
               relationships=list(rels), custom_properties=dict(props))
    w.entities[eid] = e
    return e


def _contains(target: str) -> Relationship:
    return Relationship(kind=RelationshipKind.CONTAINS, target_id=target)


def _codes(vs) -> list[str]:
    return [v.code for v in vs]


def _wall_world(opening_lo=(1, -0.1, 0), opening_hi=(2, 0.1, 2), host="wall") -> WorldIR:
    w = WorldIR(id="w", name="w")
    _ent(w, "wall", EntityType.WALL, (0, -0.1, 0), (4, 0.1, 2.5))
    _ent(w, "door", EntityType.DOOR, opening_lo, opening_hi, host_wall_id=host)
    return w


# (a) opening without a host wall
def test_opening_with_host_wall_passes():
    assert "opening_without_host_wall" not in _codes(check_architectural_constraints(_wall_world()))


def test_opening_without_host_wall_is_error():
    vs = check_architectural_constraints(_wall_world(host="ghost"))
    v = next(v for v in vs if v.code == "opening_without_host_wall")
    assert v.severity == "error" and v.entity_ids == ("door",)


def test_opening_hosted_by_non_wall_is_error():
    w = _wall_world()
    _ent(w, "slab", EntityType.FLOOR, (0, 0, -0.1), (4, 4, 0))
    w.entities["door"].custom_properties["host_wall_id"] = "slab"
    assert "opening_without_host_wall" in _codes(check_architectural_constraints(w))


# (b) opening outside host wall extent
def test_opening_inside_wall_extent_passes():
    assert "opening_outside_host_wall" not in _codes(check_architectural_constraints(_wall_world()))


def test_opening_outside_wall_extent_is_error():
    vs = check_architectural_constraints(_wall_world((3.5, -0.1, 0), (5.0, 0.1, 2)))
    v = next(v for v in vs if v.code == "opening_outside_host_wall")
    assert v.severity == "error" and v.entity_ids == ("door", "wall")


# (c) wall base vs storey floor elevation
def _storey_world(wall_base: float) -> WorldIR:
    w = WorldIR(id="w", name="w")
    _ent(w, "wall", EntityType.WALL, (0, -0.1, wall_base), (4, 0.1, wall_base + 2.5))
    _ent(w, "storey-1", EntityType.STOREY, rels=[_contains("wall")], floor_height_m=3.0)
    return w


def test_wall_base_at_storey_floor_passes():
    assert "wall_base_not_at_storey_floor" not in _codes(check_architectural_constraints(_storey_world(3.02)))


def test_wall_base_off_storey_floor_is_error():
    vs = check_architectural_constraints(_storey_world(3.5))
    v = next(v for v in vs if v.code == "wall_base_not_at_storey_floor")
    assert v.severity == "error" and v.entity_ids == ("wall", "storey-1")


def test_floor_tolerance_comes_from_config():
    w = _storey_world(3.5)
    loose = check_architectural_constraints(w, ConstraintConfig(floor_tolerance_m=1.0))
    assert "wall_base_not_at_storey_floor" not in _codes(loose)


# (d) stair connects two storeys
def _stair_world(linked: tuple[str, ...]) -> WorldIR:
    w = WorldIR(id="w", name="w")
    _ent(w, "storey-1", EntityType.STOREY, floor_height_m=0.0)
    _ent(w, "storey-2", EntityType.STOREY, floor_height_m=3.0)
    rels = [Relationship(kind=RelationshipKind.ADJACENT_TO, target_id=s) for s in linked]
    _ent(w, "stairs-001", EntityType.STAIRS, (0, 0, 0), (1, 3, 3), rels=rels)
    return w


def test_stair_connecting_two_storeys_passes():
    assert "stair_not_connecting_two_storeys" not in _codes(
        check_architectural_constraints(_stair_world(("storey-1", "storey-2"))))


def test_stair_linked_to_one_storey_is_error():
    vs = check_architectural_constraints(_stair_world(("storey-1",)))
    v = next(v for v in vs if v.code == "stair_not_connecting_two_storeys")
    assert v.severity == "error" and v.entity_ids == ("stairs-001",)


def test_stair_in_single_storey_world_is_warning_only():
    w = WorldIR(id="w", name="w")
    _ent(w, "storey-1", EntityType.STOREY, floor_height_m=0.0)
    _ent(w, "stairs-001", EntityType.STAIRS, (0, 0, 0), (1, 3, 1))
    v = next(v for v in check_architectural_constraints(w) if v.code == "stair_not_connecting_two_storeys")
    assert v.severity == "warning"


# stage wiring: error -> degraded, warning-only -> succeeded
def _run_stage(violations: list[ConstraintViolation]):
    def seg(ctx) -> SegmentReport:
        warnings, metrics = violations_to_segment_fields(violations)
        return SegmentReport(warnings=warnings, metrics=metrics)

    ctx = StageContext(job_id="j", workdir=Path("."), artifacts={}, params={}, state={"segments": {"compile_validate": seg}})
    return CompileValidate().run(ctx)


def test_error_violation_degrades_stage():
    res = _run_stage([ConstraintViolation("opening_without_host_wall", ("door",), "no host", "error")])
    assert res.status == "degraded"
    assert any("opening_without_host_wall" in w for w in res.warnings)
    assert res.metrics["constraint_errors"] == 1.0


def test_warning_only_violation_keeps_stage_succeeded():
    res = _run_stage([ConstraintViolation("stair_not_connecting_two_storeys", ("s",), "alone", "warning")])
    assert res.status == "succeeded"
    assert res.warnings == [] and res.metrics["constraint_warnings"] == 1.0 and res.metrics["constraint_errors"] == 0.0
