import pytest

from engine.physics.materials import CANONICAL_MATERIALS, PhysicsMaterial


def test_canonical_materials_present():
    for name in ("concrete", "steel", "wood", "glass", "rubber", "ice"):
        assert name in CANONICAL_MATERIALS
        assert CANONICAL_MATERIALS[name].name == name


def test_rejects_nonpositive_density():
    with pytest.raises(ValueError):
        PhysicsMaterial("bad", density=0, friction_static=0.5, friction_dynamic=0.4, restitution=0.5)


def test_rejects_out_of_range_restitution():
    with pytest.raises(ValueError):
        PhysicsMaterial("bad", density=100, friction_static=0.5, friction_dynamic=0.4, restitution=1.5)


def test_rejects_dynamic_friction_exceeding_static():
    with pytest.raises(ValueError):
        PhysicsMaterial("bad", density=100, friction_static=0.3, friction_dynamic=0.9, restitution=0.5)


def test_roundtrip_dict():
    m = CANONICAL_MATERIALS["rubber"]
    assert PhysicsMaterial.from_dict(m.to_dict()) == m
