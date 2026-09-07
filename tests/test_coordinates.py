import pytest

from world_ir.coordinates import CoordinateRegistry, Frame, Mat4, Transform


def translation(dx: float, dy: float, dz: float) -> Mat4:
    return (
        (1.0, 0.0, 0.0, dx),
        (0.0, 1.0, 0.0, dy),
        (0.0, 0.0, 1.0, dz),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_identity_apply():
    t = Transform.identity(Frame.WORLD)
    assert t.apply((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)


def test_translation_apply():
    t = Transform(Frame.SESSION_LOCAL, Frame.WORLD, matrix=translation(10, 0, 0))
    assert t.apply((1.0, 2.0, 3.0)) == (11.0, 2.0, 3.0)


def test_inverse_round_trips():
    t = Transform(Frame.SESSION_LOCAL, Frame.WORLD, matrix=translation(5, -3, 2))
    inv = t.inverse()
    p = (1.0, 2.0, 3.0)
    assert inv.apply(t.apply(p)) == pytest.approx(p)


def test_then_composes_frames():
    a_to_b = Transform(Frame.CAMERA, Frame.SESSION_LOCAL, matrix=translation(1, 0, 0))
    b_to_c = Transform(Frame.SESSION_LOCAL, Frame.WORLD, matrix=translation(0, 10, 0))
    a_to_c = a_to_b.then(b_to_c)
    assert a_to_c.source_frame == Frame.CAMERA
    assert a_to_c.target_frame == Frame.WORLD
    assert a_to_c.apply((0.0, 0.0, 0.0)) == (1.0, 10.0, 0.0)


def test_then_rejects_frame_mismatch():
    a_to_b = Transform(Frame.CAMERA, Frame.SESSION_LOCAL)
    x_to_y = Transform(Frame.WGS84, Frame.ECEF)
    with pytest.raises(ValueError):
        a_to_b.then(x_to_y)


def test_registry_direct_lookup():
    reg = CoordinateRegistry()
    reg.register(Transform(Frame.SESSION_LOCAL, Frame.WORLD, matrix=translation(1, 1, 1)))
    t = reg.get(Frame.SESSION_LOCAL, Frame.WORLD)
    assert t.apply((0, 0, 0)) == (1, 1, 1)


def test_registry_inverse_lookup():
    reg = CoordinateRegistry()
    reg.register(Transform(Frame.SESSION_LOCAL, Frame.WORLD, matrix=translation(1, 1, 1)))
    t = reg.get(Frame.WORLD, Frame.SESSION_LOCAL)
    assert t.apply((1, 1, 1)) == pytest.approx((0, 0, 0))


def test_registry_chained_lookup():
    reg = CoordinateRegistry()
    reg.register(Transform(Frame.CAMERA, Frame.SESSION_LOCAL, matrix=translation(1, 0, 0)))
    reg.register(Transform(Frame.SESSION_LOCAL, Frame.WORLD, matrix=translation(0, 2, 0)))
    t = reg.get(Frame.CAMERA, Frame.WORLD)
    assert t is not None
    assert t.apply((0, 0, 0)) == pytest.approx((1, 2, 0))


def test_registry_unreachable_frame_returns_none():
    reg = CoordinateRegistry()
    reg.register(Transform(Frame.CAMERA, Frame.SESSION_LOCAL))
    assert reg.get(Frame.CAMERA, Frame.ECEF) is None


def test_registry_same_frame_is_identity():
    reg = CoordinateRegistry()
    t = reg.get(Frame.WORLD, Frame.WORLD)
    assert t.apply((3, 4, 5)) == (3, 4, 5)
