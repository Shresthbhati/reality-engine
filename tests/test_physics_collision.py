import pytest

from engine.physics.collision.broadphase import body_pairs
from engine.physics.collision.narrowphase import box_vs_box, box_vs_plane, sphere_vs_plane, sphere_vs_sphere
from engine.physics.collision.raycast import ray_vs_box, ray_vs_plane, ray_vs_sphere
from engine.physics.collision.shapes import Box, Plane, Sphere
from engine.physics.math3 import Vec3


def test_sphere_vs_sphere_overlap():
    a = Sphere(1.0)
    b = Sphere(1.0)
    contact = sphere_vs_sphere(Vec3(0, 0, 0), a, Vec3(1.5, 0, 0), b)
    assert contact is not None
    assert contact.penetration == pytest.approx(0.5)
    assert contact.normal == Vec3(1, 0, 0)


def test_sphere_vs_sphere_no_overlap():
    a = Sphere(1.0)
    b = Sphere(1.0)
    assert sphere_vs_sphere(Vec3(0, 0, 0), a, Vec3(5, 0, 0), b) is None


def test_sphere_vs_plane():
    ground = Plane(normal=Vec3(0, 1, 0), distance=0.0)
    sphere = Sphere(1.0)
    contact = sphere_vs_plane(Vec3(0, 0.5, 0), sphere, ground)
    assert contact is not None
    assert contact.penetration == pytest.approx(0.5)


def test_box_vs_plane():
    ground = Plane(normal=Vec3(0, 1, 0), distance=0.0)
    box = Box(Vec3(0.5, 0.5, 0.5))
    contact = box_vs_plane(Vec3(0, 0.3, 0), box, ground)
    assert contact is not None
    assert contact.penetration == pytest.approx(0.2)


def test_box_vs_plane_no_overlap():
    ground = Plane(normal=Vec3(0, 1, 0), distance=0.0)
    box = Box(Vec3(0.5, 0.5, 0.5))
    assert box_vs_plane(Vec3(0, 5.0, 0), box, ground) is None


def test_box_vs_box_minimum_translation_axis():
    a = Box(Vec3(1, 1, 1))
    b = Box(Vec3(1, 1, 1))
    # Overlap of 0.5 along x, 2.0 along y/z -> x is the separating axis.
    contact = box_vs_box(Vec3(0, 0, 0), a, Vec3(1.5, 0, 0), b)
    assert contact is not None
    assert contact.penetration == pytest.approx(0.5)
    assert contact.normal == Vec3(1, 0, 0)


def test_broadphase_pairs_are_sorted_and_deterministic():
    shapes = {"b": Sphere(1.0), "a": Sphere(1.0), "c": Sphere(1.0)}
    positions = {"a": Vec3(0, 0, 0), "b": Vec3(0.5, 0, 0), "c": Vec3(100, 0, 0)}
    pairs = body_pairs(list(shapes.keys()), positions, shapes)
    assert pairs == [("a", "b")]


def test_ray_vs_sphere_hit_and_miss():
    sphere = Sphere(1.0)
    t = ray_vs_sphere(Vec3(-5, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 0), sphere)
    assert t == pytest.approx(4.0)
    assert ray_vs_sphere(Vec3(-5, 5, 0), Vec3(1, 0, 0), Vec3(0, 0, 0), sphere) is None


def test_ray_vs_box_hit():
    box = Box(Vec3(1, 1, 1))
    t = ray_vs_box(Vec3(-5, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 0), box)
    assert t == pytest.approx(4.0)


def test_ray_vs_plane_hit():
    ground = Plane(normal=Vec3(0, 1, 0), distance=0.0)
    t = ray_vs_plane(Vec3(0, 5, 0), Vec3(0, -1, 0), ground)
    assert t == pytest.approx(5.0)


def test_ray_vs_plane_parallel_misses():
    ground = Plane(normal=Vec3(0, 1, 0), distance=0.0)
    assert ray_vs_plane(Vec3(0, 5, 0), Vec3(1, 0, 0), ground) is None
