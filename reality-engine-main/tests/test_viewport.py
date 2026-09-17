"""Tests for viewport camera and frustum culling."""

from engine.render import Camera, Viewport
from engine.physics.math3 import Vec3


class TestCamera:
    def test_look_at_forward(self):
        cam = Camera.look_at(Vec3(0, 0, 0), Vec3(10, 0, 0))
        assert abs(cam.forward.x - 1.0) < 1e-9

    def test_right_perpendicular_to_forward(self):
        cam = Camera.look_at(Vec3(0, 0, 0), Vec3(1, 0, 0))
        assert abs(cam.right().dot(cam.forward)) < 1e-9

    def test_point_ahead_visible(self):
        cam = Camera.look_at(Vec3(0, 0, 0), Vec3(1, 0, 0))
        assert cam.is_visible(Vec3(10, 0, 0))

    def test_point_behind_not_visible(self):
        cam = Camera.look_at(Vec3(0, 0, 0), Vec3(1, 0, 0))
        assert not cam.is_visible(Vec3(-10, 0, 0))

    def test_point_outside_fov_not_visible(self):
        cam = Camera(position=Vec3(0, 0, 0), forward=Vec3(1, 0, 0), fov_deg=60.0)
        # 90 degrees off-axis, well outside a 60-degree FOV
        assert not cam.is_visible(Vec3(0, 10, 0))

    def test_point_within_fov_visible(self):
        cam = Camera(position=Vec3(0, 0, 0), forward=Vec3(1, 0, 0), fov_deg=90.0)
        # 30 degrees off-axis, within a 90-degree FOV (45 deg half-angle)
        assert cam.is_visible(Vec3(10, 5, 0))

    def test_beyond_far_plane_not_visible(self):
        cam = Camera(position=Vec3(0, 0, 0), forward=Vec3(1, 0, 0), far=100.0)
        assert not cam.is_visible(Vec3(200, 0, 0))

    def test_before_near_plane_not_visible(self):
        cam = Camera(position=Vec3(0, 0, 0), forward=Vec3(1, 0, 0), near=1.0)
        assert not cam.is_visible(Vec3(0.5, 0, 0))

    def test_radius_pulls_edge_point_into_view(self):
        cam = Camera(position=Vec3(0, 0, 0), forward=Vec3(1, 0, 0), fov_deg=10.0)
        far_off_axis = Vec3(10, 5, 0)  # well outside a narrow 10-degree FOV
        assert not cam.is_visible(far_off_axis, radius=0.0)
        assert cam.is_visible(far_off_axis, radius=50.0)


class TestViewport:
    def test_cull_filters_invisible(self):
        cam = Camera.look_at(Vec3(0, 0, 0), Vec3(1, 0, 0))
        vp = Viewport(cam)

        entities = [
            ("front", Vec3(10, 0, 0), 0.5),
            ("behind", Vec3(-10, 0, 0), 0.5),
        ]
        visible = vp.cull(entities)

        ids = [e["entity_id"] for e in visible]
        assert "front" in ids
        assert "behind" not in ids

    def test_cull_sorts_by_distance(self):
        cam = Camera.look_at(Vec3(0, 0, 0), Vec3(1, 0, 0))
        vp = Viewport(cam)

        entities = [
            ("far", Vec3(20, 0, 0), 0.1),
            ("near", Vec3(5, 0, 0), 0.1),
            ("mid", Vec3(10, 0, 0), 0.1),
        ]
        visible = vp.cull(entities)

        assert [e["entity_id"] for e in visible] == ["near", "mid", "far"]

    def test_set_camera(self):
        cam1 = Camera.look_at(Vec3(0, 0, 0), Vec3(1, 0, 0))
        cam2 = Camera.look_at(Vec3(0, 0, 0), Vec3(-1, 0, 0))
        vp = Viewport(cam1)
        vp.set_camera(cam2)

        assert vp.camera is cam2
