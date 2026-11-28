def camera_should_see_point(cam, point):
    """True if `point` is in front of the camera AND within the image plane."""
    forward = _cam_forward(cam)
    to_point = np.asarray(point, dtype=float) - np.asarray(cam.position, dtype=float)
    if float(np.dot(forward, to_point)) <= 1e-6:
        return False  # behind camera
    _, depth = _project_to_image(cam, point)
    if depth is None:
        return False
    fx, fy, cx, cy = cam.intrinsics_fxfycx()
    h = int(round(2.0 * fy * np.tan(np.radians(50.0) * 0.5)))
    w = int(round(2.0 * fx * np.tan(np.radians(50.0) * 0.5)))
    x, y = depth
    return (0 <= x < w) and (0 <= y < h)
