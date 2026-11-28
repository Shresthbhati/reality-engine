from perception.quality.assessment import Camera, project_world_to_frame, on_screen, gsd_mm_per_px_score
from perception.quality.assessment import _compute_center_distance, stale_center, _apply_stale_on_bounds

R = (0, 0, 0, 1)  # identity quat: camera looks down world +Z (camera axis = world +Z)
c = Camera(quat=R, pos=(0, 0, 1.0), fx=2000, fy=2000, img_w=2000, img_h=2000, depth=2.0)
lines = []
def show(label, obj):
    lines.append(f"{label}: {obj}")

# World point on +Z ahead of camera on axis
p_fwd = (0.0, 0.0, 1.5)
f_fwd = project_world_to_frame(c, p_fwd)
show("WORLD_fwd", p_fwd)
show("FRAME_fwd", f_fwd)
show("ON_SCREEN_fwd", on_screen(p_fwd, f_fwd, c))
show("GSD_fwd", gsd_mm_per_px_score(c, p_fwd, 1.0, 50))

# A point on +X at same depth (within horizontal FOV for these params)
p_right = (0.5, 0.0, 1.5)
f_right = project_world_to_frame(c, p_right)
show("WORLD_right", p_right)
show("FRAME_right", f_right)
show("ON_SCREEN_right", on_screen(p_right, f_right, c))
show("GSD_right", gsd_mm_per_px_score(c, p_right, 1.0, 50))

# A point to the "left" along +X (camera is at 0,0,1 looking +Z; +X is to its right; left is -X)
p_left = (-0.5, 0.0, 1.5)
f_left = project_world_to_frame(c, p_left)
show("WORLD_left", p_left)
show("FRAME_left", f_left)
show("ON_SCREEN_left", on_screen(p_left, f_left, c))
show("GSD_left", gsd_mm_per_px_score(c, p_left, 1.0, 50))

# Point behind camera (should be unprojectable)
p_behind = (0.0, 0.0, 0.5)
f_behind = project_world_to_frame(c, p_behind)
show("WORLD_behind", p_behind)
show("FRAME_behind", f_behind)
show("ON_SCREEN_behind", on_screen(p_behind, f_behind, c))
show("GSD_behind", gsd_mm_per_px_score(c, p_behind, 1.0, 50))

# Stale center helpers
c1 = Camera(quat=(0.985, 0.174, 0.0, 0.0), pos=(0.5, 0.0, 0.985))
d = _compute_center_distance(c1, (0.0, 0.0, 0.9))
show("center_dist_c1", d)
show("stale_c1", stale_center(c1, (0.0, 0.0, 0.9), "2000/01/01T00:00:00"))

out = "\n".join(lines)
with open("_diag2_out.txt", "w", encoding="utf-8") as fh:
    fh.write(out)
print(out)
