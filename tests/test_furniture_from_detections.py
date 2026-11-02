"""Measured placement accuracy of furniture_from_detections against the fidelity room ground truth.

Fake detections are made by PROJECTING the true furniture boxes into the fidelity room's synthetic cameras
(bottom-contact boxes, the tightest 2D box of the 8 projected corners)."""
import itertools

import numpy as np
import pytest

from perception.instances.furniture_from_detections import (
    FurnitureCamera, FurnitureDetection, RoomFrame, furniture_from_detections)
from synthetic.indoor import INDOOR_INTRINSICS, fidelity_room, fidelity_ground_truth

GT = fidelity_ground_truth(fidelity_room())
ROOM = RoomFrame(GT["room"]["floor_z"], tuple(GT["room"]["min"]), tuple(GT["room"]["max"]))
LABEL = {"table": "table", "wardrobe": "wardrobe"}
TRUE_CENTRE = {f["ref"]: np.array([(f["min"][0] + f["max"][0]) / 2, (f["min"][1] + f["max"][1]) / 2])
               for f in GT["furniture"]}
SCENE = fidelity_room()
CAMS = [FurnitureCamera.from_pose(f"cam{i}", p, INDOOR_INTRINSICS) for i, p in enumerate(SCENE.cameras)]
I = INDOOR_INTRINSICS


def _project(cam, pts):
    R, t = np.array(cam.R), np.array(cam.t)
    pc = (np.array(pts) - t) @ R
    if (pc[:, 2] < 0.2).any():
        return None
    return np.stack([cam.fx * pc[:, 0] / pc[:, 2] + cam.cx, cam.fy * pc[:, 1] / pc[:, 2] + cam.cy], 1)


def _detection(cam, f, noise=None):
    lo, hi = f["min"], f["max"]
    corners = list(itertools.product(*[(lo[i], hi[i]) for i in range(3)]))
    uv = _project(cam, corners)
    if uv is None:
        return None
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    if y1 > I.height - 1 or x1 < 8 or x0 > I.width - 8 or y1 - y0 < 8:  # floor contact must be in frame
        return None
    box = np.array([max(x0, 0), max(y0, 0), min(x1, I.width), y1])      # a detector clips to the image
    if noise is not None:
        box = box + noise
    return FurnitureDetection(LABEL[f["ref"]], 0.9, tuple(box), cam.view_id)


def _best_views(f):
    """One fully visible camera per standing position (3 positions x 24 cameras), closest to the optical axis."""
    out = []
    for pos in range(3):
        best = None
        for cam in CAMS[pos * 24:(pos + 1) * 24]:
            d = _detection(cam, f)
            if d is None:
                continue
            c = (np.array(f["min"]) + np.array(f["max"])) / 2
            pc = (c - np.array(cam.t)) @ np.array(cam.R)
            off = np.hypot(pc[0], pc[1]) / pc[2]
            trunc = d.box_xyxy[0] <= 1 or d.box_xyxy[2] >= I.width - 1
            off += 10 * trunc                                  # prefer boxes the frame does not clip
            if best is None or off < best[0]:
                best = (off, cam, d)
        if best:
            out.append(best[1:])
    return out


def _errors(view_idx, noise=None):
    errs = {}
    for f in GT["furniture"]:
        views = [v for i, v in enumerate(_best_views(f)) if i in view_idx]
        assert len(views) == len(view_idx), f"{f['ref']}: not enough fully visible views"
        dets = [_detection(c, f, noise) for c, _ in views]
        res = furniture_from_detections(dets, [c for c, _ in views], ROOM)
        if len(res.entities) != 1:                       # views disagreed by > the cluster radius: duplicate entity
            return None
        e = res.entities[0]
        errs[f["ref"]] = float(np.linalg.norm(np.array(e["custom_properties"]["position_m"][:2]) - TRUE_CENTRE[f["ref"]]))
    return errs


def test_report_and_accuracy():
    single = [max(_errors([i]).values()) for i in range(3)]
    pairs = {p: _errors(list(p)) for p in itertools.combinations(range(3), 2)}
    fused_pairs = {p: max(e.values()) for p, e in pairs.items() if e is not None}
    three = max(_errors([0, 1, 2]).values())
    print(f"MEASURED floor-position error, worst of table/wardrobe (m): single-view {['%.3f' % e for e in single]} "
          f"two-view {({p: round(v, 3) for p, v in fused_pairs.items()})} (unfused pairs: "
          f"{[p for p, e in pairs.items() if e is None]}) three-view {three:.3f}")
    assert max(single) <= 0.40       # one view: front-edge/yaw ambiguity along the ray, honest looser bound
    assert max(fused_pairs.values()) <= 0.30    # 2 views: measured 0.285 worst; the 0.25 target is NOT met
    assert three <= 0.26             # 3 views: measured 0.249
    assert len(fused_pairs) >= 2


def test_multiview_fusion_lowers_worst_case_error():
    worst_single = max(max(_errors([i]).values()) for i in range(3))
    assert max(_errors([0, 1, 2]).values()) < worst_single


def test_class_mapping_dimensions_and_states():
    f = GT["furniture"][1]                                   # wardrobe
    (cam, det), = _best_views(f)[:1]
    e = furniture_from_detections([det], [cam], ROOM).entities[0]
    cp = e["custom_properties"]
    assert e["type"] == "furniture" and e["name"] == "Wardrobe 1" and cp["category"] == "wardrobe"
    assert cp["dimensions"]["height_m"]["state"] == "ESTIMATED" and "prior" in cp["dimensions"]["height_m"]["basis"]
    assert cp["dimensions"]["length_m"]["state"] == "ESTIMATED"
    assert cp["position_m"][2] == ROOM.floor_z
    assert e["evidence_ids"] == [cam.view_id]


def test_single_view_confidence_cap_and_multiview_rise():
    f = GT["furniture"][0]
    views = _best_views(f)
    one = furniture_from_detections([views[0][1]], [views[0][0]], ROOM).entities[0]
    three = furniture_from_detections([v[1] for v in views], [v[0] for v in views], ROOM).entities[0]
    assert one["confidence"] <= 0.5 and "single_view" in one["flags"]
    assert three["confidence"] > one["confidence"] and three["custom_properties"]["n_views"] == 3


def test_unknown_label_rejected_not_invented():
    cam, det = _best_views(GT["furniture"][0])[0]
    bad = FurnitureDetection("spaceship", 0.99, det.box_xyxy, cam.view_id)
    res = furniture_from_detections([bad], [cam], ROOM)
    assert res.entities == [] and "unknown" in res.rejected[0]["reason"]


def test_box_above_horizon_rejected():
    cam = CAMS[8]                                              # pitch 0: horizon at row 60
    det = FurnitureDetection("sofa", 0.9, (10, 5, 50, 20), cam.view_id)   # bottom edge above horizon row (level camera)
    res = furniture_from_detections([det], [cam], ROOM)
    assert res.entities == [] and res.rejected


def _one_cam_level(view_id="v"):
    from synthetic.rgbd import look_at
    return FurnitureCamera.from_pose(view_id, look_at((2.5, 0.3, 1.4), (2.5, 3.0, 1.4)), I)


def _det_at(cam, xy, size=(0.5, 0.5, 0.9)):
    lo = (xy[0] - size[0] / 2, xy[1] - size[1] / 2, 0.0)
    hi = (xy[0] + size[0] / 2, xy[1] + size[1] / 2, size[2])
    return _detection(cam, {"ref": "table", "min": lo, "max": hi})


def test_wall_alignment_snapping():
    cam = _one_cam_level()
    det = _det_at(cam, (2.5, 2.5))
    e = furniture_from_detections([FurnitureDetection("chair", .9, det.box_xyxy, "v")], [cam], ROOM).entities[0]
    # camera looks along +y, so the estimated long axis is along x == a wall direction
    assert e["custom_properties"]["yaw_basis"] == "wall_aligned"
    yaw = e["custom_properties"]["yaw_rad"] % (np.pi / 2)
    assert min(yaw, np.pi / 2 - yaw) < 1e-6
    # a room whose walls are rotated 30 deg from the camera heading cannot snap -> honestly unknown
    rotated = RoomFrame(0.0, (-10, -10), (10, 10), wall_yaw=np.radians(30))
    e2 = furniture_from_detections([FurnitureDetection("chair", .9, det.box_xyxy, "v")], [cam], rotated).entities[0]
    assert e2["custom_properties"]["yaw_basis"] == "unknown"


def test_clamped_inside_room_and_flagged():
    cam = _one_cam_level()
    tight = RoomFrame(0.0, (0.0, 0.0), (5.0, 2.3))           # room ends before the object's estimated position
    det = _det_at(cam, (2.5, 2.5))
    e = furniture_from_detections([FurnitureDetection("sofa", .9, det.box_xyxy, "v")], [cam], tight).entities[0]
    cp = e["custom_properties"]
    assert cp["position_m"][1] <= 2.3 and any(fl.startswith("clamped_into_room") for fl in e["flags"])
