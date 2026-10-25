"""Synthetic indoor architecture with ground truth: rooms, corridors, doors, windows, stairs, ramps, storeys, clutter.

SYNTHETIC INDOOR VERIFIED -- REAL INDOOR DATASET VERIFICATION PENDING. These are controlled scenes for the
architectural-perception chain: a regression fixture with known answers, not evidence about real buildings.

Each factory returns an ``IndoorScene``: the quads, the ground truth (``scene.truth``) and default viewpoints.
``observe`` renders those viewpoints through the RGB-D sensor model and back-projects the VALID depth into a
``ReconstructionResult`` -- so occlusion by clutter and walls, glass holes, sensor noise and partial coverage arise
from the rendering instead of from hand-placed point grids. Points keep the id of the camera that saw them
(``source_evidence_ids``), which is what lets the engine tell a floor from a ceiling.

World frame: metres, Z up. ``truth`` keys:

    storeys   [{index, floor_z, ceiling_z}]
    rooms     [{id, storey, min:(x,y), max:(x,y), floor_z, ceiling_z, area_m2, kind: room|corridor}]
    openings  [{kind, wall, storey, center:(x,y,z), width, bottom, top}]
    stairs    [{ref, from_storey, to_storey, min, max, rise}]
    ramps     [{ref, start, end, width, rise}]
    clutter   [{ref, min, max}]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from synthetic.rgbd import Intrinsics, Pose, RgbdFrame, SensorModel, look_at, render_frame
from synthetic.scene import (
    CEILING_COLOR, CLUTTER_COLOR, FLOOR_COLOR, Opening, Scene, add_box, add_ramp, add_slab, add_stairs, add_wall,
)

INDOOR_INTRINSICS = Intrinsics.from_hfov(160, 120, hfov_deg=90.0)
#: Two sensor regimes, because the architecture chain's plane tolerance (2 cm) fixes the noise it can absorb:
#:   CLEAN  -- sigma ~0.5 mm close in, ~1.8 mm at 5 m, no dropout blobs: inside the chain's design envelope.
#:   NOISY  -- sigma ~1 mm close in, ~1.6 cm at 5 m, 2% dropout: a mid-range consumer depth camera. Measured to push
#:             the chain OUTSIDE its envelope (phantom openings, a ramp promoted as floor); kept so that regime is a
#:             pinned, documented characterisation instead of an unknown.
#: Windows return no valid depth in both (glass_invalid_prob=1): the gap in the wall's coverage is what the opening
#: scanner measures. A glass that returns depth part of the time is a different, partially-occluded case.
INDOOR_SENSOR_CLEAN = SensorModel(sigma0_m=0.0005, sigma_z2=0.00005, hole_fraction=0.0, salt_fraction=0.0,
                                  glass_invalid_prob=1.0, seed=7)
INDOOR_SENSOR_NOISY = SensorModel(sigma0_m=0.001, sigma_z2=0.0006, hole_fraction=0.02, salt_fraction=0.001,
                                  glass_invalid_prob=1.0, seed=7)
INDOOR_SENSOR = INDOOR_SENSOR_CLEAN
DOOR = dict(kind="door", bottom=0.0, top=2.05)
WINDOW = dict(kind="window", bottom=0.9, top=2.0)


@dataclass
class IndoorScene:
    scene: Scene
    cameras: List[Pose]
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class Observation:
    result: object                       # reconstruction.backend.interface.ReconstructionResult
    frames: List[RgbdFrame]


def panorama(eye: Sequence[float], *, yaws: int = 8, pitches_deg: Sequence[float] = (-35.0, 0.0, 35.0)) -> List[Pose]:
    """Viewpoints from one standing position: ``yaws`` headings x ``pitches_deg`` tilts (down sees the floor, up the
    ceiling)."""
    out = []
    for p in pitches_deg:
        for k in range(yaws):
            a = 2 * np.pi * k / yaws
            tgt = (eye[0] + np.cos(a) * np.cos(np.radians(p)), eye[1] + np.sin(a) * np.cos(np.radians(p)),
                   eye[2] + np.sin(np.radians(p)))
            out.append(look_at(eye, tgt))
    return out


# ----------------------------------------------------------------------------------------------- building blocks

def _room(scene: Scene, rid: str, x0, x1, y0, y1, z0, h, *, storey: int = 0, kind: str = "room",
          openings: Optional[Dict[str, List[Opening]]] = None, skip_walls: Sequence[str] = (),
          floor_hole=None, ceiling_hole=None) -> None:
    """Floor, ceiling and four walls (sides: south y0, east x1, north y1, west x0). ``skip_walls`` omits a wall
    another room already built (a shared wall is built once). Records the room and its openings in the truth."""
    openings = openings or {}
    add_slab(scene, x0, x1, y0, y1, z0, label="floor", color=FLOOR_COLOR, ref=f"{rid}-floor", hole=floor_hole)
    add_slab(scene, x0, x1, y0, y1, z0 + h, label="ceiling", color=CEILING_COLOR, ref=f"{rid}-ceiling",
             hole=ceiling_hole)
    sides = {"south": ((x0, y0), (x1, y0)), "east": ((x1, y0), (x1, y1)),
             "north": ((x1, y1), (x0, y1)), "west": ((x0, y1), (x0, y0))}
    for side, (p0, p1) in sides.items():
        if side in skip_walls:
            continue
        ops = openings.get(side, [])
        add_wall(scene, p0, p1, z0, z0 + h, ref=f"{rid}-wall-{side}", openings=ops)
        for op in ops:
            p0a, p1a = np.array(p0), np.array(p1)
            d = (p1a - p0a) / np.linalg.norm(p1a - p0a)
            c = p0a + d * (op.offset + op.width / 2.0)
            scene.truth["openings"].append({
                "kind": op.kind, "wall": f"{rid}-wall-{side}", "storey": storey,
                "center": (float(c[0]), float(c[1]), z0 + (op.bottom + op.top) / 2.0), "width": op.width,
                "bottom": z0 + op.bottom, "top": z0 + op.top})
    scene.truth["rooms"].append({
        "id": rid, "storey": storey, "min": (x0, y0), "max": (x1, y1), "floor_z": z0, "ceiling_z": z0 + h,
        "area_m2": (x1 - x0) * (y1 - y0), "kind": kind})


def _storey(scene: Scene, index: int, floor_z: float, ceiling_z: float) -> None:
    scene.truth["storeys"].append({"index": index, "floor_z": floor_z, "ceiling_z": ceiling_z})


def _clutter(scene: Scene, ref: str, lo, hi) -> None:
    add_box(scene, lo, hi, label="clutter", color=CLUTTER_COLOR, ref=ref)
    scene.truth["clutter"].append({"ref": ref, "min": tuple(lo), "max": tuple(hi)})


# ----------------------------------------------------------------------------------------------- scene factories

def single_room() -> IndoorScene:
    """5 x 4 x 2.6 m: one door (south), one window (north), a table and a pillar that occlude parts of the room."""
    s = Scene()
    _storey(s, 0, 0.0, 2.6)
    _room(s, "room-a", 0, 5, 0, 4, 0.0, 2.6, openings={
        "south": [Opening(offset=1.0, width=0.9, **DOOR)],
        "north": [Opening(offset=1.5, width=1.2, **WINDOW)]})
    _clutter(s, "table", (2.0, 1.2, 0.0), (3.2, 2.0, 0.75))
    _clutter(s, "pillar", (3.8, 2.6, 0.0), (4.1, 2.9, 2.6))
    cams = panorama((1.3, 1.0, 1.4)) + panorama((3.7, 1.0, 1.4)) + panorama((2.5, 3.1, 1.4))
    return IndoorScene(s, cams)


def corridor_with_rooms() -> IndoorScene:
    """An 8 x 1.6 m corridor with two 4 x 3.4 m rooms on its north side, each entered by a door; the rooms are also
    joined to each other by a door in their shared wall."""
    s = Scene()
    _storey(s, 0, 0.0, 2.6)
    _room(s, "corridor", 0, 8, 0, 1.6, 0.0, 2.6, kind="corridor", openings={
        "north": [Opening(offset=1.5, width=0.9, **DOOR), Opening(offset=5.5, width=0.9, **DOOR)]})
    _room(s, "room-a", 0, 4, 1.6, 5.0, 0.0, 2.6, skip_walls=("south",), openings={
        "east": [Opening(offset=0.8, width=0.9, **DOOR)], "north": [Opening(offset=1.2, width=1.2, **WINDOW)]})
    _room(s, "room-b", 4, 8, 1.6, 5.0, 0.0, 2.6, skip_walls=("south", "west"), openings={
        "north": [Opening(offset=1.2, width=1.2, **WINDOW)]})
    # the door in the A|B wall is part of room A's east wall above; the corridor's north wall holds the two doors
    cams = (panorama((1.9, 0.8, 1.4), yaws=6) + panorama((6.1, 0.8, 1.4), yaws=6) +
            panorama((2.0, 3.3, 1.4)) + panorama((6.0, 3.3, 1.4)))
    return IndoorScene(s, cams)


def two_storey_with_stairs() -> IndoorScene:
    """Two 6 x 6 m storeys (floors at 0 and 2.9 m, a 0.3 m slab between) joined by a straight flight in the lower
    room that climbs through a stairwell hole in both slabs."""
    s = Scene()
    rise, h0, slab = 2.9, 2.6, 0.3
    hole = (4.2, 5.2, 3.0, 5.9)
    _storey(s, 0, 0.0, h0)
    _storey(s, 1, rise, rise + 2.6)
    _room(s, "ground", 0, 6, 0, 6, 0.0, h0, storey=0, ceiling_hole=hole,
          openings={"south": [Opening(offset=1.0, width=0.9, **DOOR)]})
    _room(s, "upper", 0, 6, 0, 6, rise, 2.6, storey=1, floor_hole=hole,
          openings={"south": [Opening(offset=3.0, width=1.2, **WINDOW)]})
    add_stairs(s, (4.7, 0.8, 0.0), (0.0, 1.0), width=1.0, run=4.9, rise=rise, steps=16, ref="stair-1")
    s.truth["stairs"].append({"ref": "stair-1", "from_storey": 0, "to_storey": 1, "min": (4.2, 0.8, 0.0),
                              "max": (5.2, 5.7, rise), "rise": rise})
    cams = panorama((1.5, 1.5, 1.4)) + panorama((1.5, 4.5, 1.4)) + panorama((1.5, 1.5, rise + 1.4)) + \
        panorama((1.5, 4.5, rise + 1.4)) + panorama((3.0, 3.0, rise + 1.4))
    return IndoorScene(s, cams)


def ramp_room() -> IndoorScene:
    """A 6 x 4 m room with a 3 m ramp climbing 0.6 m. The ramp is a tilted surface: it must not be mistaken for a
    floor or ceiling, and must not create a second storey."""
    s = Scene()
    _storey(s, 0, 0.0, 2.6)
    _room(s, "room-r", 0, 6, 0, 4, 0.0, 2.6, openings={"south": [Opening(offset=2.0, width=0.9, **DOOR)]})
    add_ramp(s, (2.5, 2.0, 0.0), (5.5, 2.0, 0.6), width=1.4, ref="ramp-1")
    s.truth["ramps"].append({"ref": "ramp-1", "start": (2.5, 2.0, 0.0), "end": (5.5, 2.0, 0.6), "width": 1.4,
                             "rise": 0.6})
    cams = panorama((1.0, 1.0, 1.4)) + panorama((1.0, 3.0, 1.4)) + panorama((3.0, 0.8, 1.4))
    return IndoorScene(s, cams)


def tower(heights: Sequence[float] = (2.4, 2.4, 2.4), gap: float = 0.3, plan=(0.0, 4.0, 0.0, 4.0)) -> IndoorScene:
    """Stacked closed storeys with spatially disconnected slabs (a 'gap' of structure between them)."""
    s = Scene()
    x0, x1, y0, y1 = plan
    z, cams = 0.0, []
    for i, h in enumerate(heights):
        _storey(s, i, z, z + h)
        _room(s, f"level-{i}", x0, x1, y0, y1, z, h, storey=i)
        cams += panorama(((x0 + x1) / 2 - 0.6, (y0 + y1) / 2 - 0.5, z + h / 2)) + \
            panorama(((x0 + x1) / 2 + 0.6, (y0 + y1) / 2 + 0.5, z + h / 2))
        z += h + gap
    return IndoorScene(s, cams)


# ----------------------------------------------------------------------------------------------- observation

def voxel_downsample(points: np.ndarray, cell: float) -> np.ndarray:
    """One point per ``cell`` voxel (the first, in input order): deterministic, as a dense cloud is thinned."""
    keys = np.floor(points / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def observe(ind: IndoorScene, *, sensor: SensorModel = INDOOR_SENSOR, intr: Intrinsics = INDOOR_INTRINSICS,
            stride: int = 2, voxel_m: float = 0.04, cameras: Optional[Sequence[Pose]] = None) -> Observation:
    """Render ``cameras`` (default: the scene's viewpoints) and back-project the valid depth. The camera that saw
    each point is recorded; cameras whose view saw nothing valid are still registered poses."""
    from provenance import Uncertainty
    from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult

    cams = list(cameras if cameras is not None else ind.cameras)
    frames = [render_frame(ind.scene, intr, pose, sensor, i) for i, pose in enumerate(cams)]
    poses, pts = [], []
    for f in frames:
        eid = f"ev-{f.index:03d}"
        poses.append(ReconstructedCameraPose(evidence_id=eid, position=f.pose.t, rotation=f.pose.quat_wxyz(),
                                             uncertainty=Uncertainty(confidence=0.9)))
        rr, cc = np.nonzero(f.valid[::stride, ::stride])
        rr, cc = rr * stride, cc * stride
        z = f.depth_measured[rr, cc].astype(float)
        cam_pts = np.stack([(cc + 0.5 - intr.cx) / intr.fx * z, (rr + 0.5 - intr.cy) / intr.fy * z, z], axis=1)
        world = cam_pts @ f.pose.matrix.T + np.asarray(f.pose.t)
        keep = voxel_downsample(world, voxel_m) if len(world) else world
        for j, p in enumerate(keep):
            pts.append((eid, p))
    # one global voxel pass so overlapping views do not multiply points; first observer is kept
    if pts:
        arr = np.array([p for _, p in pts])
        keys = np.floor(arr / voxel_m).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        pts = [pts[i] for i in np.sort(first)]
    points = [ReconstructedPoint(position=tuple(float(v) for v in p), track_id=f"pt-{i:06d}",
                                 source_evidence_ids=[eid], uncertainty=Uncertainty(confidence=0.9))
              for i, (eid, p) in enumerate(pts)]
    return Observation(ReconstructionResult(points=points, camera_poses=poses, registration_status="success"), frames)
