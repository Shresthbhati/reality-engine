"""Interior semantics layered on the promoted structure: true footprint, ceiling layers, furniture, colour.

The structural chain (planes -> walls/floors/ceilings -> rooms) reports a room as an axis-aligned box whose height
is the LOWEST ceiling. That is wrong for any room that is not a rectangle with one flat ceiling. This pass, run on
the already-promoted world, adds what the boxes cannot say -- every number is measured from the reconstructed points:

* ``footprint_polygon_m`` / ``footprint_area_m2``: the room's real plan outline (any shape: L, U, notched) from the
  occupancy of the FLOOR-BAND points of the whole cloud (floor height = the dominant low height peak, not a plane average that furniture tops can pull up). The original box area is kept as ``bbox_floor_area_m2``. Windows/doors are scanned per wall edge with the existing opening detector (valid only where SfM covers the wall; a blank wall also reads as a gap).
* ceiling layers: ceilings at different heights inside ONE room are layers of that room, not extra rooms. The lower
  layer(s) that cover only part of the plan while a higher layer is visible elsewhere are labelled ``false_ceiling``
  (dropped/stepped ceiling); the highest is ``structural_ceiling``. A single ceiling stays plain ``ceiling``.
  A false ceiling that hides the whole slab cannot be told apart from a real ceiling -- it is left as ``ceiling``.
* furniture: non-structural points inside the room's shell are clustered into object entities (type UNKNOWN -- the
  ontology has no furniture categories -- labelled ``furniture_candidate``, with an oriented plan box and height).
* ``mean_color_rgb`` per entity from the observed point colours (flat colour, not a texture).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from provenance import Provenance
from world_ir import Entity, EntityType, Geometry, GeometryType, Observation, Relationship, RelationshipKind, Vector3

#: ceilings closer in height than this are one layer
LAYER_MERGE_M = 0.05
#: a height difference at least this big inside one room is a real step
CEILING_STEP_M = 0.10
#: voxel edge for furniture clustering
CLUSTER_VOXEL_M = 0.12
MIN_CLUSTER_POINTS = 25
#: An opening must be bounded by wall points at this density (points per metre of its perimeter). PROVISIONAL: measured on
#: the synthetic L-room -- the real window had ~5/m, spurious gaps 1.6-2.8/m; re-measure on real photographs.
MIN_OPENING_RING_PER_M = 4.0
#: physical sanity (not fitted): a floor-reaching gap shorter than this is not a doorway
MIN_DOOR_HEIGHT_M = 1.8
#: furniture is only looked for this far from the footprint edge (wall bands, window reveals, skirting) and above this
#: height (floor-band noise)
FURNITURE_WALL_CLEARANCE_M = 0.25
FURNITURE_MIN_HEIGHT_M = 0.10


def _basis(up) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(up, float)
    u = u / np.linalg.norm(u)
    ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1), u


def _plan(points, e1, e2, u) -> np.ndarray:
    p = np.asarray(points, float).reshape(-1, 3)
    return np.stack([p @ e1, p @ e2, p @ u], axis=1)


def _cell_size(ab: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    sub = ab[:: max(1, len(ab) // 4000)]
    if len(sub) < 3:
        return 0.1
    d, _ = cKDTree(sub).query(sub, k=2)
    # spacing of the SUBSAMPLE overstates the true spacing by sqrt(stride); scale back
    stride = max(1, len(ab) // 4000)
    return float(np.clip(2.5 * np.median(d[:, 1]) / math.sqrt(stride), 0.05, 0.25))


class _Raster:
    """Plan-view occupancy grid shared by the footprint, the ceiling coverage and the point-in-room test."""

    def __init__(self, ab: np.ndarray, cell: float):
        self.cell = cell
        self.amin, self.bmin = ab.min(axis=0) - 3 * cell
        amax, bmax = ab.max(axis=0) + 3 * cell
        self.w = int(math.ceil((amax - self.amin) / cell)) + 1
        self.h = int(math.ceil((bmax - self.bmin) / cell)) + 1

    def idx(self, ab: np.ndarray):
        ia = np.floor((ab[:, 0] - self.amin) / self.cell).astype(int)
        ib = np.floor((ab[:, 1] - self.bmin) / self.cell).astype(int)
        ok = (ia >= 0) & (ia < self.w) & (ib >= 0) & (ib < self.h)
        return ia, ib, ok

    def occupancy(self, ab: np.ndarray, close_m: float = 0.25) -> np.ndarray:
        import cv2

        img = np.zeros((self.h, self.w), np.uint8)
        ia, ib, ok = self.idx(ab)
        img[ib[ok], ia[ok]] = 255
        k = max(3, int(round(close_m / self.cell)) | 1)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    def to_metres(self, xy: np.ndarray) -> np.ndarray:
        return np.stack([self.amin + (xy[:, 0] + 0.5) * self.cell, self.bmin + (xy[:, 1] + 0.5) * self.cell], axis=1)


def _footprint(raster: _Raster, ab: np.ndarray):
    """(polygon_m [N,2], filled mask) from the occupancy of floor + wall points; None when nothing encloses."""
    import cv2

    occ = raster.occupancy(ab)
    contours, _ = cv2.findContours(occ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, max(1.5, 0.10 / raster.cell), True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    mask = np.zeros_like(occ)
    cv2.fillPoly(mask, [approx.astype(np.int32)], 255)
    return raster.to_metres(approx.astype(float)), mask


def _mask_polygon(raster: "_Raster", mask: np.ndarray) -> Optional[List[List[float]]]:
    """Largest outer contour of a boolean/uint8 plan mask as a simplified polygon in plan metres."""
    import cv2

    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, max(1.5, 0.10 / raster.cell), True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return [[round(float(a), 4), round(float(b), 4)] for a, b in raster.to_metres(approx.astype(float))]


def _shoelace(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _mean_color(tree, colors: np.ndarray, pts: Sequence) -> Optional[List[int]]:
    if tree is None or not len(pts):
        return None
    sub = np.asarray(pts, float)[:: max(1, len(pts) // 1500)]
    _, i = tree.query(sub)
    c = colors[i]
    c = c[c[:, 0] >= 0]
    return [int(v) for v in c.mean(axis=0)] if len(c) else None


def _floor_height(hz: np.ndarray) -> Optional[float]:
    """Height of the dominant low peak (the floor). Plane averages are pulled up by table tops classified as floors."""
    lo = float(np.percentile(hz, 0.5))
    sel = hz[(hz >= lo) & (hz <= lo + 1.0)]
    if len(sel) < 50:
        return None
    h, e = np.histogram(sel, bins=np.arange(lo, lo + 1.0 + 0.02, 0.02))
    k = int(np.argmax(h))
    if h[k] < max(50, 0.02 * len(hz)):
        return None
    c = (e[k] + e[k + 1]) / 2
    return float(np.median(hz[np.abs(hz - c) < 0.03]))


def _ceiling_peaks(hh: np.ndarray, min_h: float = 1.4) -> List[float]:
    """Heights (above floor) of horizontal ceiling-like layers: histogram peaks of the high points."""
    cand = hh[hh > min_h]
    if len(cand) < 100:
        return []
    top = float(np.percentile(cand, 99.8))
    if top - min_h < 0.1:
        return [float(np.median(cand))]
    h, e = np.histogram(cand, bins=np.arange(min_h, top + 0.04, 0.02))
    sm = np.convolve(h, np.ones(3) / 3, mode="same")
    thr = max(80.0, 0.08 * sm.max())   # a stepped ceiling can cover a small share of the plan: keep the bar low, gate on coverage
    peaks = [i for i in range(len(sm)) if sm[i] >= thr and (i == 0 or sm[i] >= sm[i - 1])
             and (i == len(sm) - 1 or sm[i] > sm[i + 1])]
    merged: List[int] = []
    for i in peaks:
        if merged and (i - merged[-1]) * 0.02 < 0.06:
            if sm[i] > sm[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    return [float((e[i] + e[i + 1]) / 2) for i in merged]


def _inside_distance(raster: "_Raster", mask: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Exact metres from each plan point to the footprint outline: positive inside, negative outside.

    (A raster distance transform is quantised to whole cells -- 15 cm here -- which made 12-25 cm wall bands and
    furniture clearances meaningless.) -9 for points off the raster."""
    import cv2

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea).astype(np.float32) + 0.5          # pixel indices -> pixel centres
    px = np.stack([(ab[:, 0] - raster.amin) / raster.cell, (ab[:, 1] - raster.bmin) / raster.cell], axis=1)
    d = np.full(len(ab), -9.0)
    ia, ib, ok = raster.idx(ab)
    for i in np.flatnonzero(ok):
        d[i] = cv2.pointPolygonTest(c, (float(px[i, 0]), float(px[i, 1])), True) * raster.cell
    return d


def _segment(ab: np.ndarray, a: np.ndarray, b: np.ndarray):
    """(distance to the segment, coordinate along it in metres) for plan points."""
    d = b - a
    L = float(np.linalg.norm(d))
    t = ((ab - a) @ d) / (L * L)
    proj = a + np.clip(t, 0, 1)[:, None] * d
    return np.linalg.norm(ab - proj, axis=1), t * L


def refine_interior_semantics(world, result, oriented, positions: Dict[str, list], up, *,
                              structure_prefix: str = "struct") -> dict:
    """Mutates ``world`` (room properties, windows/doors, furniture entities); returns an accounting dict.

    Cloud-first: the compiled structure is only an anchor (one ROOM entity, wall entities to host openings); every
    measurement comes from the reconstructed points, so plane fragmentation does not change the result."""
    import cv2
    from scipy import ndimage
    from scipy.spatial import cKDTree

    from perception.architecture.classify import PlaneInput
    from perception.architecture.openings import detect_openings, promote_opening_to_entity

    report: dict = {"rooms": [], "furniture": [], "openings": [], "warnings": []}

    def done():
        world.metadata["interior_semantics"] = report
        return report

    if not result.points:
        report["warnings"].append("no points")
        return done()
    e1, e2, u = _basis(up)
    P = np.array([p.position for p in result.points], float)
    colors = np.array([(p.color if getattr(p, "color", None) is not None else (-1, -1, -1)) for p in result.points], float)
    has_color = bool((colors[:, 0] >= 0).any())
    tree = cKDTree(P)
    pl = _plan(P, e1, e2, u)
    AB, HZ = pl[:, :2], pl[:, 2]

    def mean_color(idx: np.ndarray) -> Optional[List[int]]:
        if not has_color or not len(idx):
            return None
        c = colors[idx]
        c = c[c[:, 0] >= 0]
        return [int(v) for v in c.mean(axis=0)] if len(c) else None

    # per-entity colour of the promoted planes (flat colour, measured)
    for o in oriented:
        ent = world.entities.get(f"{structure_prefix}-{o.plane.plane_id}")
        pts = positions.get(o.plane.plane_id, [])
        if ent is not None and pts and has_color:
            _, idx = tree.query(np.asarray(pts, float)[:: max(1, len(pts) // 1500)])
            col = mean_color(idx)
            if col:
                ent.custom_properties["mean_color_rgb"] = col

    rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
    anchor_created = False
    if not rooms:
        # The structural chain can lose its room (e.g. topology promotion drops it when a boundary plane was never
        # promoted). The cloud still describes one space, so the semantics pass creates the room itself, from the
        # promoted wall/floor/ceiling entities, and says so.
        # Only added to the world once the measurements succeed AND there is enclosure evidence (see below): an
        # exterior scene must never be given a room.
        boundary = sorted(e.id for e in world.entities.values() if e.type in (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING))
        room = Entity(id="room-001", name="room 001", type=EntityType.ROOM, provenance=Provenance.INFERRED, confidence=0.5,
                      custom_properties={"boundary_element_ids": boundary,
                                         "created_by": "room_semantics: the structural chain produced no room entity"})
        rooms = [room]
        anchor_created = True
    anchor = max(rooms, key=lambda r: float((r.custom_properties.get("floor_area_m2") or 0.0)))
    wall_planes = [(o.plane.plane_id, np.asarray(positions.get(o.plane.plane_id, []), float).reshape(-1, 3))
                   for o in oriented if o.role == "wall" and f"{structure_prefix}-{o.plane.plane_id}" in world.entities]

    # ---- floor height + footprint (floor-band points of the whole cloud) ----
    floor_h = _floor_height(HZ)
    if floor_h is None:
        report["skipped"] = "not refined: no dominant floor height peak"   # nothing to refine (e.g. an exterior scene); not an error
        return done()
    band = np.abs(HZ - floor_h) < 0.06
    F = AB[band]
    if len(F) < 100:
        report["skipped"] = "not refined: too few floor-band points"   # nothing to refine (e.g. an exterior scene); not an error
        return done()
    cell = _cell_size(F)
    raster = _Raster(AB[(HZ - floor_h) < 3.5], cell)
    # Floors are missing under/behind furniture, so the room outline also uses the BIG vertical planes (>= 1.5 m long,
    # >= 1 m tall -- a table side is neither): the wall ring encloses the room and its outer contour is the footprint.
    occ_pts = [F]
    for _pid, pp in wall_planes:
        if len(pp) >= 30:
            w = _plan(pp, e1, e2, u)
            if max(np.ptp(w[:, 0]), np.ptp(w[:, 1])) >= 1.5 and np.ptp(w[:, 2]) >= 1.0:
                occ_pts.append(w[:, :2])
    occ = raster.occupancy(np.vstack(occ_pts), close_m=0.4)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    if n < 2:
        report["skipped"] = "not refined: floor occupancy empty"   # nothing to refine (e.g. an exterior scene); not an error
        return done()
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == big).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, max(1.5, 0.15 / cell), True).reshape(-1, 2)
    mask = np.zeros_like(comp)
    cv2.fillPoly(mask, [approx.astype(np.int32)], 255)
    poly = raster.to_metres(approx.astype(float))
    area = _shoelace(poly)

    def to_world(ab, h):
        return [[round(float(v), 4) for v in (a * e1 + b * e2 + h * u)] for a, b in ab]

    dist_in = _inside_distance(raster, mask, AB)
    HH = HZ - floor_h
    wall_band = (dist_in > -0.15) & (dist_in < 0.12)
    inside = dist_in >= FURNITURE_WALL_CLEARANCE_M

    cp = anchor.custom_properties
    cp["bbox_floor_area_m2"] = cp.get("floor_area_m2")
    cp["floor_area_m2"] = round(area, 4)
    cp["footprint_polygon_m"] = [[round(float(a), 4), round(float(b), 4)] for a, b in poly]
    cp["footprint_polygon_xyz_m"] = to_world(poly, floor_h)
    cp["footprint_basis"] = {"e1": e1.tolist(), "e2": e2.tolist(), "up": u.tolist()}
    cp["floor_height_m"] = round(floor_h, 4)
    cp["footprint_method"] = (f"plan-view occupancy of {int(band.sum())} floor-band points "
                              f"(|h-floor|<6 cm, floor h {floor_h:.3f}), cell {cell:.3f} m")
    fcol = mean_color(np.flatnonzero(band & (dist_in > 0.05)))
    if fcol:
        cp["floor_color_rgb"] = fcol
    for other in rooms:                    # one scene group = one space: fragments are not extra rooms
        if other is not anchor:
            other.custom_properties["superseded_by"] = anchor.id
            if "fragment_of_room" not in other.semantic_labels:
                other.semantic_labels.append("fragment_of_room")

    # ---- ceiling layers from height peaks of the high points inside the room ----
    cand = inside & (HH > 1.4)
    peaks = _ceiling_peaks(HH[cand]) if cand.sum() >= 100 else []
    layers: List[dict] = []
    k = max(3, int(round(0.35 / cell)) | 1)   # reach back to the walls that the interior-only candidates leave out
    for h in peaks:
        sel = cand & (np.abs(HH - h) < 0.04)
        cov = raster.occupancy(AB[sel], close_m=0.3)
        cov = cv2.dilate(cov, np.ones((k, k), np.uint8))
        cov = np.logical_and(cov > 0, mask > 0)
        frac = float(cov.sum() / max(1, (mask > 0).sum()))
        if frac < 0.03:
            continue
        ids = []
        for o in oriented:
            ent = world.entities.get(f"{structure_prefix}-{o.plane.plane_id}")
            pts = positions.get(o.plane.plane_id, [])
            if (o.role == "ceiling" and ent is not None and pts
                    and abs(float(np.mean(np.asarray(pts, float) @ u)) - floor_h - h) < 0.08):
                ids.append(ent.id)
        layers.append({"height": h, "cov": cov, "frac": frac, "entity_ids": ids,
                       "color": mean_color(np.flatnonzero(sel))})
    stepped = len(layers) >= 2 and layers[-1]["height"] - layers[0]["height"] >= CEILING_STEP_M
    for i, l in enumerate(layers):
        if stepped and i < len(layers) - 1:
            top = layers[-1]["cov"]
            ov = np.logical_and(l["cov"], top).sum() / max(1, min(l["cov"].sum(), top.sum()))
            l["role"] = "false_ceiling" if (l["frac"] < 0.95 and ov < 0.5) else "ceiling"
        else:
            l["role"] = "structural_ceiling" if stepped else "ceiling"
        for eid in l["entity_ids"]:
            ent = world.entities[eid]
            ent.custom_properties.update({"ceiling_role": l["role"], "height_above_floor_m": round(l["height"], 4),
                                          "plan_coverage_fraction": round(l["frac"], 4)})
            if l["role"] in ("false_ceiling", "structural_ceiling") and l["role"] not in ent.semantic_labels:
                ent.semantic_labels.append(l["role"])
    cp["ceiling_layers"] = []
    for l in layers:
        poly_l = _mask_polygon(raster, l["cov"])
        entry = {"entity_ids": l["entity_ids"], "role": l["role"], "height_above_floor_m": round(l["height"], 4),
                 "plan_coverage_fraction": round(l["frac"], 4), "polygon_m": poly_l, "color_rgb": l["color"]}
        if poly_l:
            entry["polygon_xyz_m"] = to_world(np.asarray(poly_l), floor_h + l["height"])
        cp["ceiling_layers"].append(entry)
    cp["ceiling_evidence"] = "measured" if layers else "missing"
    if layers:
        cp["ceiling_height_min_m"] = round(layers[0]["height"], 4)
        cp["ceiling_height_max_m"] = round(layers[-1]["height"], 4)

    # ---- wall height per footprint edge (ceiling layer above it, else the measured top of the wall points) ----
    edge_heights: List[Optional[float]] = []
    edge_info = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        dseg, s = _segment(AB, a, b)
        L = float(np.linalg.norm(b - a))
        on = (dseg < 0.12) & (s > -0.05) & (s < L + 0.05) & (HH > 0.02)
        edge_info.append((a, b, L, on))
        h = None
        mid = (a + b) / 2.0
        nrm = np.array([-(b - a)[1], (b - a)[0]])
        nrm = nrm / (np.linalg.norm(nrm) or 1.0)
        probe = mid + nrm * 0.3
        if _inside_distance(raster, mask, probe[None, :])[0] <= 0:
            probe = mid - nrm * 0.3
        ia, ib = int((probe[0] - raster.amin) / cell), int((probe[1] - raster.bmin) / cell)
        for l in layers:
            if 0 <= ib < raster.h and 0 <= ia < raster.w and l["cov"][ib, ia]:
                h = round(l["height"], 4)
                break
        if h is None and on.sum() >= 50:
            h = round(float(np.percentile(HH[on], 98)), 4)
        if h is None and layers:
            h = round(layers[0]["height"], 4)
        edge_heights.append(h)
    cp["footprint_edge_heights_m"] = edge_heights
    if anchor_created:
        n_walls = sum(1 for e in world.entities.values() if e.type == EntityType.WALL)
        if not layers or n_walls < 3:
            report["skipped"] = ("no room created: no enclosure evidence (needs a measured ceiling and >= 3 promoted walls); "
                                 f"ceiling layers {len(layers)}, walls {n_walls}")
            return done()
        world.entities[anchor.id] = anchor
        report["warnings"].append("structural chain produced no room; room entity created from the point cloud")
    top = max([h for h in edge_heights if h] + [l["height"] for l in layers] + [2.0])
    wxyz = np.asarray(cp["footprint_polygon_xyz_m"], float)
    lo = np.minimum(wxyz.min(axis=0), wxyz.min(axis=0) - 0.0)
    hi = wxyz.max(axis=0) + u * top
    gid = anchor.geometry_ids[0] if anchor.geometry_ids else f"geom-{anchor.id}"
    world.geometries[gid] = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(*map(float, lo)),
                                     bounds_max=Vector3(*map(float, hi)), provenance=Provenance.INFERRED, confidence=0.5)
    if gid not in anchor.geometry_ids:
        anchor.geometry_ids.append(gid)
    wcol = mean_color(np.flatnonzero(wall_band & (HH > 0.3) & (HH < 1.8)))
    if wcol:
        cp["wall_color_rgb"] = wcol

    # ---- windows / doors: per wall edge, scan the points on that wall line with the existing opening detector ----
    n_open = 0
    for i, (a, b, L, on) in enumerate(edge_info):
        if L < 0.8 or on.sum() < 200:
            continue
        pts3 = P[on]
        nrm2 = np.array([-(b - a)[1], (b - a)[0]])
        nrm2 /= np.linalg.norm(nrm2) or 1.0
        nrm3 = nrm2[0] * e1 + nrm2[1] * e2
        plane = PlaneInput(plane_id=f"edge-{anchor.id}-{i}", normal=tuple(float(v) for v in nrm3),
                           centroid=tuple(float(v) for v in pts3.mean(axis=0)),
                           bounds_min=tuple(float(v) for v in pts3.min(axis=0)),
                           bounds_max=tuple(float(v) for v in pts3.max(axis=0)),
                           inlier_positions=tuple(tuple(float(v) for v in p) for p in pts3))
        try:
            fits = detect_openings(plane, up=tuple(float(v) for v in u), floor_height=floor_h)
        except Exception as exc:  # noqa: BLE001
            report["warnings"].append(f"opening scan edge {i}: {exc}")
            continue
        host, best = None, 0
        for pid, pp in wall_planes:
            if not len(pp):
                continue
            dd, ss = _segment(_plan(pp, e1, e2, u)[:, :2], a, b)
            cnt = int(((dd < 0.12) & (ss > -0.05) & (ss < L + 0.05)).sum())
            if cnt > best:
                host, best = f"{structure_prefix}-{pid}", cnt
        for fit in fits:
            perim = 2.0 * (fit.width_m + fit.height_m)
            if fit.n_points < MIN_OPENING_RING_PER_M * perim:
                report.setdefault("rejected_openings", []).append({
                    "edge": i, "kind": fit.kind, "width_m": round(fit.width_m, 3), "height_m": round(fit.height_m, 3),
                    "ring_points": fit.n_points, "needed": round(MIN_OPENING_RING_PER_M * perim, 1),
                    "reason": "bounding wall points too sparse to distinguish a real opening from missing SfM coverage"})
                continue
            if fit.kind == "door" and fit.height_m < MIN_DOOR_HEIGHT_M:
                report.setdefault("rejected_openings", []).append({
                    "edge": i, "kind": fit.kind, "width_m": round(fit.width_m, 3), "height_m": round(fit.height_m, 3),
                    "reason": f"floor-reaching gap only {fit.height_m:.2f} m tall: not a doorway (< {MIN_DOOR_HEIGHT_M} m)"})
                continue
            if host is None:
                report["warnings"].append(f"edge {i}: {fit.kind} found but no promoted wall entity hosts it")
                continue
            dup = any(e.type.value in ("window", "door", "opening") and np.linalg.norm(
                np.asarray(e.custom_properties.get("position", (1e9, 1e9, 1e9)), float) - np.asarray(fit.position)) < 0.4
                for e in world.entities.values())
            if dup:
                continue
            n_open += 1
            eid = f"opening-{anchor.id}-{i}-{n_open:02d}"
            try:
                promote_opening_to_entity(fit, world, eid, host_wall_id=host)
                report["openings"].append({"id": eid, "kind": fit.kind, "width_m": round(fit.width_m, 3),
                                           "height_m": round(fit.height_m, 3), "sill_m": round(fit.sill_height_m, 3),
                                           "edge": i, "host": host})
            except Exception as exc:  # noqa: BLE001
                report["warnings"].append(f"opening {eid}: {exc}")

    # ---- furniture: inside the footprint, off the walls, above the floor, below the lowest ceiling ----
    ceil_cap = (layers[0]["height"] - 0.15) if layers else 2.0
    sel = inside & (HH > FURNITURE_MIN_HEIGHT_M) & (HH < ceil_cap)
    n_obj = 0
    if sel.sum() >= MIN_CLUSTER_POINTS:
        idx_all = np.flatnonzero(sel)
        sc = np.stack([AB[sel, 0], AB[sel, 1], HH[sel]], axis=1)
        vox = np.floor(sc / CLUSTER_VOXEL_M).astype(int)
        vox -= vox.min(axis=0)
        grid = np.zeros(vox.max(axis=0) + 3, bool)
        grid[vox[:, 0] + 1, vox[:, 1] + 1, vox[:, 2] + 1] = True
        lab3, n3 = ndimage.label(ndimage.binary_dilation(grid, iterations=1), structure=np.ones((3, 3, 3)))
        cl = lab3[vox[:, 0] + 1, vox[:, 1] + 1, vox[:, 2] + 1]
        floor_ent = next((e for e in world.entities.values() if e.type == EntityType.FLOOR), None)
        for kk in range(1, n3 + 1):
            m = cl == kk
            if m.sum() < 40:
                continue
            cpts = sc[m]
            h0, h1 = float(np.percentile(cpts[:, 2], 2)), float(np.percentile(cpts[:, 2], 98))
            if h0 > 0.35:
                continue                   # not resting near the floor: fixture / noise, not furniture
            (ca, cb), (sa, sb), ang = cv2.minAreaRect(cpts[:, :2].astype(np.float32))
            if max(sa, sb) < 0.20 and (h1 - h0) < 0.3:
                continue
            thin = min(sa, sb) < 0.12
            if thin and (h1 - h0) > 1.2:
                continue                   # a tall zero-thickness vertical sheet is a wall remnant, not an object
            wp = P[idx_all[m]]
            bmin, bmax = wp.min(axis=0), wp.max(axis=0)
            centre = ca * e1 + cb * e2 + (floor_h + (h0 + h1) / 2) * u
            n_obj += 1
            eid = f"object-{n_obj:03d}"
            conf = min(0.7, 0.3 + float(m.sum()) / 800.0) * (0.6 if thin else 1.0)
            geom = Geometry(id=f"geom-{eid}", type=GeometryType.BOX, bounds_min=Vector3(*map(float, bmin)),
                            bounds_max=Vector3(*map(float, bmax)), provenance=Provenance.INFERRED, confidence=conf,
                            observations=[Observation(id=f"obs-object-geom-{eid}", sensor_type="point_cluster",
                                                      confidence=conf, metadata={"n_points": int(m.sum())})])
            world.geometries[geom.id] = geom
            ent = Entity(id=eid, name=f"furniture candidate {n_obj}", type=EntityType.UNKNOWN,
                         transform={"position": {"x": float(centre[0]), "y": float(centre[1]), "z": float(centre[2])}},
                         geometry_ids=[geom.id], semantic_labels=["furniture_candidate"],
                         provenance=Provenance.INFERRED, confidence=conf)
            ent.custom_properties.update({
                "room_id": anchor.id, "n_points": int(m.sum()), "height_m": round(h1 - h0, 4),
                "base_height_above_floor_m": round(h0, 4), "plan_obb_size_m": [round(float(sa), 4), round(float(sb), 4)],
                "plan_obb_centre_m": [round(float(ca), 4), round(float(cb), 4)], "plan_obb_angle_deg": round(float(ang), 2),
                "bounds_min": [float(v) for v in bmin], "bounds_max": [float(v) for v in bmax],
                "top_surface_fraction": round(float((cpts[:, 2] > h1 - 0.06).mean()), 3),
                "observed_faces": "single_face: only one side was seen, so depth is unknown" if thin else "multiple"})
            col = mean_color(idx_all[m])
            if col:
                ent.custom_properties["mean_color_rgb"] = col
            if floor_ent is not None:
                ent.relationships.append(Relationship(kind=RelationshipKind.RESTS_ON, target_id=floor_ent.id,
                                                      confidence=conf, provenance=Provenance.INFERRED))
            world.entities[eid] = ent
            report["furniture"].append({"id": eid, "size_m": [round(float(sa), 2), round(float(sb), 2), round(h1 - h0, 2)],
                                        "n_points": int(m.sum())})
    report["rooms"].append({"room_id": anchor.id, "footprint_vertices": len(poly), "footprint_area_m2": round(area, 3),
                            "bbox_area_m2": cp.get("bbox_floor_area_m2"), "floor_height_m": round(floor_h, 4),
                            "ceiling_layers": [(l["role"], round(l["height"], 3), round(l["frac"], 3)) for l in layers],
                            "superseded_rooms": [r.id for r in rooms if r is not anchor]})
    return done()
