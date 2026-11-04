"""Solid wall / floor / ceiling entities from the solved room layout, with openings carved into their host walls
(plan tasks S3 + S4).

Runs after ``room_semantics.refine_interior_semantics``. Per room that carries a measured footprint:

* ``room_layout.solve_room_layout`` regularises the footprint into an ordered wall loop;
* each wall becomes a WALL entity (label ``wall_slab``) whose geometry is a real extruded slab -- a MESH in the artifact
  store when one is given (so glTF export carries the wall), else a BOX with the slab's exact description in
  ``custom_properties["wall_slab"]``, which the viewer builds from;
* the floor and every measured ceiling layer become polygon slabs (FLOOR / CEILING entities, label ``polygon_slab``);
* the reconstructed plane entities stay as EVIDENCE: the slab is ``ATTACHED_TO`` the planes that support it and the
  planes carry ``slab_wall_id`` -- they are not deleted and not re-typed;
* each detected door / window / opening is attached to its host wall by plane distance and in-plane bounds, its in-wall
  rectangle is stored, and the slab mesh is cut around it. A gap in a region no camera could have viewed is NOT an
  opening: it is recorded as ``unobserved_gap``, not carved.

Nothing here invents a measurement: slab thickness is a labelled default, a wall with no measured height is not built,
inferred wall segments carry ``provenance=INFERRED`` and low confidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from provenance import Provenance
from world_ir import Entity, EntityType, Geometry, GeometryType, Relationship, RelationshipKind, Vector3

from perception.architecture.openings import host_opening_on_wall
from perception.architecture.room_layout import (
    DEFAULT_THICKNESS_M, VIEW_MODEL, polygon_slab_mesh, solve_room_layout, wall_slab_mesh, wall_view_grid,
)
from perception.architecture.room_semantics import _basis, _plan

#: labelled defaults: no stage observes slab thickness
FLOOR_THICKNESS_M = 0.15
CEILING_THICKNESS_M = 0.10
#: an opening is carved only if at least this share of its rectangle was inside some camera's (assumed) view
MIN_VIEWED_FRACTION = 0.5
#: opposite parallel walls whose spans overlap at least this share of the shorter wall must agree in length
OPPOSITE_OVERLAP_FRACTION = 0.9
#: agreement tolerance: max(5 cm, 3 % of the longer wall)
OPPOSITE_TOL_ABS_M = 0.05
OPPOSITE_TOL_REL = 0.03
#: a plane is evidence for a wall when this many of its points lie on the wall's inner face
EVIDENCE_MIN_POINTS = 20
#: a plane is evidence for a wall only when MOST of its own points lie on that wall. Without this, the corner columns
#: of a perpendicular wall (within one band width of the edge end) made every slab claim 3 of the 4 planes, and the
#: slab's measured colour became the average of three unrelated walls.
EVIDENCE_MIN_FRACTION = 0.5
EVIDENCE_BAND_M = 0.12
CREATED_BY = "room_layout"

_OPENING_TYPES = (EntityType.DOOR, EntityType.WINDOW, EntityType.OPENING)


def _store_mesh(world, artifact_store, gid: str, verts, faces, conf: float, prov: Provenance) -> Geometry:
    v = np.asarray(verts, float)
    lo, hi = v.min(axis=0), v.max(axis=0)
    geom = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(*map(float, lo)), bounds_max=Vector3(*map(float, hi)),
                    provenance=prov, confidence=conf)
    if artifact_store is not None:
        from world_ir.geometry_data import MeshData

        mesh = MeshData(vertices=tuple(map(tuple, v.tolist())), faces=tuple(tuple(int(i) for i in f) for f in faces))
        geom.type = GeometryType.MESH
        geom.data_uri, geom.data_hash = artifact_store.put(mesh.to_bytes())
        geom.vertex_count, geom.triangle_count = len(mesh.vertices), len(mesh.faces)
    world.geometries[gid] = geom
    return geom


def _centre(geom: Geometry) -> dict:
    lo, hi = geom.bounds_min, geom.bounds_max
    return {"position": {"x": (lo.x + hi.x) / 2, "y": (lo.y + hi.y) / 2, "z": (lo.z + hi.z) / 2}}


def _purge(world, room_id: str) -> None:
    """Idempotence: a recompile into the same world replaces this room's slabs instead of stacking duplicates."""
    gone = {e.id for e in world.entities.values()
            if e.custom_properties.get("created_by") == CREATED_BY and e.custom_properties.get("room_id") == room_id}
    for eid in gone:
        for gid in world.entities[eid].geometry_ids:
            world.geometries.pop(gid, None)
        del world.entities[eid]
    if gone:
        for e in world.entities.values():
            e.relationships[:] = [r for r in e.relationships if r.target_id not in gone]


def _link(a: Entity, b: Entity, kind_ab: RelationshipKind, kind_ba: Optional[RelationshipKind], why: str, conf: float) -> None:
    a.relationships.append(Relationship(kind=kind_ab, target_id=b.id, confidence=conf, provenance=Provenance.INFERRED,
                                        metadata={"derived_from": why}))
    if kind_ba is not None:
        b.relationships.append(Relationship(kind=kind_ba, target_id=a.id, confidence=conf, provenance=Provenance.INFERRED,
                                            metadata={"derived_from": why}))


def _mean_rgb(ents: Sequence[Entity]) -> Optional[List[int]]:
    cols = [e.custom_properties["mean_color_rgb"] for e in ents if e.custom_properties.get("mean_color_rgb")]
    return [int(round(sum(c[k] for c in cols) / len(cols))) for k in range(3)] if cols else None


def layout_consistency(layout, ceiling_layers: Sequence[dict] = ()) -> dict:
    """D3 dimensional consistency of the solved layout, measured -- never forced.

    * L/W are the layout's plan extents (not the axis-aligned box of a possibly L-shaped room);
    * height is reported per measured ceiling layer, so a soffit or a false ceiling is not flattened to one number;
    * opposite parallel walls whose spans cover most of each other must agree in length within max(5 cm, 3 %); a
      disagreement is recorded with both lengths and flagged ``ok=False`` -- the mismatch is the evidence, not an error
      to hide. Openings are clipped into their host wall by ``host_opening_on_wall`` (offset/width/sill/height always
      fit by construction), and are reported separately per wall."""
    walls = [w for w in layout.walls if w.length_m > 1e-6]
    if not walls:
        return {"ok": False, "reason": "no walls to check", "extent_u_m": None, "extent_v_m": None,
                "height_layers_m": [], "mismatches": []}
    xs = [float(p) for w in walls for p in (w.a[0], w.b[0])]
    ys = [float(p) for w in walls for p in (w.a[1], w.b[1])]
    ext_u, ext_v = max(xs) - min(xs), max(ys) - min(ys)
    mismatches, seen = [], set()
    for i, w in enumerate(walls):
        L = w.length_m
        di = np.array([(w.b[0] - w.a[0]) / L, (w.b[1] - w.a[1]) / L])
        group = []                              # (wall, interval of it projected onto w)
        for j, v in enumerate(walls):
            if j == i:
                continue
            M = v.length_m
            dj = np.array([(v.b[0] - v.a[0]) / M, (v.b[1] - v.a[1]) / M])
            if float(di @ dj) > -0.995:        # not opposite-parallel
                continue
            offset = abs(float((v.a[0] - w.a[0]) * di[1] - (v.a[1] - w.a[1]) * di[0]))
            if offset < 0.05:                  # collinear continuation, not the far side of the room
                continue
            sb = sorted(float((p[0] - w.a[0]) * di[0] + (p[1] - w.a[1]) * di[1]) for p in (v.a, v.b))
            overlap = max(0.0, min(L, sb[1]) - max(0.0, sb[0]))
            if overlap < OPPOSITE_OVERLAP_FRACTION * min(L, M):
                continue
            group.append((v, sb))
        if not group:
            continue
        # A wall of an L / U-shaped room faces SEVERAL collinear walls whose lengths add up to its own, so the
        # comparison is the part of this wall that nothing faces, never its length against one neighbour.
        merged = []
        for lo, hi in sorted((max(0.0, a), min(L, b)) for _, (a, b) in group):
            if merged and lo <= merged[-1][1] + 0.05:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        union_len = sum(hi - lo for lo, hi in merged)
        tol = max(OPPOSITE_TOL_ABS_M, OPPOSITE_TOL_REL * max([L] + [v.length_m for v, _ in group]))
        if L - union_len <= tol:               # fully faced (a facing wall may extend past it: judged from its side)
            continue
        ids = sorted([w.index + 1] + [v.index + 1 for v, _ in group])
        key = tuple(ids)
        if key in seen:                        # the same disagreement seen from the other wall
            continue
        seen.add(key)
        if len(group) == 1:
            v = group[0][0]
            label = f"opposite walls {w.index + 1} ({L:.3f} m) and {v.index + 1} ({v.length_m:.3f} m)"
            lb = union_len
        else:
            label = (f"wall {w.index + 1} ({L:.3f} m) and the walls facing it "
                     f"{[v.index + 1 for v, _ in group]} ({union_len:.3f} m together)")
            lb = union_len
        mismatches.append({"walls": ids, "length_a_m": round(L, 4), "length_b_m": round(lb, 4),
                           "difference_m": round(abs(L - lb), 4), "tol_m": round(tol, 4),
                           "detail": f"{label} differ by {abs(L - lb):.3f} m > {tol:.3f} m"})
    heights = sorted({round(float(w.height_m), 4) for w in walls if w.height_m is not None})
    layer_heights = sorted({round(float(layer["height_above_floor_m"]), 4)
                            for layer in ceiling_layers if layer.get("height_above_floor_m") is not None})
    return {"ok": not mismatches, "extent_u_m": round(ext_u, 4), "extent_v_m": round(ext_v, 4),
            "length_m": round(max(ext_u, ext_v), 4), "width_m": round(min(ext_u, ext_v), 4),
            "height_m": max(heights) if heights else None, "wall_heights_m": heights,
            "height_layers_m": layer_heights or heights, "mismatches": mismatches}


def promote_room_layout(world, result, up, *, wall_planes: Sequence[Tuple[str, np.ndarray]] = (),
                        artifact_store=None, structure_prefix: str = "struct",
                        layout_edges_ab: Optional[Sequence[Sequence[Sequence[float]]]] = None) -> dict:
    """Mutates ``world``; returns an accounting dict (also stored at ``world.metadata["room_layout"]``)."""
    report: dict = {"rooms": [], "warnings": [], "view_model": dict(VIEW_MODEL)}
    world.metadata["room_layout"] = report
    rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM and "superseded_by" not in e.custom_properties
             and e.custom_properties.get("footprint_polygon_m") and e.custom_properties.get("footprint_basis")]
    if not rooms:
        report["skipped"] = "no room with a measured footprint polygon"
        return report
    P = np.array([p.position for p in result.points], float)
    for room in rooms:
        try:
            _promote_room(world, room, result, P, up, wall_planes, artifact_store, report,
                          layout_edges_ab=layout_edges_ab)
        except Exception as exc:  # noqa: BLE001 -- the structural world stands; the failure is visible
            report["warnings"].append(f"{room.id}: {type(exc).__name__}: {exc}")
    return report


def _promote_room(world, room: Entity, result, P: np.ndarray, up, wall_planes, artifact_store, report: dict,
                  layout_edges_ab=None) -> None:
    cp = room.custom_properties
    e1, e2, u = _basis(cp["footprint_basis"].get("up", up))
    floor_h = float(cp["floor_height_m"])
    poly = np.asarray(cp["footprint_polygon_m"], float)
    heights = cp.get("footprint_edge_heights_m")

    pl = _plan(P, e1, e2, u)
    AB, HH = pl[:, :2], pl[:, 2] - floor_h
    top = max([h for h in (heights or []) if h] + [cp.get("ceiling_height_max_m") or 0.0, 2.0])
    near = (HH > 0.05) & (HH < top + 0.3)
    layout = solve_room_layout(poly.tolist(), heights, AB[near], layout_edges_ab=layout_edges_ab)
    consistency = layout_consistency(layout, cp.get("ceiling_layers") or [])
    cp["layout_dimension_u_m"] = consistency["extent_u_m"]
    cp["layout_dimension_v_m"] = consistency["extent_v_m"]
    cp["layout_height_layers_m"] = consistency["height_layers_m"]
    cp["dimensional_consistency"] = consistency
    report_room = {"room_id": room.id, "closed": layout.closed, "walls": len(layout.walls),
                   "dominant_angle_deg": layout.dominant_angle_deg, "manhattan_fraction": layout.manhattan_fraction,
                   "consistency": consistency,
                   "inferred_walls": [], "openings": [], "unobserved_gaps": [], "skipped_walls": [], "slabs": []}
    report["warnings"].extend(f"{room.id}: dimensional inconsistency: {m['detail']}" for m in consistency["mismatches"])
    report["rooms"].append(report_room)
    report["warnings"].extend(f"{room.id}: {w}" for w in layout.warnings)
    if not layout.walls:
        return
    _purge(world, room.id)

    world_pt = lambda ab, h: ab[0] * e1 + ab[1] * e2 + h * u               # noqa: E731

    # ---- openings: candidates already promoted by the detectors, attached to a host wall in the layout ----
    placed: Dict[int, List[dict]] = {}
    for ent in world.entities.values():
        if ent.type not in _OPENING_TYPES or ent.custom_properties.get("created_by") == CREATED_BY:
            continue
        bmin, bmax = ent.custom_properties.get("bounds_min"), ent.custom_properties.get("bounds_max")
        if not bmin or not bmax:
            continue
        corners = [(x, y, z) for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])]
        cabh = [(float(c @ e1), float(c @ e2), float(c @ u) - floor_h) for c in np.asarray(corners, float)]
        rect = host_opening_on_wall(cabh, layout.walls)
        if rect is None:
            continue
        placed.setdefault(rect.wall_index, []).append({"entity": ent, "rect": rect, "kind": ent.type.value})

    # ---- wall slabs ----
    wall_ents: List[Entity] = []
    plan_wall_xy = {pid: _plan(np.asarray(pp, float).reshape(-1, 3), e1, e2, u)[:, :2] for pid, pp in wall_planes if len(pp)}
    for w in layout.walls:
        if w.height_m is None or w.height_m <= 0:
            report_room["skipped_walls"].append({"index": w.index, "reason": "no measured wall height: not extruded, not guessed"})
            continue
        L = w.length_m
        d2 = np.array([(w.b[0] - w.a[0]) / L, (w.b[1] - w.a[1]) / L])
        in2 = np.array(layout.inward(w))
        origin = world_pt(np.asarray(w.a), floor_h)
        dir_w, in_w = d2[0] * e1 + d2[1] * e2, in2[0] * e1 + in2[1] * e2
        eid = f"wall-{room.id}-{w.index + 1:02d}"
        # openings on this wall: viewed ones are carved, the rest are recorded as unobserved gaps
        grid = wall_view_grid(origin, dir_w, in_w, u, L, w.height_m, getattr(result, "camera_poses", []))
        carve, gaps = [], []
        for p in placed.get(w.index, []):
            r = p["rect"]
            sx0, sx1 = int(r.offset_m / 0.25), max(int(r.offset_m / 0.25) + 1, int(np.ceil((r.offset_m + r.width_m) / 0.25)))
            sy0, sy1 = int(r.sill_m / 0.25), max(int(r.sill_m / 0.25) + 1, int(np.ceil((r.sill_m + r.height_m) / 0.25)))
            cells = grid[sy0:sy1, sx0:sx1]
            viewed = float(cells.mean()) if cells.size else 0.0
            rec = {"entity_id": p["entity"].id, "kind": p["kind"], **r.to_dict(), "viewed_fraction": round(viewed, 3)}
            (carve if viewed >= MIN_VIEWED_FRACTION else gaps).append(rec)
        slab_openings = [{k: v for k, v in c.items() if k in ("entity_id", "kind", "offset_m", "sill_m", "width_m", "height_m")}
                         for c in carve]
        verts, faces = wall_slab_mesh(origin, dir_w, -in_w, u, L, w.height_m, w.thickness_m, slab_openings)
        conf = w.confidence
        prov = Provenance.RECONSTRUCTED if w.provenance == "OBSERVED" else Provenance.INFERRED
        geom = _store_mesh(world, artifact_store, f"geom-{eid}", verts, faces, conf, prov)
        ev = [pid for pid, xy in plan_wall_xy.items()
              if len(xy) and _on_edge(xy, np.asarray(w.a), np.asarray(w.b))
              >= max(EVIDENCE_MIN_POINTS, EVIDENCE_MIN_FRACTION * len(xy))]
        ent = Entity(id=eid, type=EntityType.WALL, name=f"Wall {w.index + 1}", transform=_centre(geom),
                     geometry_ids=[geom.id], semantic_labels=["wall_slab"] + (["inferred_wall_segment"] if w.provenance == "INFERRED" else []),
                     provenance=prov, confidence=conf)
        ent.custom_properties.update({
            "created_by": CREATED_BY, "room_id": room.id, "length_m": round(L, 4), "height_m": round(w.height_m, 4),
            "thickness_m": w.thickness_m, "thickness_basis": w.thickness_basis, "height_basis": w.height_basis,
            "observed_fraction": w.observed_fraction, "unobserved_spans_m": [list(s) for s in w.unobserved_spans_m],
            "snapped_to_dominant_axes": w.snapped, "evidence_plane_ids": ev,
            "wall_slab": {
                "start_xyz_m": [round(float(v), 5) for v in origin], "end_xyz_m": [round(float(v), 5) for v in world_pt(np.asarray(w.b), floor_h)],
                "direction": [round(float(v), 6) for v in dir_w], "outward": [round(float(-v), 6) for v in in_w],
                "up": [round(float(v), 6) for v in u], "length_m": round(L, 4), "height_m": round(w.height_m, 4),
                "thickness_m": w.thickness_m, "openings": slab_openings,
                "unobserved_gaps": [{k: v for k, v in g.items() if k in ("entity_id", "kind", "offset_m", "sill_m", "width_m", "height_m")} for g in gaps]},
        })
        col = _mean_rgb([world.entities[p] for p in ev if p in world.entities]) or cp.get("wall_color_rgb")
        if col:
            ent.custom_properties["mean_color_rgb"] = list(col)
        if w.provenance == "INFERRED":
            report_room["inferred_walls"].append(eid)
        world.entities[eid] = ent
        wall_ents.append(ent)
        _link(ent, room, RelationshipKind.PART_OF, RelationshipKind.CONTAINS, "room_layout_wall", conf)
        for pid in ev:
            if pid in world.entities:
                world.entities[pid].custom_properties["slab_wall_id"] = eid
                _link(ent, world.entities[pid], RelationshipKind.ATTACHED_TO, None, "wall_slab_evidence_plane", conf)
        for c in carve:
            opening = world.entities[c["entity_id"]]
            opening.custom_properties["in_wall"] = {"wall_id": eid, **{k: c[k] for k in ("offset_m", "sill_m", "width_m", "height_m", "plane_distance_m", "viewed_fraction")},
                                                    "observation_state": "observed", "carved": True}
            _link(opening, ent, RelationshipKind.PART_OF, RelationshipKind.CONTAINS, "opening_host_wall", opening.confidence)
            report_room["openings"].append({"id": c["entity_id"], "wall": eid, **{k: c[k] for k in ("kind", "offset_m", "sill_m", "width_m", "height_m")}})
        for g in gaps:
            opening = world.entities[g["entity_id"]]
            opening.custom_properties["in_wall"] = {"wall_id": eid, **{k: g[k] for k in ("offset_m", "sill_m", "width_m", "height_m", "plane_distance_m", "viewed_fraction")},
                                                    "observation_state": "unobserved_gap", "carved": False,
                                                    "note": "no camera viewed this region: a gap here is missing coverage, not evidence of an opening"}
            report_room["unobserved_gaps"].append({"id": g["entity_id"], "wall": eid, "viewed_fraction": g["viewed_fraction"]})

    # ---- floor + ceiling slabs ----
    xyz = lambda ab, h: [[round(float(v), 5) for v in world_pt(np.asarray(p), h)] for p in ab]     # noqa: E731
    ev_floor = [world.entities[b] for b in cp.get("boundary_element_ids", [])
                if b in world.entities and world.entities[b].type == EntityType.FLOOR]
    if not ev_floor:      # a room created by the semantics pass has no boundary list; its floor plane is PART_OF it
        ev_floor = [e for e in world.entities.values() if e.type == EntityType.FLOOR
                    and any(r.target_id == room.id and r.kind.value == "part_of" for r in e.relationships)]
    slabs: List[Tuple[str, EntityType, list, list, float, int, dict, Optional[list], List[Entity], float]] = [
        (f"floor-slab-{room.id}", EntityType.FLOOR, layout.polygon, xyz(layout.polygon, floor_h), FLOOR_THICKNESS_M, -1,
         {}, cp.get("floor_color_rgb"), ev_floor, 0.8)]
    for k, layer in enumerate(cp.get("ceiling_layers") or []):
        lp, lxyz = layer.get("polygon_m"), layer.get("polygon_xyz_m")
        if not lp or not lxyz:
            continue
        slabs.append((f"ceiling-slab-{room.id}-{k + 1:02d}", EntityType.CEILING, lp, lxyz, CEILING_THICKNESS_M, 1,
                      {"ceiling_role": layer.get("role"), "height_above_floor_m": layer.get("height_above_floor_m")},
                      layer.get("color_rgb"), ([world.entities[i] for i in layer.get("entity_ids", []) if i in world.entities]
                                               or [e for e in world.entities.values() if e.type == EntityType.CEILING
                                                   and any(r.target_id == room.id and r.kind.value == "part_of"
                                                           for r in e.relationships)]), 0.7))
    for eid, etype, ab, top_xyz, thick, direction, extra, color, evidence, conf in slabs:
        verts, faces = polygon_slab_mesh(top_xyz, u, thick, direction, ab)
        if not verts:
            report["warnings"].append(f"{eid}: footprint polygon could not be triangulated; slab not built")
            continue
        geom = _store_mesh(world, artifact_store, f"geom-{eid}", verts, faces, conf, Provenance.RECONSTRUCTED)
        ent = Entity(id=eid, type=etype, name=eid, transform=_centre(geom), geometry_ids=[geom.id], semantic_labels=["polygon_slab"],
                     provenance=Provenance.RECONSTRUCTED, confidence=conf)
        ent.custom_properties.update({
            "created_by": CREATED_BY, "room_id": room.id, "thickness_m": thick, "thickness_basis": "default_unmeasured",
            "evidence_plane_ids": [e.id for e in evidence],
            "slab": {"polygon_xyz_m": top_xyz, "up": [round(float(v), 6) for v in u], "thickness_m": thick, "direction": direction},
            **extra})
        col = color or _mean_rgb(evidence)
        if col:
            ent.custom_properties["mean_color_rgb"] = list(col)
        world.entities[eid] = ent
        _link(ent, room, RelationshipKind.PART_OF, RelationshipKind.CONTAINS, "room_layout_slab", conf)
        for ev in evidence:
            ev.custom_properties["slab_id"] = eid
            _link(ent, ev, RelationshipKind.ATTACHED_TO, None, "polygon_slab_evidence_plane", conf)
        report_room["slabs"].append(eid)
    cp["wall_slab_ids"] = [e.id for e in wall_ents]


def _on_edge(xy: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return 0
    u = d / L
    rel = xy - a
    s = rel @ u
    off = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    return int(((off < EVIDENCE_BAND_M) & (s > -0.05) & (s < L + 0.05)).sum())
