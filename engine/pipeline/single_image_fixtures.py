"""Wall-mounted architectural fixtures (sockets, switches, vents) for single-image and multi-view reconstruction (Task 8.8, spec M2).

Uses `HighResCropPass` (Blueprint Task 9.1) on wall-plane regions of interest to detect
small electrical and mechanical fixtures, unprojects them onto the fitted host wall plane,
and creates traceable `MOUNTED_ON` relationships in the World IR.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.detail.crop_pass import HighResCropPass
from provenance import Provenance, Uncertainty
from world_ir import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
)


def unproject_ray_to_plane(
    u: float,
    v: float,
    f: float,
    cx: float,
    cy: float,
    plane_normal: Sequence[float],
    plane_d: float,
) -> Optional[Tuple[float, float, float]]:
    """Unproject image pixel (u, v) from pinhole camera at origin to 3D plane n·P + d = 0.

    Coordinate system (bootstrap frame):
      +X: right
      +Y: up (inverted vs image rows)
      +Z: forward (optical axis)
    """
    rx = (u - cx) / f
    ry = -(v - cy) / f
    rz = 1.0

    ray = np.array([rx, ry, rz], dtype=float)
    normal = np.array(plane_normal, dtype=float)

    denom = float(np.dot(normal, ray))
    if abs(denom) < 1e-6:
        return None  # Ray parallel to plane

    t = -float(plane_d) / denom
    if t <= 0.05:  # Behind camera or too close to lens
        return None

    pt = ray * t
    return (float(pt[0]), float(pt[1]), float(pt[2]))


def place_wall_fixtures(
    world,
    item,
    wall_pixels: Dict[str, Sequence[Tuple[int, int]]],
    oriented_walls: Sequence[Any],
    *,
    f: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
    pre: str = "",
    crop_pass: Optional[HighResCropPass] = None,
    image_array: Optional[np.ndarray] = None,
    target_classes: Sequence[str] = ("fixtures/electrical/socket", "fixtures/electrical/switch"),
) -> Dict[str, Any]:
    """Detect and place wall-mounted fixtures on fitted wall planes in the World.

    Returns a facts summary dictionary detailing attempted regions, detected fixtures,
    and placed entities.
    """
    facts: Dict[str, Any] = {
        "attempted": True,
        "method": "HighResCropPass + optical ray intersection on host wall planes",
        "host_walls_searched": len(wall_pixels),
        "fixtures_detected": 0,
        "fixtures_placed": 0,
        "placed_ids": [],
        "rejected": [],
    }

    if not wall_pixels or not oriented_walls:
        facts["attempted"] = False
        facts["reason"] = "no wall planes available for fixture mounting"
        return facts

    # Map host wall plane entity ID to oriented plane
    wall_plane_map = {op.plane.plane_id: op for op in oriented_walls if hasattr(op, "plane")}
    if not wall_plane_map:
        wall_plane_map = {f"wall_{i}": op for i, op in enumerate(oriented_walls)}

    # Acquire image array
    if image_array is None:
        try:
            from pathlib import Path
            from urllib.parse import unquote, urlparse
            from urllib.request import url2pathname
            from PIL import Image

            from engine.core.image_io import open_oriented

            u = urlparse(getattr(item, "source_uri", str(item)))
            path = Path(url2pathname(unquote(u.path)) if u.scheme == "file" else getattr(item, "source_uri", str(item)))
            with open_oriented(path) as im:
                image_array = np.array(im.convert("RGB"))
        except Exception as exc:
            facts["attempted"] = False
            facts["reason"] = f"could not load image for crop pass: {exc}"
            return facts

    img_h, img_w = image_array.shape[:2]

    # Initialize HighResCropPass if not provided
    if crop_pass is None:
        crop_pass = HighResCropPass(target_classes=target_classes)

    fixture_count = 0

    for host_eid, wpix in wall_pixels.items():
        if not wpix:
            continue

        op = wall_plane_map.get(host_eid)
        if op is None:
            continue

        # Compute host wall bounding region in image coordinates
        rows = [p[0] for p in wpix]
        cols = [p[1] for p in wpix]
        r_min, r_max = max(0, min(rows)), min(img_h - 1, max(rows))
        c_min, c_max = max(0, min(cols)), min(img_w - 1, max(cols))

        # Discard degenerate regions (< 30px width or height)
        if (r_max - r_min) < 30 or (c_max - c_min) < 30:
            continue

        wall_region = (float(c_min), float(r_min), float(c_max), float(r_max))

        # Run crop pass over the wall region
        frame_id = getattr(item, "id", "single_view")
        crop_res = crop_pass.run_pass(
            image_array,
            candidate_regions=[wall_region],
            frame_id=frame_id,
        )

        facts["fixtures_detected"] += len(crop_res.accepted_observations)
        for rej in crop_res.rejected_detections:
            facts["rejected"].append({
                "class_path": rej.class_path,
                "reason": rej.reason,
                "confidence": rej.confidence,
                "host_wall": host_eid,
            })

        # Plane parameters (n·x + d = 0)
        plane_normal = op.normal
        plane_d = op.d

        for obs in crop_res.accepted_observations:
            if obs.bbox_px is None:
                continue

            bx0, by0, bx1, by1 = obs.bbox_px
            u_center = (bx0 + bx1) / 2.0
            v_center = (by0 + by1) / 2.0

            # Scale to intrinsic pixel grid if needed
            u_scaled = u_center * (w / float(img_w))
            v_scaled = v_center * (h / float(img_h))

            # Unproject ray to host wall plane
            pt_3d = unproject_ray_to_plane(u_scaled, v_scaled, f, cx, cy, plane_normal, plane_d)
            if pt_3d is None:
                facts["rejected"].append({
                    "class_path": obs.label,
                    "reason": "ray does not intersect host wall plane in front of camera",
                    "host_wall": host_eid,
                })
                continue

            label_leaf = obs.label.split("/")[-1]
            ent_id = f"{pre}fixture_{label_leaf}_{fixture_count:03d}"
            geom_id = f"geom_{ent_id}"
            fixture_count += 1

            # Standard standard electrical/mechanical fixture dimensions (plate size: ~8.6cm x 8.6cm x 2cm)
            half_w = 0.086 / 2.0
            half_h = 0.086 / 2.0
            half_d = 0.02 / 2.0

            geom = Geometry(
                id=geom_id,
                type=GeometryType.BOX,
                bounds_min=Vector3(x=pt_3d[0] - half_w, y=pt_3d[1] - half_h, z=pt_3d[2] - half_d),
                bounds_max=Vector3(x=pt_3d[0] + half_w, y=pt_3d[1] + half_h, z=pt_3d[2] + half_d),
                provenance=Provenance.INFERRED,
                confidence=float(obs.confidence),
            )
            world.geometries[geom_id] = geom

            mounted_rel = Relationship(
                kind=RelationshipKind.MOUNTED_ON,
                target_id=host_eid,
                confidence=float(obs.confidence),
                provenance=Provenance.INFERRED,
                metadata={
                    "basis": "optical ray unprojection onto fitted wall plane",
                    "pixel_box": [float(c) for c in obs.bbox_px],
                },
            )

            fixture_entity = Entity(
                id=ent_id,
                name=f"{label_leaf.title()} on {host_eid}",
                type=EntityType.INFRASTRUCTURE,
                transform={
                    "position": {"x": pt_3d[0], "y": pt_3d[1], "z": pt_3d[2]},
                    "normal": [float(n) for n in plane_normal],
                },
                geometry_ids=[geom_id],
                relationships=[mounted_rel],
                semantic_labels=[obs.label],
                provenance=Provenance.INFERRED,
                confidence=float(obs.confidence),
                uncertainty=Uncertainty(
                    confidence=float(obs.confidence),
                    note="optical ray unprojection onto fitted wall plane",
                ),
                custom_properties={
                    "subclass": obs.label,
                    "host_wall": host_eid,
                    "pixel_box": [float(c) for c in obs.bbox_px],
                },
            )

            world.entities[ent_id] = fixture_entity
            if hasattr(world, "relationships"):
                world.relationships.append(mounted_rel)

            facts["fixtures_placed"] += 1
            facts["placed_ids"].append(ent_id)

    return facts
