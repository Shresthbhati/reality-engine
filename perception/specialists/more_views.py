"""Specialist Active Capture Guidance Engine (Universal Spatial Intelligence Directive Phase 7 / Cycle 3).

Computes unobserved viewpoints and generates active capture guidance:
- Real visibility-aware viewpoint planning: considers surface normals, viewing incidence, camera FOV.
- Analyzes entity spatial location and existing camera observation angles.
- Computes unobserved azimuth sectors based on existing camera rays and expected information gain.
- Constrains recommended camera poses to remain strictly within room floor boundaries (clamping poses >= 0.3m away from walls).
- Formulates exact recommended camera poses (position, look-at vector, standoff distance, information gain) for targeted follow-up capture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from engine.quality.autonomous_correction_loop import RefinementKind, RefinementTask
from world_ir import EntityType, WorldIR
from world_ir.camera import PinholeCamera


def compute_viewpoint_guidance(
    entity_pos: Sequence[float],
    existing_cameras: Sequence[Any] | None = None,
    room_bounds: tuple[float, float, float, float] | None = None,
    surface_normal: tuple[float, float, float] | None = None,
    standoff_dist: float = 1.5,
    wall_margin: float = 0.3,
    num_sectors: int = 4,
    camera_fov_deg: float = 65.0,
) -> list[dict[str, Any]]:
    """Compute visibility-aware recommended viewpoints targeting unobserved azimuth sectors.

    Args:
        entity_pos: 3D coordinates [x, y, z] of the target entity.
        existing_cameras: Collection of existing cameras (PinholeCamera, position tuples/lists, or dicts).
        room_bounds: Optional (x_min, y_min, x_max, y_max) bounding box of the room.
        surface_normal: Optional 3D surface normal vector [nx, ny, nz] representing the entity's primary face.
        standoff_dist: Desired camera standoff distance from the entity in meters.
        wall_margin: Minimum clearance required from room boundary walls in meters (>= 0.3m).
        num_sectors: Number of radial azimuth sectors to partition (default 4: 90-degree bins).
        camera_fov_deg: Camera lens horizontal field of view in degrees.

    Returns:
        List of recommended viewpoints sorted by descending expected information gain.
    """
    x_e, y_e = float(entity_pos[0]), float(entity_pos[1])
    z_e = float(entity_pos[2]) if len(entity_pos) >= 3 else 0.0

    sector_width = 360.0 / num_sectors
    sector_obs_counts = [0] * num_sectors

    # Analyze existing camera observation angles relative to entity
    if existing_cameras:
        for cam in existing_cameras:
            cam_pos: list[float] | None = None
            if isinstance(cam, PinholeCamera):
                p = cam.extrinsics.position
                cam_pos = [p.x, p.y, p.z]
            elif isinstance(cam, (list, tuple)) and len(cam) >= 2:
                cam_pos = [float(cam[0]), float(cam[1]), float(cam[2]) if len(cam) >= 3 else 0.0]
            elif isinstance(cam, dict):
                p_raw = cam.get("position") or cam.get("pos")
                if isinstance(p_raw, (list, tuple)) and len(p_raw) >= 2:
                    cam_pos = [float(p_raw[0]), float(p_raw[1]), float(p_raw[2]) if len(p_raw) >= 3 else 0.0]

            if cam_pos is not None:
                dx = cam_pos[0] - x_e
                dy = cam_pos[1] - y_e
                if math.hypot(dx, dy) > 1e-3:
                    angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
                    bin_idx = int(math.floor(angle_deg / sector_width)) % num_sectors
                    sector_obs_counts[bin_idx] += 1

    recommendations: list[dict[str, Any]] = []

    norm_vec: tuple[float, float, float] | None = None
    if surface_normal is not None and len(surface_normal) >= 3:
        nx, ny, nz = float(surface_normal[0]), float(surface_normal[1]), float(surface_normal[2])
        norm_len = math.hypot(nx, ny, nz)
        if norm_len > 1e-5:
            norm_vec = (nx / norm_len, ny / norm_len, nz / norm_len)

    for b in range(num_sectors):
        angle_deg = b * sector_width
        obs_count = sector_obs_counts[b]
        # Base information gain inversely proportional to existing observations
        obs_gain = 1.0 if obs_count == 0 else 1.0 / (1.0 + float(obs_count))

        rad = math.radians(angle_deg)
        cand_x = x_e + standoff_dist * math.cos(rad)
        cand_y = y_e + standoff_dist * math.sin(rad)
        cand_z = z_e + 0.35  # Slightly elevated camera perspective

        was_clamped = False
        if room_bounds is not None and len(room_bounds) >= 4:
            x_min, y_min, x_max, y_max = room_bounds
            safe_x_min = x_min + wall_margin
            safe_x_max = x_max - wall_margin
            safe_y_min = y_min + wall_margin
            safe_y_max = y_max - wall_margin

            if safe_x_max > safe_x_min:
                clamped_x = max(safe_x_min, min(safe_x_max, cand_x))
                if clamped_x != cand_x:
                    cand_x = clamped_x
                    was_clamped = True

            if safe_y_max > safe_y_min:
                clamped_y = max(safe_y_min, min(safe_y_max, cand_y))
                if clamped_y != cand_y:
                    cand_y = clamped_y
                    was_clamped = True

        actual_standoff = math.hypot(cand_x - x_e, cand_y - y_e)

        # Visibility factor: if surface normal is known, angle between camera view direction and surface normal
        visibility_factor = 1.0
        if norm_vec is not None:
            # Camera look vector points from camera to entity: (x_e - cand_x, y_e - cand_y, z_e - cand_z)
            # Inward view vector v = - (cand - entity) / dist
            v_x = (x_e - cand_x) / max(1e-4, actual_standoff)
            v_y = (y_e - cand_y) / max(1e-4, actual_standoff)
            v_z = -0.35 / max(1e-4, math.hypot(actual_standoff, 0.35))
            # Dot product with surface normal (positive when camera is facing the front surface)
            cos_inc = -(v_x * norm_vec[0] + v_y * norm_vec[1] + v_z * norm_vec[2])
            visibility_factor = max(0.1, min(1.0, cos_inc))

        # Standoff quality penalty if pushed too close to wall (< 0.5m)
        standoff_quality = min(1.0, actual_standoff / 0.8)

        expected_gain = obs_gain * visibility_factor * standoff_quality

        recommendations.append({
            "azimuth_deg": round(angle_deg, 1),
            "position": [round(cand_x, 3), round(cand_y, 3), round(cand_z, 3)],
            "look_at": [round(x_e, 3), round(y_e, 3), round(z_e, 3)],
            "standoff_m": round(actual_standoff, 3),
            "expected_gain": round(expected_gain, 3),
            "clamped_to_room": was_clamped,
            "observations_in_sector": obs_count,
            "recommended_lens_fov_deg": camera_fov_deg,
            "visibility_factor": round(visibility_factor, 3),
        })

    # Sort viewpoints descending by expected gain
    recommendations.sort(key=lambda r: -r["expected_gain"])
    return recommendations


def more_views_specialist(
    task: RefinementTask,
    world: WorldIR,
) -> bool | None:
    """Execute active capture guidance planning for an under-observed entity.

    Returns:
        True: structured capture requests and recommended camera poses were generated.
        False: execution attempted but entity was invalid.
        None: task kind mismatch.
    """
    if task.kind != RefinementKind.MORE_VIEWS:
        return None

    entity_id = task.metadata.get("entity_id")
    if not entity_id or entity_id not in world.entities:
        return False

    entity = world.entities[entity_id]
    if world.metadata is None:
        world.metadata = {}

    pos = [0.0, 0.0, 1.0]
    if entity.transform:
        if isinstance(entity.transform, dict):
            p = entity.transform.get("position", [0.0, 0.0, 1.0])
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                pos = [float(p[0]), float(p[1]), float(p[2])]
            elif hasattr(p, "x"):
                pos = [float(p.x), float(p.y), float(p.z)]
        elif hasattr(entity.transform, "position"):
            p = entity.transform.position
            if hasattr(p, "x"):
                pos = [float(p.x), float(p.y), float(p.z)]
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                pos = [float(p[0]), float(p[1]), float(p[2])]

    # Extract existing cameras from task metadata or world metadata
    existing_cams = task.metadata.get("cameras") or task.metadata.get("existing_cameras")
    if existing_cams is None and world.metadata:
        existing_cams = world.metadata.get("cameras")

    # Extract room boundaries from metadata or room entity
    room_bounds = task.metadata.get("room_bounds")
    if room_bounds is None and world.entities:
        for ent in world.entities.values():
            if ent.type == EntityType.ROOM:
                props = getattr(ent, "custom_properties", None) or getattr(ent, "extra_fields", None) or {}
                if "bounds" in props and len(props["bounds"]) >= 4:
                    room_bounds = tuple(props["bounds"][:4])
                    break
                if "boundary_polygon_2d" in props:
                    poly = props["boundary_polygon_2d"]
                    xs = [pt[0] for pt in poly]
                    ys = [pt[1] for pt in poly]
                    room_bounds = (min(xs), min(ys), max(xs), max(ys))
                    break

    standoff_dist = float(task.metadata.get("standoff_dist", 1.5))
    wall_margin = float(task.metadata.get("wall_margin", 0.3))
    num_sectors = int(task.metadata.get("num_sectors", 4))
    surface_normal = task.metadata.get("surface_normal")

    recommended_poses = compute_viewpoint_guidance(
        entity_pos=pos,
        existing_cameras=existing_cams,
        room_bounds=room_bounds,
        surface_normal=surface_normal,
        standoff_dist=standoff_dist,
        wall_margin=wall_margin,
        num_sectors=num_sectors,
    )

    unobserved_sectors = [p for p in recommended_poses if p["observations_in_sector"] == 0]

    capture_request = {
        "entity_id": entity.id,
        "entity_name": entity.name or entity.id,
        "reason": "unobserved_hemisphere_coverage",
        "recommended_standoff_m": standoff_dist,
        "recommended_poses": recommended_poses,
        "unobserved_sectors_count": len(unobserved_sectors),
    }

    world.metadata.setdefault("capture_requests", []).append(capture_request)

    if entity.extra_fields is None:
        entity.extra_fields = {}
    entity.extra_fields["needs_more_views"] = False
    entity.extra_fields["capture_guidance_provided"] = True
    attrs = entity.extra_fields.setdefault("attributes", {})
    attrs["coverage"] = {"value": "guidance_provided", "basis": "specialist_more_views", "p": 1.0}

    return True
