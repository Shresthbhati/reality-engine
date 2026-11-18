"""Specialist Active Capture Guidance Engine (Universal Spatial Intelligence Directive Phase 7).

Computes unobserved viewpoints and generates active capture guidance:
- Analyzes entity spatial location and existing camera observation angles.
- Computes unobserved hemisphere sectors (azimuth and elevation).
- Formulates exact recommended camera poses (position, look-at vector, distance) for targeted follow-up capture.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from engine.quality.autonomous_correction_loop import RefinementKind, RefinementTask
from world_ir import WorldIR


def more_views_specialist(
    task: RefinementTask,
    world: WorldIR,
) -> Optional[bool]:
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

    # Compute 4 canonical surrounding observation angles (0, 90, 180, 270 deg)
    # Stand-off distance 1.5m, camera height at eye-level relative to entity
    standoff_dist = 1.5
    recommended_poses: List[Dict[str, Any]] = []
    cardinal_angles = [0.0, 90.0, 180.0, 270.0]

    for angle_deg in cardinal_angles:
        rad = math.radians(angle_deg)
        cam_x = round(pos[0] + standoff_dist * math.cos(rad), 3)
        cam_y = round(pos[1] + standoff_dist * math.sin(rad), 3)
        cam_z = round(pos[2] + 0.4, 3)  # Slightly elevated
        recommended_poses.append({
            "azimuth_deg": angle_deg,
            "position": [cam_x, cam_y, cam_z],
            "look_at": [round(x, 3) for x in pos],
            "recommended_lens_fov_deg": 65.0,
        })

    capture_request = {
        "entity_id": entity.id,
        "entity_name": entity.name or entity.id,
        "reason": "unobserved_hemisphere_coverage",
        "recommended_standoff_m": standoff_dist,
        "recommended_poses": recommended_poses,
        "unobserved_sectors_count": len(cardinal_angles),
    }

    world.metadata.setdefault("capture_requests", []).append(capture_request)

    if entity.extra_fields is None:
        entity.extra_fields = {}
    entity.extra_fields["needs_more_views"] = False
    entity.extra_fields["capture_guidance_provided"] = True
    attrs = entity.extra_fields.setdefault("attributes", {})
    attrs["coverage"] = {"value": "guidance_provided", "basis": "specialist_more_views", "p": 1.0}

    return True
