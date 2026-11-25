"""Validated adapter between the perception ``SpatialInputsV1`` contract
(``perception/spatial_inputs.py``) and the global spatial optimizer's input
contract (``reconstruction/spatial/stage.py``).

The two structures are deliberately *not* interchangeable:

* ``SpatialInputsV1`` carries 3-D camera poses (position + quaternion),
  general detections (``observations``) and structural observations whose
  payloads are ``WallSlabObservation`` / ``OpeningObservation`` dicts, plus
  an explicit ``ScaleStateInput`` that says whether scale is metric.
* The spatial optimizer expects a flat dict with schema ``spatial_inputs.v1``,
  ``scale_state == "metric"`` (metre scale is never assumed), plan-view poses
  (``theta``, ``tx``, ``ty``) and structural observations as
  ``StructuralObservation`` dicts (plan-view end points ``p0``/``p1``,
  ``sigma_m`` in metres).

The adapter translates between them **without fabricating anything**:

* A camera with no position/orientation contributes no pose (absent stays
  absent -- never a guessed pose).
* A structural observation whose payload lacks plan-view end points or a
  positive sigma is rejected (never turned into a confident wall).
* Non-metric scale is rejected: relative / SFM-gauge / unknown scale never
  becomes "metric".
* Every rejected observation is recorded in the returned ``rejected`` list so
  the caller can see what was dropped and why.

Returns ``(inputs_dict, rejected)`` where ``inputs_dict`` is the spatial
stage input dict (or ``None`` when there is nothing usable) and ``rejected``
is a list of ``{"observation_id", "reason"}`` dicts.
"""

from __future__ import annotations

import math
from typing import Any

from perception.spatial_inputs import SpatialInputsV1

SCHEMA = "spatial_inputs.v1"


def _quat_yaw(w: float, x: float, y: float, z: float) -> float:
    """Yaw (rotation about Z) from a unit quaternion (w, x, y, z)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _finite_pos(v) -> bool:
    return v is not None and all(isinstance(c, (int, float)) and math.isfinite(c) for c in v)


def _sigma_positive(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0


def _wall_from_payload(observation_id: str, image_id: str, payload: dict[str, Any], evidence_id: str) -> dict[str, Any] | None:
    """Build a ``StructuralObservation`` dict from a ``wall_slab`` payload, or
    ``None`` when the payload does not carry enough plan-view geometry."""
    start = payload.get("start_xyz_m")
    end = payload.get("end_xyz_m")
    if not (_finite_pos(start) and _finite_pos(end)):
        return None
    sigma = payload.get("measurement_uncertainty_m")
    if not _sigma_positive(sigma):
        return None
    # Plan-view end points (X, Y). Z is preserved as the vertical extent.
    z0, z1 = start[2], end[2]
    raw_vis = payload.get("visibility_state", "observed")
    # Map perception visibility states to spatial stage Visibility enum values
    _VIS_MAP = {"observed": "full", "partial": "partial", "inferred": "partial", "unresolved": "occluded"}
    visibility = _VIS_MAP.get(raw_vis, "full")
    end0_observed = raw_vis == "observed"
    end1_observed = raw_vis == "observed"
    return {
        "evidence_id": evidence_id,
        "image_id": image_id,
        "kind": "wall",
        "p0": [float(start[0]), float(start[1])],
        "p1": [float(end[0]), float(end[1])],
        "sigma_m": float(sigma),
        "z0": float(z0),
        "z1": float(z1),
        "visibility": visibility,
        "end0_observed": end0_observed,
        "end1_observed": end1_observed,
        "label_confidence": None,
        "depth_source": payload.get("depth_basis", "unknown"),
        "frame_id": image_id,  # frame_id must equal image_id for the spatial optimizer
        "producer": f"perception:{payload.get('source_camera', 'unknown')}",
        "storey": None,
    }


def adapt_spatial_inputs_v1(
    si: SpatialInputsV1,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Convert a validated ``SpatialInputsV1`` into the spatial optimizer's
    input dict.

    Returns ``(inputs, rejected)``. ``inputs`` is ``None`` when the scale is
    not metric or there are no usable poses/observations. ``rejected`` lists
    every observation that was dropped and why.
    """
    rejected: list[dict[str, str]] = []

    # ---- scale: never assume metric. Relative / SFM-gauge / unknown stay as-is;
    # the optimizer refuses to optimize without explicit metric scale. ----
    if not si.scale_state.is_metric:
        return None, [{"observation_id": "<scale>", "reason": f"scale_state={si.scale_state.basis!r} is not metric; metre scale is never assumed"}]

    # ---- poses from cameras ------------------------------------------------
    poses: list[dict[str, Any]] = []
    for cam in si.cameras:
        if not _finite_pos(cam.position_m) or cam.orientation_wxyz is None:
            rejected.append({"observation_id": f"camera:{cam.camera_id}", "reason": "camera has no position or orientation"})
            continue
        w, x, y, z = cam.orientation_wxyz
        poses.append({
            "image_id": cam.image_id,
            "theta": _quat_yaw(w, x, y, z),
            "tx": float(cam.position_m[0]),
            "ty": float(cam.position_m[1]),
            "sigma_theta": None,
            "sigma_t": None,
        })

    # ---- structural observations ------------------------------------------
    observations: list[dict[str, Any]] = []
    for so in si.structural_observations:
        payload = so.payload or {}
        kind = so.kind
        if kind == "wall_slab":
            obs = _wall_from_payload(so.observation_id, so.evidence_id, payload, so.evidence_id)
            if obs is None:
                rejected.append({"observation_id": so.observation_id, "reason": "wall_slab payload lacks plan-view end points or positive sigma"})
                continue
            obs["image_id"] = so.evidence_id  # the evidence id doubles as the source image id for structural observations
            observations.append(obs)
        elif kind == "opening":
            # Openings are recorded but not converted to plan-view line observations
            # unless the payload carries a host wall and location we can use.
            rejected.append({"observation_id": so.observation_id, "reason": "opening observations are not yet converted to plan-view structural lines"})
        else:
            rejected.append({"observation_id": so.observation_id, "reason": f"unhandled structural kind {kind!r}"})

    if not poses and not observations:
        return None, rejected

    storeys: dict[str, float] = {}
    if si.storey_id:
        storeys[si.storey_id] = 0.0

    inputs: dict[str, Any] = {
        "schema": SCHEMA,
        "scale_state": "metric",
        "frame": si.provenance.get("frame", "world") if isinstance(si.provenance, dict) else "world",
        "storeys": storeys,
        "poses": poses,
        "observations": observations,
    }
    return inputs, rejected


__all__ = ["adapt_spatial_inputs_v1", "SCHEMA"]
