"""Detected-plane -> WorldIR promotion (Goal D closing another loop:
geometry-derived entities + measurements, with provenance).

Turns an OrientedPlane (from perception/geometry/orientation.py, itself
fed by RANSAC over a ReconstructionResult) into a real WorldIR Entity
typed WALL/FLOOR/CEILING per the role the geometry earned, carrying:

  - a PLANE Geometry whose bounds are the axis-aligned bounding box of
    the plane's own inlier points (real reconstructed points, never
    invented fills), provenance RECONSTRUCTED (the points are;
    classification is layered on top, see below),
  - an extent Measurement computed from the inliers' actual span
    (diagonal of the bounding box, precision from the plane's fit RMS),
    provenance ESTIMATED -- geometry-derived numbers are estimates from
    reconstructed data, never presented as exact observations,
  - a wall-thickness Measurement ONLY for walls whose parallel
    counterpart face is present within WALL_PAIR_THICKNESS_M with
    sufficient span overlap: the honest, purely geometric way to get a
    thickness. A single-surface wall has no measurable thickness here
    and gets none -- inferring one would be fabrication. (Because
    orientation.py flips every plane's normal toward the cameras, the
    two faces of one wall photographed from the same side carry
    same-facing normals; what defines a wall pair geometrically is
    parallelism + small perpendicular gap + overlapping span, not
    normal-sign opposition -- that sign test was a real bug observed and
    fixed during this work.)
  - provenance INFERRED on the entity itself (the *type* WALL/FLOOR/
    CEILING is an inference from geometry; the underlying points are
    RECONSTRUCTED and that stays visible on the Geometry),
  - an Observation recording the supporting evidence chain (inlier
    count, fit RMS, classification note) so "why does the engine believe
    this is a wall?" is answerable from the WorldIR alone.

Refuses (raises) rather than fabricates: an "unknown"-role plane cannot
be promoted through this path because there is no honest EntityType for
"a plane we couldn't classify" -- callers can still promote the raw
points via the existing reconstruction promotion path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from provenance import Provenance
from world_ir import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Measurement,
    Observation,
    Vector3,
    WorldIR,
)

from perception.geometry.orientation import OrientedPlane
from reconstruction.backend.interface import ReconstructionResult

#: Maximum perpendicular gap between two opposite-facing parallel planes
#: for them to count as the two faces of one wall (and yield a thickness
#: measurement). Beyond this they are just two separate planes.
WALL_PAIR_THICKNESS_M = 0.5

#: Minimum overlap fraction (projected along the wall's span direction)
#: two opposite faces must share to be considered faces of the same wall
#: -- a 1 m stub next to a 10 m wall is not a pair.
WALL_PAIR_MIN_OVERLAP_FRACTION = 0.5

#: Two planes count as parallel within this many degrees.
PARALLEL_TOLERANCE_DEG = 5.0

_ROLE_TO_ENTITY_TYPE: Dict[str, EntityType] = {
    "wall": EntityType.WALL,
    "floor": EntityType.FLOOR,
    "ceiling": EntityType.CEILING,
}


class PlanePromotionError(ValueError):
    pass


@dataclass(frozen=True)
class PlanePromotionResult:
    entity: Entity
    geometry: Geometry
    extent_measurement: Measurement
    #: (partner plane_id, thickness measurement) when an opposite wall
    #: face was found; None otherwise -- absence is honest, not failure.
    thickness_measurement: Tuple[str, Measurement] | None


def positions_by_plane(
    result: ReconstructionResult,
    planes: List[OrientedPlane],
) -> Dict[str, List[Tuple[float, float, float]]]:
    """Resolve every plane's inlier positions once, keyed by plane id.

    Raises PlanePromotionError if any plane references a point id absent
    from the reconstruction result -- promotion would otherwise silently
    shrink a plane's evidence set.
    """
    by_id = {p.track_id: p.position for p in result.points}
    resolved: Dict[str, List[Tuple[float, float, float]]] = {}
    for oriented in planes:
        inlier_ids = oriented.plane.inlier_ids
        missing = [pid for pid in inlier_ids if pid not in by_id]
        if missing:
            raise PlanePromotionError(
                f"plane {oriented.plane.plane_id} references points not in the "
                f"reconstruction result: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        resolved[oriented.plane.plane_id] = [by_id[pid] for pid in inlier_ids]
    return resolved


def _bounds(positions: List[Tuple[float, float, float]]) -> Tuple[Vector3, Vector3]:
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    return Vector3(min(xs), min(ys), min(zs)), Vector3(max(xs), max(ys), max(zs))


def _wall_span_direction(normal: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """The horizontal axis least aligned with a wall's normal: for a wall
    with normal near +-X this is Y, near +-Y it is X -- the direction the
    wall physically extends along, used for face-overlap testing."""
    if abs(normal[0]) >= abs(normal[1]):
        return (0.0, 1.0, 0.0)
    return (1.0, 0.0, 0.0)


def _span_along(
    positions: List[Tuple[float, float, float]], direction: Tuple[float, float, float]
) -> Tuple[float, float]:
    projections = [p[0] * direction[0] + p[1] * direction[1] + p[2] * direction[2] for p in positions]
    return min(projections), max(projections)


def _plane_gap(a: OrientedPlane, b: OrientedPlane) -> float:
    """Perpendicular distance between two parallel planes (normal . p + d = 0).

    b's plane constant is re-signed into a's normal direction first: with
    normals as unit vectors, two parallel planes share the normal up to
    sign, and the distance is |d_a - d_b'| / |n| where d_b' is b's constant
    expressed against a's normal (negated when the normals anti-align).
    Getting the sign wrong here yields |d_a + d_b| -- correct only for the
    anti-aligned representation, and silently wrong distances otherwise
    (observed: the first version of this function was only correct because
    the test walls happened to sit at x~0, i.e. d_a~0)."""
    dot = a.normal[0] * b.normal[0] + a.normal[1] * b.normal[1] + a.normal[2] * b.normal[2]
    sign = 1.0 if dot >= 0.0 else -1.0
    n_len = math.sqrt(sum(c * c for c in a.normal))
    return abs(a.d - sign * b.d) / max(1e-12, n_len)


def find_wall_pair_face(
    oriented: OrientedPlane,
    positions: List[Tuple[float, float, float]],
    other_planes: List[OrientedPlane],
    positions_for_planes: Dict[str, List[Tuple[float, float, float]]],
) -> OrientedPlane | None:
    """The counterpart face of a wall, by the WALL_PAIR_* rules: parallel
    (normal aligned or anti-aligned -- orientation.py has already flipped
    each normal toward the cameras, so same-facing is the common case for
    two faces seen from one side), within WALL_PAIR_THICKNESS_M, and with
    sufficient overlap along the wall's span direction. Returns None when
    no candidate qualifies -- no thickness is measured, never invented."""
    if not positions:
        return None

    direction = _wall_span_direction(oriented.normal)
    own_lo, own_hi = _span_along(positions, direction)
    own_len = own_hi - own_lo
    if own_len <= 0.0:
        return None

    parallel_threshold = math.cos(math.radians(PARALLEL_TOLERANCE_DEG))
    best: OrientedPlane | None = None
    best_overlap = 0.0
    for other in other_planes:
        if other.plane.plane_id == oriented.plane.plane_id:
            continue
        if other.role != "wall":
            continue
        dot = (
            oriented.normal[0] * other.normal[0]
            + oriented.normal[1] * other.normal[1]
            + oriented.normal[2] * other.normal[2]
        )
        if abs(dot) < parallel_threshold:
            continue  # not parallel within tolerance
        gap = _plane_gap(oriented, other)
        if gap <= 0.0 or gap > WALL_PAIR_THICKNESS_M:
            continue
        other_positions = positions_for_planes.get(other.plane.plane_id)
        if not other_positions:
            continue  # no positions for the candidate: skip honestly
        o_lo, o_hi = _span_along(other_positions, direction)
        overlap = max(0.0, min(own_hi, o_hi) - max(own_lo, o_lo))
        fraction = overlap / own_len
        if fraction >= WALL_PAIR_MIN_OVERLAP_FRACTION and fraction > best_overlap:
            best = other
            best_overlap = fraction
    return best


def promote_plane_to_entity(
    oriented: OrientedPlane,
    result: ReconstructionResult,
    world: WorldIR,
    entity_id: str,
    entity_name: str = "",
    other_planes: List[OrientedPlane] | None = None,
    plane_positions: Dict[str, List[Tuple[float, float, float]]] | None = None,
) -> PlanePromotionResult:
    """Promote one classified plane into WorldIR.

    `result` must be the ReconstructionResult the plane was detected in
    (its points are the evidence; ids are cross-checked). `other_planes`
    enables the wall-thickness pairing search and requires `plane_positions`
    (use positions_by_plane()); when omitted, no thickness is measured --
    never fabricated.

    Mutates `world` (adds Entity + Geometry) exactly like
    evidence/promote_reconstruction.py does: promotion is the write path;
    detection/classification stay pure.
    """
    entity_type = _ROLE_TO_ENTITY_TYPE.get(oriented.role)
    if entity_type is None:
        raise PlanePromotionError(
            f"plane {oriented.plane.plane_id} has role '{oriented.role}' -- only "
            "wall/floor/ceiling planes can be promoted as structure; unknown-role "
            "planes stay unpromoted rather than typed by guesswork"
        )

    plane_positions = plane_positions or {}
    positions = plane_positions.get(oriented.plane.plane_id)
    if positions is None:
        positions = positions_by_plane(result, [oriented])[oriented.plane.plane_id]

    bounds_min, bounds_max = _bounds(positions)

    # Extent: diagonal of the inlier bounding box -- a real,
    # geometry-derived size of the observed structure. Precision comes
    # from the plane's own fit RMS (floored so a suspiciously perfect fit
    # doesn't claim zero uncertainty).
    extent = math.sqrt(
        (bounds_max.x - bounds_min.x) ** 2
        + (bounds_max.y - bounds_min.y) ** 2
        + (bounds_max.z - bounds_min.z) ** 2
    )
    precision = max(oriented.plane.inlier_rms_distance_m, 0.01)
    extent_measurement = Measurement(
        value=extent,
        unit="meter",
        precision=precision,
        provenance=Provenance.ESTIMATED,
        confidence=oriented.uncertainty.confidence,
    )

    geometry = Geometry(
        id=f"geom-{entity_id}",
        type=GeometryType.PLANE,
        vertex_count=len(oriented.plane.inlier_ids),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        provenance=Provenance.RECONSTRUCTED,
        confidence=oriented.plane.uncertainty.confidence,
        observations=[Observation(
            id=f"obs-plane-{entity_id}",
            sensor_type="geometric_reasoning",
            confidence=oriented.uncertainty.confidence,
            metadata={
                "plane_id": oriented.plane.plane_id,
                "inlier_count": oriented.plane.inlier_count,
                "inlier_rms_distance_m": oriented.plane.inlier_rms_distance_m,
                "role": oriented.role,
                "classification_note": oriented.uncertainty.note or "",
            },
        )],
    )
    world.geometries[geometry.id] = geometry

    # Transform: the AABB centroid (midpoint of bounds_min/bounds_max) --
    # real data already computed above from the plane's own inliers, not
    # a fabricated placement. Without this, exporters (exporters/gltf,
    # exporters/usd, exporters/blender) have no `transform.position` to
    # place an object at and silently skip every promoted plane entity.
    centroid = Vector3(
        (bounds_min.x + bounds_max.x) / 2.0,
        (bounds_min.y + bounds_max.y) / 2.0,
        (bounds_min.z + bounds_max.z) / 2.0,
    )

    entity = Entity(
        id=entity_id,
        name=entity_name or entity_id,
        type=entity_type,
        transform={"position": {"x": centroid.x, "y": centroid.y, "z": centroid.z}},
        geometry_ids=[geometry.id],
        provenance=Provenance.INFERRED,
        confidence=oriented.uncertainty.confidence,
    )

    # Wall thickness: a real measurement only when an opposite face exists.
    thickness: Tuple[str, Measurement] | None = None
    if oriented.role == "wall" and other_planes:
        if plane_positions is None:
            raise PlanePromotionError(
                "other_planes requires plane_positions (use positions_by_plane()) "
                "-- pairing needs the candidates' actual inlier positions"
            )
        partner = find_wall_pair_face(oriented, positions, other_planes, plane_positions)
        if partner is not None:
            gap = _plane_gap(oriented, partner)
            thickness = (
                partner.plane.plane_id,
                Measurement(
                    value=gap,
                    unit="meter",
                    precision=max(
                        oriented.plane.inlier_rms_distance_m,
                        partner.plane.inlier_rms_distance_m,
                        0.01,
                    ),
                    provenance=Provenance.ESTIMATED,
                    confidence=min(oriented.uncertainty.confidence, partner.uncertainty.confidence),
                ),
            )

    if thickness is not None:
        entity.custom_properties["thickness_m"] = thickness[1].value
        entity.custom_properties["thickness_partner_plane_id"] = thickness[0]

    entity.observations.append(Observation(
        id=f"obs-promotion-{entity_id}",
        sensor_type="plane_promotion",
        confidence=oriented.uncertainty.confidence,
        metadata={
            "role": oriented.role,
            "inlier_count": oriented.plane.inlier_count,
            "extent_m": extent,
            "thickness_m": thickness[1].value if thickness else None,
            "thickness_partner_plane_id": thickness[0] if thickness else None,
        },
    ))

    world.entities[entity_id] = entity
    return PlanePromotionResult(
        entity=entity,
        geometry=geometry,
        extent_measurement=extent_measurement,
        thickness_measurement=thickness,
    )
