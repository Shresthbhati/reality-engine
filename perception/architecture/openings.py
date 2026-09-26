"""Opening detection + discrimination (INTERIOR RECONSTRUCTION
MISSION requirements 4/5).

The engine already measured two SEPARATE opening facts about a wall's
inlier coverage:

  classify.detect_wall_opening  -> the single largest floor-reaching
      interior gap (the "door-shaped" fact; classifies DOOR_MAX_WIDTH_M
      and DOOR_SCAN_BAND_M);
  windows.detect_window         -> the largest sill-bounded interior
      rectangle (edge-bounded on all four sides, sill in a window band).

Nothing composed them: a wall carrying both a door AND a window, or a
hole that is NEITHER door-shaped nor window-shaped (a pass-through, a
wall break), was either silently one or the other or nothing. This
module scans the wall's coverage grid ONCE, measures every interior
empty rectangle, and classifies each by measured shape:

  door      gap reaches the floor band (sill ~ 0) and is door-sized
  window    gap is edge-bounded with sill in the window band
  opening   measured hole that neither class can claim -- still
            reported, with its real geometry, as a generic opening
            (uncertainty preserved instead of a forced label)

Discrimination rules (all measured, no thresholds invented here):
  - sill = gap bottom - floor_height (the measured bottom edge);
  - a gap whose sill is ~0 (within SCAN_BUCKET_M of the floor) is
    door-SHAPED; it is a DOOR iff its height reaches the door band
    (>= windows.WIN_MIN_HEIGHT_M, i.e. tall enough to walk through --
    reusing the window band's floor keeps a single source of truth for
    "tall"); otherwise it is a generic opening (a knee-wall gap);
  - a gap with sill >= MIN_SILL_M is a WINDOW iff width and height sit
    inside the window bands (windows.py's own constants); otherwise it
    is a generic opening (e.g. a clerestory slot or oversized break);
  - boundary-touching gaps are refused as evidence of missing data,
    never openings (windows.py's edge-bounded rule, applied to ALL
    kinds: a door whose sides touch the wall-face boundary is exactly
    as suspect as a window in the same position).

Deterministic: fixed grid scan order, stable sorting, pure float math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from perception.architecture.classify import (
    DOOR_MAX_WIDTH_M,
    SCAN_BUCKET_M,
    PlaneInput,
)
from perception.architecture.windows import (
    MAX_SILL_M,
    MIN_SILL_M,
    WIN_MAX_HEIGHT_M,
    WIN_MAX_WIDTH_M,
    WIN_MIN_HEIGHT_M,
    WIN_MIN_WIDTH_M,
)

Vec3 = Tuple[float, float, float]

#: A gap bottom within one bucket of the floor band is "reaching the
#: floor" (door-shaped bottom). One bucket = the scan resolution; two
#: gaps differing by less than the measurement granularity are the same
#: measurement.
DOOR_SILL_BAND_M = SCAN_BUCKET_M

#: Minimum height for a floor-reaching gap to count as a DOOR (tall
#: enough to walk through). Reuses the window band's minimum height as
#: the single "tall" reference -- no new threshold invented.
DOOR_MIN_HEIGHT_M = WIN_MIN_HEIGHT_M

#: Minimum size for ANY opening to be reported (a one-bucket pinhole
#: is measurement noise, not architecture).
MIN_OPENING_WIDTH_M = 0.3
MIN_OPENING_HEIGHT_M = 0.3


@dataclass(frozen=True)
class OpeningFit:
    """One measured wall opening: real geometry + measured class.

    kind is the measured shape classification ("door" | "window" |
    "opening"); a generic opening is still fully measured -- only the
    SEMANTIC class is uncertain, never the geometry.
    """

    kind: str
    wall_plane_id: str
    position: Vec3
    width_m: float
    height_m: float
    #: Gap bottom above the caller's floor reference.
    sill_height_m: float
    bounds_min: Vec3
    bounds_max: Vec3
    #: Wall inlier points bordering the gap (the bounding ring).
    n_points: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "wall_plane_id": self.wall_plane_id,
            "position": list(self.position),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "sill_height_m": self.sill_height_m,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "n_points": self.n_points,
            "confidence": self.confidence,
        }


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: Vec3) -> Vec3:
    n = math.sqrt(_dot(v, v))
    if n < 1e-12:
        raise ValueError(f"degenerate vector {v!r}")
    return (v[0] / n, v[1] / n, v[2] / n)


def _largest_rectangles(
    grid: List[List[bool]], limit: int = 8
) -> List[Tuple[int, int, int, int]]:
    """Up to `limit` disjoint interior empty rectangles, largest first.

    Classic largest-rectangle-per-row histogram, but instead of stopping
    at the first result, zero out the found rectangle's cells and
    continue scanning -- so a wall with a door AND a window reports
    both. Deterministic: per-row left-to-right, ties broken by position.
    Returns (ix0, iz0, ix1, iz1) with exclusive x1/y1, in grid indices.
    """
    out: List[Tuple[int, int, int, int]] = []
    nx = len(grid[0]) if grid else 0
    nz = len(grid)
    work = [row[:] for row in grid]
    while len(out) < limit:
        best = None
        best_area = 0
        heights = [0] * nx
        for iz in range(nz):
            for ix in range(nx):
                heights[ix] = 0 if work[iz][ix] else heights[ix] + 1
            stack: List[Tuple[int, int]] = []
            for ix in range(nx + 1):
                h = heights[ix] if ix < nx else 0
                start = ix
                while stack and stack[-1][1] >= h:
                    sx, sh = stack.pop()
                    area = sh * (ix - sx)
                    if area > best_area:
                        best_area = area
                        best = (sx, iz - sh + 1, ix, iz)
                    start = sx
                stack.append((start, h))
        if best is None or best_area == 0:
            break
        bx0, bz0, bx1, bz1 = best
        out.append((bx0, bz0, bx1, bz1 + 1))
        for iz in range(bz0, bz1 + 1):
            for ix in range(bx0, bx1):
                work[iz][ix] = True  # consumed
    return out


def _coverage_grid(
    wall: PlaneInput, up_u: Vec3, lateral: Vec3
) -> Tuple[List[List[bool]], float, float, int, int]:
    """Boolean coverage grid over the wall's lateral-x-height extent
    (True = occupied by an inlier). Same projection as windows.py."""
    lats = [_dot(p, lateral) for p in wall.inlier_positions]
    hs = [_dot(p, up_u) for p in wall.inlier_positions]
    lat_min, lat_max = min(lats), max(lats)
    z_min, z_max = min(hs), max(hs)
    nx = max(1, int(math.ceil((lat_max - lat_min) / SCAN_BUCKET_M)))
    nz = max(1, int(math.ceil((z_max - z_min) / SCAN_BUCKET_M)))
    grid = [[False] * nx for _ in range(nz)]
    for lat, h in zip(lats, hs):
        ix = min(nx - 1, int((lat - lat_min) / SCAN_BUCKET_M))
        iz = min(nz - 1, int((h - z_min) / SCAN_BUCKET_M))
        grid[iz][ix] = True
    return grid, lat_min, z_min, nx, nz


def _classify_gap(
    gap_w: float, gap_h: float, sill: float
) -> str:
    """Measured shape classification for one interior gap."""
    if sill <= DOOR_SILL_BAND_M:
        # Door-shaped bottom: a DOOR iff tall enough to walk through
        # and not wider than the door band allows.
        if (
            gap_h >= DOOR_MIN_HEIGHT_M
            and gap_w <= max(DOOR_MAX_WIDTH_M, WIN_MAX_WIDTH_M)
        ):
            return "door"
        return "opening"
    if sill >= MIN_SILL_M and sill <= MAX_SILL_M:
        if (
            WIN_MIN_WIDTH_M <= gap_w <= WIN_MAX_WIDTH_M
            and WIN_MIN_HEIGHT_M <= gap_h <= WIN_MAX_HEIGHT_M
        ):
            return "window"
    return "opening"


def detect_openings(
    wall: PlaneInput, up: Vec3 = (0.0, 0.0, 1.0), floor_height: float = 0.0
) -> List[OpeningFit]:
    """Measure every interior opening in a wall's inlier coverage.

    Returns openings sorted by lateral position (deterministic). A wall
    with no inliers, no lateral extent, or no interior gaps returns []
    (recorded absence -- never a guessed opening).
    """
    if not wall.inlier_positions:
        return []
    up_u = _normalize(up)
    lateral = (
        wall.normal[1] * up_u[2] - wall.normal[2] * up_u[1],
        wall.normal[2] * up_u[0] - wall.normal[0] * up_u[2],
        wall.normal[0] * up_u[1] - wall.normal[1] * up_u[0],
    )
    lateral_len = math.sqrt(_dot(lateral, lateral))
    if lateral_len < 1e-9:
        return []  # not a wall (normal parallel to up)
    lateral = (lateral[0] / lateral_len, lateral[1] / lateral_len, lateral[2] / lateral_len)

    grid, lat_min, z_min, nx, nz = _coverage_grid(wall, up_u, lateral)

    fits: List[OpeningFit] = []
    centroid_lat = _dot(wall.centroid, lateral)
    centroid_h = _dot(wall.centroid, up_u)

    for ix0, iz0, ix1, iz1 in _largest_rectangles(grid):
        gap_w = (ix1 - ix0) * SCAN_BUCKET_M
        gap_h = (iz1 - iz0) * SCAN_BUCKET_M
        if gap_w < MIN_OPENING_WIDTH_M or gap_h < MIN_OPENING_HEIGHT_M:
            continue
        # Edge-bounded rule: a gap touching the wall-face boundary (any
        # side except the bottom, which a door legitimately reaches) is
        # missing data, not an opening edge.
        interior_sides = ix0 > 0 and ix1 < nx and iz1 < nz
        if not interior_sides:
            continue
        gap_z_lo = z_min + iz0 * SCAN_BUCKET_M
        sill = gap_z_lo - floor_height
        kind = _classify_gap(gap_w, gap_h, sill)
        if kind == "window":
            # Window band re-check with the measured sill: a sill above
            # the window band is a generic opening, not a window.
            if sill > MAX_SILL_M:
                kind = "opening"

        gap_lat_lo = lat_min + ix0 * SCAN_BUCKET_M
        gap_lat_hi = gap_lat_lo + gap_w
        gap_z_hi = gap_z_lo + gap_h

        def _point(lat_v: float, h_v: float) -> Vec3:
            return tuple(
                wall.centroid[k]
                + (lat_v - centroid_lat) * lateral[k]
                + (h_v - centroid_h) * up_u[k]
                for k in range(3)
            )

        corner_a = _point(gap_lat_lo, gap_z_lo)
        corner_b = _point(gap_lat_hi, gap_z_hi)
        bounds_min = tuple(min(corner_a[k], corner_b[k]) for k in range(3))
        bounds_max = tuple(max(corner_a[k], corner_b[k]) for k in range(3))
        center = (
            (corner_a[0] + corner_b[0]) / 2.0,
            (corner_a[1] + corner_b[1]) / 2.0,
            (corner_a[2] + corner_b[2]) / 2.0,
        )

        # Bounding-ring coverage: the inliers around the gap are what
        # make it edge-bounded rather than missing data (windows.py's
        # measured confidence, reused).
        ring_needed = 2 * ((ix1 - ix0) + (iz1 - iz0))
        ring_have = 0
        for pos in wall.inlier_positions:
            lat = _dot(pos, lateral)
            h = _dot(pos, up_u)
            ix = int((lat - lat_min) / SCAN_BUCKET_M)
            iz = int((h - z_min) / SCAN_BUCKET_M)
            if ix0 - 1 <= ix <= ix1 and (iz == iz0 - 1 or iz == iz1):
                ring_have += 1
            elif iz0 - 1 <= iz <= iz1 and (ix == ix0 - 1 or ix == ix1):
                ring_have += 1
        ring_frac = ring_have / max(1, ring_needed)
        confidence = max(0.05, min(1.0, ring_frac))

        fits.append(OpeningFit(
            kind=kind,
            wall_plane_id=wall.plane_id,
            position=center,
            width_m=gap_w,
            height_m=gap_h,
            sill_height_m=sill,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            n_points=ring_have,
            confidence=round(confidence, 3),
        ))

    fits.sort(key=lambda f: (f.position[0], f.position[1], f.position[2]))
    return fits


# ------------------------------------------------------------------
# First-class WorldIR promotion (mission requirement 4)
# ------------------------------------------------------------------


def promote_opening_to_entity(
    fit: OpeningFit, world, entity_id: str, host_wall_id: Optional[str] = None
):
    """Promote one measured opening into a first-class WorldIR entity.

    Writes (same discipline as promotion.promote_component_to_entity):
      - an entity typed by the measured kind ("door" -> DOOR, "window"
        -> WINDOW, "opening" -> the additive OPENING type -- a generic
        measured void keeps its uncertainty in the type system instead
        of being force-labeled);
      - measured facts in custom_properties (width/height/sill/bounds/
        host wall id);
      - PART_OF opening -> host wall plus CONTAINS wall -> opening
        (first-class, traversable from either end).

    Refuses (OpeningPromotionError) when the host wall entity is absent
    -- an opening in a phantom wall is a dangling topology, not
    evidence. Promote the wall first.
    """
    from world_ir.schema_v1 import Entity, EntityType, Provenance, Relationship, RelationshipKind
    from perception.architecture.promotion import _store_entity

    host_id = host_wall_id or f"wall-{fit.wall_plane_id}"
    if host_id not in world.entities:
        raise OpeningPromotionError(
            f"host wall {host_id!r} not in world -- promote the wall "
            "plane first (evidence.promote_planes.promote_plane_to_entity)"
        )
    type_map = {
        "door": EntityType.DOOR,
        "window": EntityType.WINDOW,
        "opening": EntityType.OPENING,
    }
    etype = type_map.get(fit.kind)
    if etype is None:
        raise OpeningPromotionError(
            f"opening kind {fit.kind!r} has no WorldIR entity type"
        )

    ent = Entity(
        id=entity_id,
        type=etype,
        name=f"{fit.kind} {entity_id.rsplit('-', 1)[-1]}",
        custom_properties={
            "kind": fit.kind,
            "host_wall_id": host_id,
            "wall_plane_id": fit.wall_plane_id,
            "position": list(fit.position),
            "width_m": fit.width_m,
            "height_m": fit.height_m,
            "sill_height_m": fit.sill_height_m,
            "bounds_min": list(fit.bounds_min),
            "bounds_max": list(fit.bounds_max),
            "n_support_points": fit.n_points,
        },
        confidence=fit.confidence,
        provenance=Provenance.INFERRED,
    )
    ent.relationships.append(Relationship(
        kind=RelationshipKind.PART_OF,
        target_id=host_id,
        confidence=fit.confidence,
        provenance=Provenance.INFERRED,
        metadata={"derived_from": "opening_in_wall_coverage"},
    ))
    _store_entity(world, ent)
    host = world.entities.get(host_id) if hasattr(world.entities, "get") \
        else world.entities[host_id]
    host.relationships.append(Relationship(
        kind=RelationshipKind.CONTAINS,
        target_id=entity_id,
        confidence=fit.confidence,
        provenance=Provenance.INFERRED,
        metadata={"derived_from": "opening_in_wall_coverage"},
    ))
    return ent


class OpeningPromotionError(ValueError):
    """Opening promotion refused (host wall absent, unknown kind)."""
