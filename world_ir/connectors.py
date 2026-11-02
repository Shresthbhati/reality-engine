"""Imagined connectors (driveway, street, corridor) between placed worlds.

Everything here is GENERATED, never captured: a placeholder ground shape sized by user-given parameters, flagged
(``generated_by='connector'``, state ESTIMATED, confidence <= 0.3, label "Generated (not captured)", asserted False)
so no consumer can mistake it for a reconstruction. ``world_ir.compose`` drops any of it that overlaps a captured or
imported entity (precedence: captured > imported > generated).

Points are in the Site frame (x = east, z = south), metres. ``vertical`` picks the composed world's up axis so the
strips lie flat on the ground in either convention (Y-up: (x, 0, z); Z-up: (x, z, 0)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from provenance import Provenance, Uncertainty

from .schema_v1 import Entity, EntityType, Geometry, GeometryType, Relationship, RelationshipKind, Vector3
from .statement_state import StatementState

KINDS = ("driveway", "street", "corridor")
DEFAULT_WIDTH_M = {"driveway": 3.0, "street": 7.0, "corridor": 1.5}
MIN_WIDTH_M, MAX_WIDTH_M = 0.5, 60.0
SIDEWALK_WIDTH_M = 1.5
MAX_CONFIDENCE = 0.3
LABEL = "Generated (not captured)"


@dataclass
class Connector:
    kind: str
    entities: list[Entity] = field(default_factory=list)
    geometries: list[Geometry] = field(default_factory=list)


def is_generated(e: Entity) -> bool:
    return bool(e.custom_properties.get("generated_by"))


def _xyz(x: float, z: float, vertical: str) -> list[float]:
    return [x, 0.0, z] if vertical == "y" else [x, z, 0.0]


def _strip(a, b, width, offset=0.0):
    """Rectangle of `width` along a->b, shifted `offset` sideways. Returns 4 (x, z) corners."""
    dx, dz = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dz)
    ux, uz = dx / n, dz / n
    nx, nz = -uz, ux
    h = width / 2.0
    return [((a, b)[end][0] + nx * (offset + s * h), (a, b)[end][1] + nz * (offset + s * h))
            for end, s in ((0, -1), (1, -1), (1, 1), (0, 1))]


def _entity(eid, etype, name, ring, kind, width, vertical, params) -> tuple[Entity, Geometry]:
    poly = [_xyz(x, z, vertical) for x, z in ring]
    lo = [min(p[i] for p in poly) for i in range(3)]
    hi = [max(p[i] for p in poly) for i in range(3)]
    gid = f"geom-{eid}"
    geom = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(*lo), bounds_max=Vector3(*hi),
                    provenance=Provenance.GENERATED, confidence=MAX_CONFIDENCE)
    ent = Entity(
        id=eid, type=etype, name=name, geometry_ids=[gid],
        provenance=Provenance.GENERATED, confidence=MAX_CONFIDENCE,
        uncertainty=Uncertainty(confidence=MAX_CONFIDENCE, note="user-parameterised placeholder, not reconstructed"),
        statement_state=StatementState.PROCEDURAL, semantic_labels=["generated", kind],
        custom_properties={
            "generated_by": "connector", "connector_kind": kind, "label": LABEL, "asserted": False,
            "state": "ESTIMATED", "confidence": MAX_CONFIDENCE,
            "footprint_polygon_xyz_m": poly, "bounds_min": lo, "bounds_max": hi,
            "dimensions": {"width_m": {"value": width, "unit": "m", "state": "ESTIMATED",
                                       "basis": "user-specified parameter, not measured", "uncertainty_m": None}},
            **params,
        },
    )
    return ent, geom


def make_connector(kind: str, from_point_xz, to_point_xz, width_m: float | None = None, *,
                   connector_id: str | None = None, name: str | None = None, vertical: str = "y",
                   sidewalk_width_m: float = SIDEWALK_WIDTH_M) -> Connector:
    if kind not in KINDS:
        raise ValueError(f"connector kind must be one of {KINDS}, got {kind!r}")
    if vertical not in ("y", "z"):
        raise ValueError("vertical must be 'y' or 'z'")
    a, b = tuple(map(float, from_point_xz)), tuple(map(float, to_point_xz))
    if not all(map(math.isfinite, a + b)):
        raise ValueError("connector points must be finite")
    if math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-6:
        raise ValueError("connector endpoints coincide; nothing to connect")
    width = DEFAULT_WIDTH_M[kind] if width_m is None else float(width_m)
    if not (MIN_WIDTH_M <= width <= MAX_WIDTH_M):
        raise ValueError(f"width_m must be within {MIN_WIDTH_M}..{MAX_WIDTH_M}, got {width}")
    cid = connector_id or f"gen/{kind}"
    common = {"from_xz_m": list(a), "to_xz_m": list(b)}
    out = Connector(kind=kind)

    def add(eid, etype, nm, ring, w, **extra):
        e, g = _entity(eid, etype, nm, ring, kind, w, vertical, {**common, **extra})
        out.entities.append(e)
        out.geometries.append(g)
        return e

    if kind == "driveway":
        add(cid, EntityType.ROAD, name or "Driveway (generated)", _strip(a, b, width), width, road_kind="driveway")
    elif kind == "corridor":
        add(cid, EntityType.CORRIDOR, name or "Corridor (generated)", _strip(a, b, width), width)
    else:
        road = add(cid, EntityType.ROAD, (name or "Street") + " carriageway (generated)", _strip(a, b, width), width,
                   road_kind="street")
        parts = [road]
        for side, sgn in (("l", -1), ("r", 1)):
            off = sgn * (width / 2.0 + sidewalk_width_m / 2.0)
            parts.append(add(f"{cid}/sidewalk-{side}", EntityType.SIDEWALK, f"Sidewalk {side.upper()} (generated)",
                             _strip(a, b, sidewalk_width_m, off), sidewalk_width_m))
        st = add(f"{cid}/street", EntityType.STREET, name or "Street (generated)", _strip(a, b, width), width,
                 street_name=name or None)
        st.relationships = [Relationship(RelationshipKind.CONTAINS, p.id, confidence=MAX_CONFIDENCE, provenance=Provenance.GENERATED)
                            for p in parts]
        for p in parts:
            p.relationships.append(Relationship(RelationshipKind.PART_OF, st.id, confidence=MAX_CONFIDENCE, provenance=Provenance.GENERATED))
        # the STREET entity is a named container of the strips, not extra surface: no geometry of its own
        st.geometry_ids = []
        out.geometries.pop()
    return out
