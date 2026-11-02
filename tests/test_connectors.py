"""world_ir.connectors: imagined driveway / street / corridor, always flagged as generated."""

from __future__ import annotations

import pytest

from provenance import Provenance
from world_ir.connectors import LABEL, make_connector
from world_ir.schema_v1 import EntityType


def _area(poly, h):
    pts = [(p[h[0]], p[h[1]]) for p in poly]
    return abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))) / 2


def _flagged(e):
    cp = e.custom_properties
    assert cp["generated_by"] == "connector" and cp["label"] == LABEL == "Generated (not captured)"
    assert cp["asserted"] is False and cp["state"] == "ESTIMATED"
    assert e.confidence <= 0.3 and cp["confidence"] <= 0.3 and e.uncertainty.confidence <= 0.3
    assert e.provenance is Provenance.GENERATED
    assert cp["dimensions"]["width_m"]["state"] == "ESTIMATED"


def test_driveway_is_a_flat_road_strip_of_the_given_width():
    c = make_connector("driveway", (0, 0), (10, 0), 3.0)
    (e,) = c.entities
    assert e.type is EntityType.ROAD and e.custom_properties["road_kind"] == "driveway"
    _flagged(e)
    assert _area(e.custom_properties["footprint_polygon_xyz_m"], (0, 2)) == pytest.approx(30.0)
    assert all(p[1] == 0.0 for p in e.custom_properties["footprint_polygon_xyz_m"])  # lies on the ground (Y-up)
    g = c.geometries[0]
    assert e.geometry_ids == [g.id] and g.confidence <= 0.3 and g.bounds_min.z == pytest.approx(-1.5)


def test_street_has_road_two_sidewalks_and_a_named_street():
    c = make_connector("street", (0, 0), (0, 20), 7.0, name="Mill Road", vertical="z")
    by = {e.type: [x for x in c.entities if x.type is e.type] for e in c.entities}
    assert len(by[EntityType.ROAD]) == 1 and len(by[EntityType.SIDEWALK]) == 2 and len(by[EntityType.STREET]) == 1
    for e in c.entities:
        _flagged(e)
    street = by[EntityType.STREET][0]
    assert street.name == "Mill Road" and street.geometry_ids == []
    assert {r.target_id for r in street.relationships} == {e.id for e in c.entities if e is not street}
    road = by[EntityType.ROAD][0].custom_properties
    sw = [s.custom_properties for s in by[EntityType.SIDEWALK]]
    assert all(p[2] == 0.0 for p in road["footprint_polygon_xyz_m"])  # Z-up: ground is z=0
    xs = sorted([(s["bounds_min"][0], s["bounds_max"][0]) for s in sw])
    assert xs[0][1] == pytest.approx(road["bounds_min"][0]) and xs[1][0] == pytest.approx(road["bounds_max"][0])  # kerbs touch
    assert len(c.geometries) == 3  # road + 2 sidewalks; the STREET is a name, not extra surface


def test_corridor_entity_type_and_defaults():
    c = make_connector("corridor", (0, 0), (0, 4))
    (e,) = c.entities
    assert e.type is EntityType.CORRIDOR and e.custom_properties["dimensions"]["width_m"]["value"] == 1.5
    _flagged(e)


@pytest.mark.parametrize("kwargs,match", [
    ({"kind": "bridge"}, "kind"), ({"width_m": 0.4}, "width"), ({"width_m": 61}, "width"),
    ({"to": (0, 0)}, "coincide"), ({"vertical": "x"}, "vertical"), ({"to": (float("nan"), 0)}, "finite"),
])
def test_invalid_inputs_are_rejected(kwargs, match):
    kw = {"kind": "driveway", "from": (0, 0), "to": (5, 0), "width_m": 3.0, "vertical": "y"}
    kw.update(kwargs)
    with pytest.raises(ValueError, match=match):
        make_connector(kw["kind"], kw["from"], kw["to"], kw["width_m"], vertical=kw["vertical"])
