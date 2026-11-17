"""B03: one ontology, with parts and attributes (docs/BUILD_SPEC.md section 3, B03)."""

from __future__ import annotations

import dataclasses

import pytest

from perception.entities import taxonomy
from perception.ontology import Ontology, OntologyNode, get_default_ontology

# Every id taxonomy.all_classes() returned before B03 (frozen so the shim cannot hide a lost class).
OLD_TAXONOMY_IDS = (
    "building", "building.storey", "room", "room.corridor", "room.kitchen", "room.bathroom", "room.bedroom",
    "room.living", "envelope.wall", "envelope.floor", "envelope.ceiling", "envelope.skirting",
    "envelope.cornice", "opening.door", "opening.door.leaf", "opening.door.frame", "opening.door.handle",
    "opening.door.hinge", "opening.door.lock", "opening.window", "opening.window.frame",
    "opening.window.glazing", "opening.window.mullion", "opening.window.handle", "opening.opening",
    "structural.column", "structural.beam", "structural.arch", "circulation.stairs",
    "circulation.stairs.tread", "circulation.stairs.riser", "circulation.stairs.stringer",
    "circulation.stairs.landing", "circulation.railing", "circulation.railing.post",
    "circulation.railing.glass", "electrical.switchboard", "electrical.switchboard.module",
    "electrical.switchboard.module.socket", "electrical.switchboard.module.switch",
    "electrical.switchboard.module.dimmer", "electrical.switchboard.module.usb",
    "electrical.switchboard.module.data", "electrical.switchboard.module.blank", "electrical.junction_box",
    "electrical.distribution_board", "electrical.conduit", "electrical.wiring", "electrical.light_fitting",
    "electrical.fan", "plumbing.pipe", "plumbing.tap", "plumbing.sink", "plumbing.toilet", "plumbing.bath",
    "plumbing.drain", "hvac.vent", "hvac.ac_unit", "hvac.radiator", "hvac.duct", "hvac.thermostat",
    "furniture.sofa", "furniture.chair", "furniture.table", "furniture.bed", "furniture.wardrobe",
    "furniture.shelf", "furniture.tv_unit", "furniture.desk", "furniture.rug", "furniture.tv",
    "furniture.refrigerator", "furniture.oven", "furniture.washing_machine", "furniture.microwave",
    "finish.curtain", "finish.picture", "finish.mirror", "finish.tiling", "finish.moulding",
    "exterior.facade", "exterior.terrain", "exterior.roof", "exterior.roof.plane", "exterior.balcony",
    "exterior.gate", "exterior.fence", "exterior.chimney", "exterior.gutter", "road.carriageway",
    "road.footpath", "road.kerb", "road.marking", "road.drain", "vegetation.tree", "vegetation.tree.trunk",
    "vegetation.tree.canopy", "vegetation.shrub", "vegetation.plant", "vegetation.flowerbed",
    "vegetation.lawn", "street.lamppost", "street.lamppost.shaft", "street.lamppost.fixture", "street.bench",
    "street.bin", "street.postbox", "street.barricade", "utility.meter", "utility.transformer",
    "utility.hydrant", "utility.cable_run", "signage.sign", "signage.pole", "signage.traffic_light",
    "misc.object", "misc.unknown",
)

SOCKET = "electrical.switchboard.module.socket"


@pytest.fixture(scope="module")
def onto() -> Ontology:
    return get_default_ontology()


def _node(path: str, **kw) -> OntologyNode:
    return OntologyNode(path=path, label=path, prompts=("x",), **kw)


def test_default_ontology_validates_clean(onto):
    assert onto.validate() == []


def test_alias_wardrobe_and_almirah_same_path(onto):
    assert onto.alias_to_path["wardrobe"] == onto.alias_to_path["almirah"]
    assert onto.resolve_or_unknown("Almirah").path == onto.resolve_or_unknown("wardrobe").path


def test_alias_keys_are_lowercase_and_space_collapsed(onto):
    assert onto.alias_to_path["chest of drawers"] == "furniture/storage/chest_of_drawers"
    assert all(k == " ".join(k.lower().split()) for k in onto.alias_to_path)


def test_old_taxonomy_count_frozen():
    assert len(OLD_TAXONOMY_IDS) == 117 and len(set(OLD_TAXONOMY_IDS)) == 117


@pytest.mark.parametrize("old_id", OLD_TAXONOMY_IDS)
def test_every_old_taxonomy_id_still_resolves(onto, old_id):
    node = onto.resolve_or_unknown(old_id)
    assert node.path == old_id.replace(".", "/")
    assert taxonomy.get(old_id).id == old_id


def test_taxonomy_all_classes_is_exactly_the_old_set():
    assert [c.id for c in taxonomy.all_classes()] == list(OLD_TAXONOMY_IDS)


def test_taxonomy_shim_reads_the_ontology(onto):
    cls = taxonomy.get(SOCKET)
    node = onto.get(SOCKET.replace(".", "/"))
    assert cls.label == node.label and cls.prompts == node.prompts and cls.small and node.small
    assert cls.size_m == node.size_m == (0.03, 0.12)
    assert cls.hosts == ("electrical.switchboard",)
    assert taxonomy.ancestors(SOCKET) == ["electrical.switchboard.module", "electrical.switchboard"]
    assert [c.id for c in taxonomy.children("electrical.switchboard")] == ["electrical.switchboard.module"]
    assert taxonomy.class_for_label("Wall Socket").id == SOCKET
    assert taxonomy.class_for_label("spaceship") is None
    assert taxonomy.size_plausible(SOCKET, 0.08) and not taxonomy.size_plausible(SOCKET, 5)
    with pytest.raises(KeyError):
        taxonomy.get("no.such.class")


def test_ontology_children_and_ancestors(onto):
    assert "furniture/seating/sofa/sectional" in onto.children["furniture/seating/sofa"]
    assert onto.ancestors("electrical/switchboard/module/socket") == [
        "electrical/switchboard/module", "electrical/switchboard"]
    assert onto.ancestors("architecture/wall") == []


def test_every_attribute_list_has_unknown(onto):
    with_attrs = [n for n in onto.nodes.values() if n.attributes]
    assert with_attrs
    for n in with_attrs:
        for name, values in n.attributes.items():
            assert "unknown" in values, (n.path, name)


@pytest.mark.parametrize("path,attr,sample", [
    ("furniture/storage/wardrobe", "handle_type", ("bar", "knob", "recessed", "ring")),
    ("furniture/storage/wardrobe", "lock_type", ("key", "numpad", "padlock", "none")),
    ("furniture/storage/cabinet", "door_count", ("1", "2")),
    ("fixtures/electrical/socket", "socket_standard", ("type_c_euro", "type_d_india_5a", "type_m_india_15a", "rj45_data")),
    ("fixtures/electrical/switch", "switch_kind", ("rocker", "dimmer", "regulator", "bell")),
    ("architecture/window/casement", "pane_count", ("1", "2")),
    ("architecture/door/hinged_door", "swing", ("left", "right")),
    ("architecture/door/hinged_door", "handle_type", ("lever", "knob", "pull")),
    ("architecture/wall", "finish", ("paint", "tile", "stone", "wood", "plaster", "exposed")),
    ("furniture/seating/chair", "leg_count", ("4",)),
    ("electrical/fan", "mount", ("ceiling", "wall", "floor")),
])
def test_section4_attributes(onto, path, attr, sample):
    values = onto.get(path).attributes[attr]
    assert set(sample) <= set(values) and values[-1] == "unknown"


@pytest.mark.parametrize("whole,part", [
    ("furniture/storage/wardrobe", "handle"), ("furniture/storage/wardrobe", "lock"),
    ("furniture/storage/chest_of_drawers", "drawer_front"),
    ("fixtures/electrical/socket", "pin_holes"), ("fixtures/electrical/switch", "rocker"),
    ("architecture/door/sliding_door", "threshold"), ("architecture/door/double_door", "closer"),
    ("architecture/window/fixed", "pane"), ("furniture/sleeping/bed", "leg"),
    ("electrical/fan", "blade"), ("plumbing/tap", "spout"),
])
def test_section4_parts_are_child_nodes(onto, whole, part):
    child = onto.get(f"{whole}/{part}")
    assert child is not None and child.parent == whole
    assert part in onto.get(whole).components


def test_part_aliases_resolve_to_existing_part_nodes(onto):
    assert onto.alias_to_path["baluster"] == "circulation/railing/post"
    assert onto.alias_to_path["step"] == "circulation/stairs/tread"


def test_existing_components_still_short_names(onto):
    assert "seat_cushion" in onto.get("furniture/seating/sofa/sectional").components


def test_validate_flags_missing_component_attach_and_cycle():
    bad = Ontology(version="t", nodes={
        "fallback/unknown_object": _node("fallback/unknown_object"),
        "a/x": _node("a/x", components=("nope",), attach=({"hosted_by": "a/missing"},)),
        "a/c1": _node("a/c1", parent="a/c2"),
        "a/c2": _node("a/c2", parent="a/c1"),
    })
    issues = " | ".join(bad.validate())
    assert "a/x" in issues and "nope" in issues and "a/missing" in issues
    assert "cycle" in issues.lower()


def test_validate_flags_attribute_without_unknown_and_duplicate_alias():
    bad = Ontology(version="t", nodes={
        "fallback/unknown_object": _node("fallback/unknown_object"),
        "a/x": _node("a/x", attributes={"colour": ("red", "blue")}, aliases=("twin",)),
        "a/y": _node("a/y", aliases=("twin",)),
    })
    issues = " | ".join(bad.validate())
    assert "colour" in issues and "unknown" in issues
    assert "twin" in issues


def test_new_node_fields_have_no_invented_defaults():
    n = _node("a/b")
    assert n.attributes == {} and n.aliases == () and n.world_type is None and n.size_m is None
    assert "world_type" in {f.name for f in dataclasses.fields(OntologyNode)}
