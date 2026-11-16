"""B15: relations from boxes via a uniform grid; unseen interiors stay not_observed."""
from world_ir.relations import RELATION_KINDS, Box, derive_relations, interior_state


def test_cup_on_table():
    table = Box("table", (0, 0, 0), (1.5, 1, 0.75)); cup = Box("cup", (0.5, 0.4, 0.75), (0.6, 0.5, 0.85))
    assert ("cup", "on", "table") in derive_relations([table, cup])


def test_item_inside_an_observed_container_gets_contains():
    cab = Box("cab", (0, 0, 0), (1, 1, 1), is_container=True, interior_observed=True)
    item = Box("item", (0.2, 0.2, 0.2), (0.4, 0.4, 0.4))
    rel = derive_relations([cab, item])
    assert ("item", "inside", "cab") in rel and ("cab", "contains", "item") in rel


def test_unobserved_interior_has_no_contains_and_is_marked_not_observed():
    cab = Box("cab", (0, 0, 0), (1, 1, 1), is_container=True)
    item = Box("item", (0.2, 0.2, 0.2), (0.4, 0.4, 0.4))
    assert not any(k == "contains" for _, k, _ in derive_relations([cab, item]))
    assert interior_state(cab) == "not_observed"


def test_wall_attachment_and_distant_things_are_unrelated():
    wall = Box("wall", (0, 0, 0), (4, 0.1, 3), is_wall=True); sw = Box("sw", (1, 0.1, 1.2), (1.1, 0.13, 1.3))
    far = Box("far", (10, 10, 0), (11, 11, 1))
    rel = derive_relations([wall, sw, far])
    assert ("sw", "attached_to", "wall") in rel and not any("far" in (a, b) for a, _, b in rel)


def test_every_emitted_kind_is_in_the_closed_set():
    boxes = [Box("a", (0, 0, 0), (1, 1, 1)), Box("b", (1.05, 0, 0), (2, 1, 1))]
    assert all(k in RELATION_KINDS for _, k, _ in derive_relations(boxes))
