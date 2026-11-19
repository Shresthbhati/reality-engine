"""Tests for the Multi-Scale Spatial Hierarchy Tree and Containment."""

import pytest

from world.hierarchy.spatial_hierarchy import (
    SpatialHierarchyLevel,
    SpatialHierarchyNode,
    SpatialHierarchyTree,
    get_hierarchy_level_depth,
)


def test_hierarchy_level_depths_strictly_ordered() -> None:
    levels = [
        SpatialHierarchyLevel.CITY,
        SpatialHierarchyLevel.DISTRICT,
        SpatialHierarchyLevel.BLOCK,
        SpatialHierarchyLevel.BUILDING,
        SpatialHierarchyLevel.STOREY,
        SpatialHierarchyLevel.ROOM,
        SpatialHierarchyLevel.SURFACE_OPENING_OBJECT,
        SpatialHierarchyLevel.OBJECT_PART,
    ]
    depths = [get_hierarchy_level_depth(lvl) for lvl in levels]
    assert depths == list(range(8))


def test_spatial_hierarchy_tree_construction_and_containment() -> None:
    tree = SpatialHierarchyTree()

    building = SpatialHierarchyNode(
        node_id="bld_01",
        name="Tower A",
        level=SpatialHierarchyLevel.BUILDING,
        geo_anchor={"lat": 37.7749, "lon": -122.4194, "alt_m": 15.0},
    )
    tree.add_node(building)

    storey = SpatialHierarchyNode(
        node_id="storey_02",
        name="Level 2",
        level=SpatialHierarchyLevel.STOREY,
        parent_id="bld_01",
    )
    tree.add_node(storey)

    room = SpatialHierarchyNode(
        node_id="room_201",
        name="Conference Room",
        level=SpatialHierarchyLevel.ROOM,
        parent_id="storey_02",
    )
    tree.add_node(room)

    chair = SpatialHierarchyNode(
        node_id="chair_01",
        name="Executive Chair",
        level=SpatialHierarchyLevel.SURFACE_OPENING_OBJECT,
        parent_id="room_201",
    )
    tree.add_node(chair)

    part = SpatialHierarchyNode(
        node_id="chair_01_seat",
        name="Chair Seat",
        level=SpatialHierarchyLevel.OBJECT_PART,
        parent_id="chair_01",
    )
    tree.add_node(part)

    # Validate tree
    valid, errors = tree.validate_hierarchy()
    assert valid is True
    assert errors == []

    # Check children and ancestors
    children = tree.get_children("bld_01")
    assert len(children) == 1
    assert children[0].node_id == "storey_02"

    ancestors = tree.get_ancestors("chair_01_seat")
    assert [a.node_id for a in ancestors] == ["chair_01", "room_201", "storey_02", "bld_01"]


def test_spatial_hierarchy_rejects_inversion() -> None:
    tree = SpatialHierarchyTree()
    room = SpatialHierarchyNode(
        node_id="room_1",
        name="Room",
        level=SpatialHierarchyLevel.ROOM,
    )
    tree.add_node(room)

    # Attempt to put a Building inside a Room
    with pytest.raises(ValueError, match="cannot be placed inside parent level"):
        tree.add_node(
            SpatialHierarchyNode(
                node_id="bld_invalid",
                name="Building",
                level=SpatialHierarchyLevel.BUILDING,
                parent_id="room_1",
            )
        )


def test_spatial_hierarchy_transform_propagation() -> None:
    tree = SpatialHierarchyTree()
    # Root with translation [0, 0, 10]
    t_root = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 10.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    node1 = SpatialHierarchyNode(
        node_id="n1",
        name="Node1",
        level=SpatialHierarchyLevel.BUILDING,
        local_transform=t_root,
    )
    tree.add_node(node1)

    # Child with translation [5, 0, 0]
    t_child = [
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    node2 = SpatialHierarchyNode(
        node_id="n2",
        name="Node2",
        level=SpatialHierarchyLevel.STOREY,
        parent_id="n1",
        local_transform=t_child,
    )
    tree.add_node(node2)

    canonical_mat = tree.compute_canonical_transform("n2")
    # Result should combine translations: x=5, y=0, z=10
    assert canonical_mat[0][3] == 5.0
    assert canonical_mat[1][3] == 0.0
    assert canonical_mat[2][3] == 10.0
