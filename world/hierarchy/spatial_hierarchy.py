"""Hierarchical World Spatial Representation (world/hierarchy/spatial_hierarchy.py).

Defines the multi-scale, tree-structured spatial hierarchy spanning:
  City → District → Block → Building → Storey → Room → Surface / Opening / Object → Object Part.

Provides coordinate frame propagation, transform accumulation, geo-anchoring,
and strict architectural nesting validation.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SpatialHierarchyLevel(str, Enum):
    CITY = "city"
    DISTRICT = "district"
    BLOCK = "block"
    BUILDING = "building"
    STOREY = "storey"
    ROOM = "room"
    SURFACE_OPENING_OBJECT = "surface_opening_object"
    OBJECT_PART = "object_part"


_LEVEL_DEPTH: dict[SpatialHierarchyLevel, int] = {
    SpatialHierarchyLevel.CITY: 0,
    SpatialHierarchyLevel.DISTRICT: 1,
    SpatialHierarchyLevel.BLOCK: 2,
    SpatialHierarchyLevel.BUILDING: 3,
    SpatialHierarchyLevel.STOREY: 4,
    SpatialHierarchyLevel.ROOM: 5,
    SpatialHierarchyLevel.SURFACE_OPENING_OBJECT: 6,
    SpatialHierarchyLevel.OBJECT_PART: 7,
}


def get_hierarchy_level_depth(level: SpatialHierarchyLevel) -> int:
    return _LEVEL_DEPTH[level]


def _identity_4x4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


@dataclass
class SpatialHierarchyNode:
    """A node in the spatial containment tree."""

    node_id: str
    name: str
    level: SpatialHierarchyLevel
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    world_ir_entity_id: str | None = None
    local_transform: list[list[float]] = field(default_factory=_identity_4x4)
    canonical_world_transform: list[list[float]] | None = None
    geo_anchor: dict[str, float] | None = None  # {"lat": float, "lon": float, "alt_m": float}
    bounding_box: dict[str, Any] | None = None  # {"min": [x,y,z], "max": [x,y,z]}
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialHierarchyNode:
        data_copy = copy.deepcopy(data)
        data_copy["level"] = SpatialHierarchyLevel(data_copy["level"])
        return cls(**data_copy)


class SpatialHierarchyTree:
    """Manages the hierarchical tree of spatial nodes."""

    def __init__(self, root_id: str | None = None) -> None:
        self.nodes: dict[str, SpatialHierarchyNode] = {}
        self.root_id: str | None = root_id

    def add_node(self, node: SpatialHierarchyNode) -> None:
        """Adds a node to the hierarchy, linking parent-child relationships."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists in hierarchy")

        if node.parent_id is not None:
            if node.parent_id not in self.nodes:
                raise ValueError(f"Parent node {node.parent_id} does not exist in hierarchy")
            parent = self.nodes[node.parent_id]
            # Verify level depth constraint
            if get_hierarchy_level_depth(node.level) <= get_hierarchy_level_depth(parent.level):
                raise ValueError(
                    f"Child level {node.level.value} (depth {get_hierarchy_level_depth(node.level)}) "
                    f"cannot be placed inside parent level {parent.level.value} (depth {get_hierarchy_level_depth(parent.level)})"
                )
            if node.node_id not in parent.children_ids:
                parent.children_ids.append(node.node_id)
        elif self.root_id is None:
            self.root_id = node.node_id

        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> SpatialHierarchyNode | None:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> list[SpatialHierarchyNode]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_ancestors(self, node_id: str) -> list[SpatialHierarchyNode]:
        """Returns parent, grandparent, ..., root ordered from immediate parent to root."""
        ancestors = []
        curr = self.nodes.get(node_id)
        while curr and curr.parent_id:
            parent = self.nodes.get(curr.parent_id)
            if not parent:
                break
            ancestors.append(parent)
            curr = parent
        return ancestors

    def compute_canonical_transform(self, node_id: str) -> list[list[float]]:
        """Computes accumulated 4x4 matrix from root to this node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        chain = [self.nodes[node_id]]
        curr = self.nodes[node_id]
        while curr.parent_id:
            parent = self.nodes.get(curr.parent_id)
            if not parent:
                break
            chain.append(parent)
            curr = parent

        # chain is [node, parent, grandparent, ..., root]
        # multiply from root down to node: T_world = T_root * T_p1 * ... * T_node
        mat = np.eye(4, dtype=np.float64)
        for n in reversed(chain):
            loc = np.array(n.local_transform, dtype=np.float64)
            mat = mat @ loc

        result = mat.tolist()
        self.nodes[node_id].canonical_world_transform = result
        return result

    def validate_hierarchy(self) -> tuple[bool, list[str]]:
        """Validates acyclicity, child-level monotonic depth, and parent-child consistency."""
        errors: list[str] = []
        visited: dict[str, int] = {}  # 0: visiting, 1: visited

        def dfs(nid: str) -> None:
            visited[nid] = 0
            node = self.nodes[nid]
            for cid in node.children_ids:
                if cid not in self.nodes:
                    errors.append(f"Node {nid} references missing child {cid}")
                    continue
                child = self.nodes[cid]
                if child.parent_id != nid:
                    errors.append(f"Child {cid} has parent_id {child.parent_id}, expected {nid}")
                if get_hierarchy_level_depth(child.level) <= get_hierarchy_level_depth(node.level):
                    errors.append(
                        f"Depth violation: {node.level} ({get_hierarchy_level_depth(node.level)}) "
                        f"-> {child.level} ({get_hierarchy_level_depth(child.level)})"
                    )
                if visited.get(cid) == 0:
                    errors.append(f"Cycle detected involving node {cid}")
                elif cid not in visited:
                    dfs(cid)
            visited[nid] = 1

        for nid in self.nodes:
            if nid not in visited:
                dfs(nid)

        return len(errors) == 0, errors

    def find_by_level(self, level: SpatialHierarchyLevel) -> list[SpatialHierarchyNode]:
        return [n for n in self.nodes.values() if n.level == level]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialHierarchyTree:
        tree = cls(root_id=data.get("root_id"))
        nodes_dict = data.get("nodes", {})
        # First instantiate all nodes without strict parent check to allow out-of-order restoration
        for nid, raw in nodes_dict.items():
            tree.nodes[nid] = SpatialHierarchyNode.from_dict(raw)
        return tree
