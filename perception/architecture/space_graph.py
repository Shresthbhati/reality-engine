"""Unified Interior Space Graph & Query Engine.

Provides coherent connectivity querying:
    ROOM <-> DOORWAY <-> CORRIDOR <-> ROOM
    LEVEL <-> STAIR <-> LEVEL

Answers questions directly from evidence-backed geometry:
- "What rooms connect to Room A?"
- "Which rooms are reachable from this corridor?"
- "What openings connect these spaces?"
- "Which level is this room on?"
- "Which stair connects these levels?"
- "What is the evidence trail for this space?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from perception.architecture.corridor import CorridorGraph
from perception.architecture.room_graph import BuildingGraph, RoomGraph, RoomOpening


@dataclass
class InteriorSpaceGraph:
    """The unified, queryable interior space and connectivity graph."""

    building_id: str
    levels: Dict[str, dict] = field(default_factory=dict)
    rooms: Dict[str, dict] = field(default_factory=dict)
    corridors: Dict[str, dict] = field(default_factory=dict)
    openings: List[dict] = field(default_factory=list)
    stairs: Dict[str, dict] = field(default_factory=dict)

    # Topological indices
    _room_adjacency: Dict[str, Set[str]] = field(default_factory=dict)
    _corridor_reachability: Dict[str, Set[str]] = field(default_factory=dict)
    _space_level: Dict[str, str] = field(default_factory=dict)
    _level_stairs: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    _pair_openings: Dict[Tuple[str, str], List[dict]] = field(default_factory=dict)

    @classmethod
    def from_building(
        cls,
        building: BuildingGraph,
        rooms: Sequence[RoomGraph],
        corridors: Optional[Sequence[CorridorGraph]] = None,
        stairs: Optional[Sequence[object]] = None,
    ) -> "InteriorSpaceGraph":
        """Construct an InteriorSpaceGraph from building components."""
        graph = cls(building_id=building.building_id)
        rooms_list = list(rooms)
        corridors_list = list(corridors or [])
        stairs_list = list(stairs or [])

        # 1. Index Levels (Storeys)
        for s in building.storeys:
            graph.levels[s.storey_id] = {
                "level_id": s.storey_id,
                "elevation_m": s.floor_height_m,
                "room_ids": list(s.room_ids),
                "corridor_ids": list(s.corridor_ids),
                "stair_ids": list(s.stair_ids),
            }
            for r_id in s.room_ids:
                graph._space_level[r_id] = s.storey_id
            for c_id in s.corridor_ids:
                graph._space_level[c_id] = s.storey_id

        # 2. Index Rooms
        for r in rooms_list:
            r_dict = r.to_dict()
            graph.rooms[r.room_id] = r_dict
            graph._room_adjacency.setdefault(r.room_id, set())
            for adj in r.adjacent_room_ids:
                graph._room_adjacency[r.room_id].add(adj)

        # 3. Index Corridors
        for c in corridors_list:
            c_dict = c.to_dict() if hasattr(c, "to_dict") else dict(c)
            cid = c_dict["corridor_id"]
            graph.corridors[cid] = c_dict
            graph._corridor_reachability.setdefault(cid, set())
            for r_id in c_dict.get("connected_room_ids", []):
                graph._corridor_reachability[cid].add(r_id)
                # Corridors also connect adjacent rooms
                graph._room_adjacency.setdefault(r_id, set()).add(cid)

        # 4. Index Openings & Doorway Connectivity
        all_spaces = list(rooms_list) + list(corridors_list)
        opening_idx = 0
        for sp in all_spaces:
            sp_id = getattr(sp, "room_id", getattr(sp, "corridor_id", ""))
            ops = getattr(sp, "openings", ())
            for op in ops:
                op_dict = op.to_dict() if hasattr(op, "to_dict") else dict(op)
                op_id = f"opening-{opening_idx:04d}"
                opening_idx += 1
                op_info = {
                    "opening_id": op_id,
                    "host_wall_id": op_dict.get("wall_element_id"),
                    "kind": op_dict.get("kind", "doorway"),
                    "width_m": op_dict.get("width_m", 0.9),
                    "height_m": op_dict.get("height_m", 2.1),
                    "sill_height_m": op_dict.get("sill_height_m", 0.0),
                    "source_space_id": sp_id,
                    "connected_space_ids": op_dict.get("connected_space_ids", []),
                }
                graph.openings.append(op_info)

                # Map opening between pairs
                for other_id in op_dict.get("connected_space_ids", []):
                    pair1 = (sp_id, other_id)
                    pair2 = (other_id, sp_id)
                    graph._pair_openings.setdefault(pair1, []).append(op_info)
                    graph._pair_openings.setdefault(pair2, []).append(op_info)
                    graph._room_adjacency.setdefault(sp_id, set()).add(other_id)
                    graph._room_adjacency.setdefault(other_id, set()).add(sp_id)

        # 5. Index Stairs & Vertical Connectivity
        for st in stairs_list:
            st_id = getattr(st, "stair_id", getattr(st, "id", "stair-001"))
            st_dict = st.to_dict() if hasattr(st, "to_dict") else dict(st)
            graph.stairs[st_id] = st_dict

            # Find which levels this stair connects
            connected_levels = []
            for s in building.storeys:
                if st_id in s.stair_ids:
                    connected_levels.append(s.storey_id)
            if len(connected_levels) >= 2:
                for i in range(len(connected_levels) - 1):
                    l1, l2 = connected_levels[i], connected_levels[i + 1]
                    graph._level_stairs.setdefault((l1, l2), []).append(st_id)
                    graph._level_stairs.setdefault((l2, l1), []).append(st_id)

        return graph

    @classmethod
    def from_dict(cls, data: dict) -> "InteriorSpaceGraph":
        """Deserialize an InteriorSpaceGraph from a dictionary representation."""
        graph = cls(building_id=data.get("building_id", "building-default"))
        for lvl in data.get("levels", []):
            lid = lvl.get("level_id", "")
            if lid:
                graph.levels[lid] = lvl
                for rid in lvl.get("room_ids", []):
                    graph._space_level[rid] = lid
                for cid in lvl.get("corridor_ids", []):
                    graph._space_level[cid] = lid
        for rm in data.get("rooms", []):
            rid = rm.get("room_id", "")
            if rid:
                graph.rooms[rid] = rm
                graph._room_adjacency.setdefault(rid, set())
                for adj in rm.get("adjacent_room_ids", []):
                    graph._room_adjacency[rid].add(adj)
        for corr in data.get("corridors", []):
            cid = corr.get("corridor_id", "")
            if cid:
                graph.corridors[cid] = corr
                graph._corridor_reachability.setdefault(cid, set())
                for rid in corr.get("connected_room_ids", []):
                    graph._corridor_reachability[cid].add(rid)
                    graph._room_adjacency.setdefault(rid, set()).add(cid)
        for op in data.get("openings", []):
            graph.openings.append(op)
            src_id = op.get("source_space_id", "")
            for tgt_id in op.get("connected_space_ids", []):
                pair1 = (src_id, tgt_id)
                pair2 = (tgt_id, src_id)
                graph._pair_openings.setdefault(pair1, []).append(op)
                graph._pair_openings.setdefault(pair2, []).append(op)
                if src_id and tgt_id:
                    graph._room_adjacency.setdefault(src_id, set()).add(tgt_id)
                    graph._room_adjacency.setdefault(tgt_id, set()).add(src_id)
        for st in data.get("stairs", []):
            sid = st.get("stair_id", st.get("id", "stair"))
            graph.stairs[sid] = st
            conn_lvls = st.get("connected_level_ids", [])
            if not conn_lvls:
                conn_lvls = [lvl["level_id"] for lvl in data.get("levels", []) if sid in lvl.get("stair_ids", [])]
            if len(conn_lvls) >= 2:
                for i in range(len(conn_lvls) - 1):
                    l1, l2 = conn_lvls[i], conn_lvls[i + 1]
                    graph._level_stairs.setdefault((l1, l2), []).append(sid)
                    graph._level_stairs.setdefault((l2, l1), []).append(sid)
        return graph

    # ---- Query Engine Methods ----

    def rooms_connected_to(self, room_id: str) -> List[str]:
        """Return all room/corridor IDs directly adjacent or connected via doorways to room_id."""
        return sorted(list(self._room_adjacency.get(room_id, set())))

    def rooms_reachable_from_corridor(self, corridor_id: str) -> List[str]:
        """Return all room IDs directly accessible from the given corridor."""
        return sorted(list(self._corridor_reachability.get(corridor_id, set())))

    def openings_connecting(self, space_a_id: str, space_b_id: str) -> List[dict]:
        """Return opening definitions connecting space_a and space_b."""
        return self._pair_openings.get((space_a_id, space_b_id), [])

    def level_of_space(self, space_id: str) -> Optional[dict]:
        """Return the level containing the given room or corridor."""
        lvl_id = self._space_level.get(space_id)
        if lvl_id and lvl_id in self.levels:
            return self.levels[lvl_id]
        return None

    def stairs_connecting_levels(self, level_a_id: str, level_b_id: str) -> List[dict]:
        """Return all staircases connecting level_a and level_b."""
        stair_ids = self._level_stairs.get((level_a_id, level_b_id), [])
        return [self.stairs[sid] for sid in stair_ids if sid in self.stairs]

    def trace_space_evidence(self, space_id: str) -> List[str]:
        """Collect all evidence IDs associated with the space's boundaries and openings."""
        evidence: Set[str] = set()
        sp = self.rooms.get(space_id) or self.corridors.get(space_id)
        if not sp:
            return []
        for op in sp.get("openings", []):
            for ev in op.get("evidence_ids", []):
                evidence.add(ev)
        return sorted(list(evidence))

    def to_dict(self) -> dict:
        """Serialize full interior space graph to dictionary."""
        return {
            "building_id": self.building_id,
            "levels": list(self.levels.values()),
            "rooms": list(self.rooms.values()),
            "corridors": list(self.corridors.values()),
            "openings": self.openings,
            "stairs": list(self.stairs.values()),
            "summary": {
                "level_count": len(self.levels),
                "room_count": len(self.rooms),
                "corridor_count": len(self.corridors),
                "opening_count": len(self.openings),
                "stair_count": len(self.stairs),
            },
        }
