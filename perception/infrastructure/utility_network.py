"""Infrastructure and Utility Network Intelligence (Wave 2, Block 15).

Builds connected topological graphs for MEP (Mechanical, Electrical, Plumbing) infrastructure:
- Electrical: Distribution boards -> Conduits -> Switchboards -> Sockets, Switches, Lighting fixtures.
- Plumbing Water Supply: Mains / Supply pipes -> Valves -> Taps, Showers, Tanks.
- Plumbing Drainage: Sanitary fixtures -> Traps -> Drain pipes -> Manholes / Stacks.
- HVAC & Ventilation: AC / Compressor units -> Supply/Return ducts -> Vents and grilles.
- Fire & Life Safety: Distribution runs -> Sprinklers, Detectors, Extinguishers.

Generates explicit network connectivity in WorldIR (RelationshipKind.CONNECTS and PART_OF),
enabling circuit tracing, supply path queries, and disconnected fixture discovery.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Relationship, RelationshipKind
from world_ir.world_v1 import WorldIR


class UtilityDomain(str, Enum):
    """MEP utility domains."""
    ELECTRICAL = "electrical"
    PLUMBING_SUPPLY = "plumbing_supply"
    PLUMBING_DRAINAGE = "plumbing_drainage"
    HVAC = "hvac"
    FIRE_SAFETY = "fire_safety"
    COMMUNICATIONS = "communications"
    UNKNOWN = "unknown"


class FlowRole(str, Enum):
    """Role of the component within the utility flow path."""
    SOURCE = "source"              # Distribution board, water main intake, AC air handler
    CONDUIT_CARRIER = "conduit"    # Pipe, wire run, ductwork
    CONTROL = "control"            # Switch, valve, thermostat, damper
    TERMINAL = "terminal"          # Socket, tap, drain, light fixture, diffuser
    PASSIVE = "passive"            # Junction box, cleanout, fitting


@dataclass
class UtilityNode:
    """A node in a utility network graph."""
    entity_id: str
    domain: UtilityDomain
    flow_role: FlowRole
    class_id: str
    position: Tuple[float, float, float]
    connections: List[str] = field(default_factory=list)


@dataclass
class UtilityCircuit:
    """A logically connected sub-network (e.g. electrical circuit or plumbing loop)."""
    circuit_id: str
    name: str
    domain: UtilityDomain
    source_entity_id: Optional[str]
    member_entity_ids: List[str] = field(default_factory=list)
    total_length_m: float = 0.0


@dataclass
class UtilityNetworkReport:
    """Comprehensive status of detected utility networks."""
    total_utility_entities: int
    circuits: List[UtilityCircuit] = field(default_factory=list)
    isolated_entities: List[str] = field(default_factory=list)
    relationships_added: int = 0
    domain_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_utility_entities": self.total_utility_entities,
            "circuit_count": len(self.circuits),
            "isolated_entity_count": len(self.isolated_entities),
            "relationships_added": self.relationships_added,
            "domain_counts": dict(self.domain_counts),
            "circuits": [
                {
                    "circuit_id": c.circuit_id,
                    "name": c.name,
                    "domain": c.domain.value,
                    "source_id": c.source_entity_id,
                    "member_count": len(c.member_entity_ids),
                    "total_length_m": round(c.total_length_m, 2),
                }
                for c in self.circuits
            ],
        }


def _get_entity_coords(world: WorldIR, entity: Entity) -> Tuple[float, float, float]:
    """Extract 3D coordinate vector for an entity."""
    if entity.geometry_ids:
        for gid in entity.geometry_ids:
            if gid in world.geometries:
                g = world.geometries[gid]
                if g.bounds_min and g.bounds_max:
                    return (
                        (g.bounds_min.x + g.bounds_max.x) / 2.0,
                        (g.bounds_min.y + g.bounds_max.y) / 2.0,
                        (g.bounds_min.z + g.bounds_max.z) / 2.0,
                    )
    pos = entity.custom_properties.get("position")
    if pos and isinstance(pos, dict):
        return (float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0)))
    return (0.0, 0.0, 0.0)


def _classify_utility_domain_and_role(class_id: str) -> Tuple[UtilityDomain, FlowRole]:
    """Map taxonomy class id to utility domain and flow role."""
    cid = class_id.lower()

    if cid.startswith("electrical"):
        if "distribution" in cid or "panel" in cid:
            return UtilityDomain.ELECTRICAL, FlowRole.SOURCE
        elif "conduit" in cid or "wiring" in cid:
            return UtilityDomain.ELECTRICAL, FlowRole.CONDUIT_CARRIER
        elif "switch" in cid or "dimmer" in cid:
            return UtilityDomain.ELECTRICAL, FlowRole.CONTROL
        else:
            return UtilityDomain.ELECTRICAL, FlowRole.TERMINAL

    elif cid.startswith("plumbing"):
        if "drain" in cid or "toilet" in cid:
            return UtilityDomain.PLUMBING_DRAINAGE, FlowRole.TERMINAL
        elif "tap" in cid or "sink" in cid or "bath" in cid or "shower" in cid:
            return UtilityDomain.PLUMBING_SUPPLY, FlowRole.TERMINAL
        elif "valve" in cid:
            return UtilityDomain.PLUMBING_SUPPLY, FlowRole.CONTROL
        elif "pipe" in cid:
            return UtilityDomain.PLUMBING_SUPPLY, FlowRole.CONDUIT_CARRIER
        else:
            return UtilityDomain.PLUMBING_SUPPLY, FlowRole.PASSIVE

    elif cid.startswith("hvac"):
        if "ac_unit" in cid or "radiator" in cid:
            return UtilityDomain.HVAC, FlowRole.SOURCE
        elif "duct" in cid:
            return UtilityDomain.HVAC, FlowRole.CONDUIT_CARRIER
        elif "thermostat" in cid:
            return UtilityDomain.HVAC, FlowRole.CONTROL
        else:
            return UtilityDomain.HVAC, FlowRole.TERMINAL

    elif "fire" in cid or "sprinkler" in cid:
        return UtilityDomain.FIRE_SAFETY, FlowRole.TERMINAL

    return UtilityDomain.UNKNOWN, FlowRole.PASSIVE


def build_utility_networks(
    world: WorldIR,
    max_connection_distance_m: float = 6.0,
) -> UtilityNetworkReport:
    """Build and link utility network graphs across all MEP entities in the world."""
    nodes: Dict[str, UtilityNode] = {}
    domain_counts: Dict[str, int] = defaultdict(int)

    # 1. Identify and categorize utility entities
    for ent in world.entities.values():
        c_class = str(ent.custom_properties.get("entity_class", "")).lower()
        e_type = str(ent.type).lower() if hasattr(ent, "type") else ""

        is_utility = (
            c_class.startswith("electrical.")
            or c_class.startswith("plumbing.")
            or c_class.startswith("hvac.")
            or "conduit" in c_class
            or "pipe" in c_class
            or "duct" in c_class
            or "switchboard" in c_class
        )
        if not is_utility:
            continue

        domain, role = _classify_utility_domain_and_role(c_class)
        coords = _get_entity_coords(world, ent)

        nodes[ent.id] = UtilityNode(
            entity_id=ent.id,
            domain=domain,
            flow_role=role,
            class_id=c_class,
            position=coords,
        )
        domain_counts[domain.value] += 1

    # 2. Group nodes by domain
    nodes_by_domain: Dict[UtilityDomain, List[UtilityNode]] = defaultdict(list)
    for n in nodes.values():
        nodes_by_domain[n.domain].append(n)

    # 3. Infer connections (CONNECTS relationships) using domain proximity and flow hierarchy
    relationships_added = 0
    circuits: List[UtilityCircuit] = []

    for domain, domain_nodes in nodes_by_domain.items():
        if domain == UtilityDomain.UNKNOWN or not domain_nodes:
            continue

        sources = [n for n in domain_nodes if n.flow_role == FlowRole.SOURCE]
        carriers = [n for n in domain_nodes if n.flow_role == FlowRole.CONDUIT_CARRIER]
        terminals = [n for n in domain_nodes if n.flow_role in (FlowRole.TERMINAL, FlowRole.CONTROL)]

        # If no explicit source exists, select most central or upstream carrier/distribution element
        if not sources and carriers:
            sources = [carriers[0]]

        # Adjacency list for graph clustering
        adj: Dict[str, Set[str]] = defaultdict(set)

        # Connect carriers to nearest terminals within threshold
        all_dnodes = domain_nodes
        for i, n1 in enumerate(all_dnodes):
            for j, n2 in enumerate(all_dnodes):
                if j <= i:
                    continue
                # Calculate Euclidean distance
                p1 = np.array(n1.position)
                p2 = np.array(n2.position)
                dist = float(np.linalg.norm(p1 - p2))

                # Allow connection if within reach
                if dist <= max_connection_distance_m:
                    r1, r2 = n1.flow_role, n2.flow_role
                    # Allow connections between conduits, sources, controls, and terminals symmetrically
                    valid_connection = (
                        {r1, r2} <= {FlowRole.SOURCE, FlowRole.CONDUIT_CARRIER}
                        or {r1, r2} <= {FlowRole.CONDUIT_CARRIER, FlowRole.TERMINAL}
                        or {r1, r2} <= {FlowRole.CONDUIT_CARRIER, FlowRole.CONTROL}
                        or (r1 == FlowRole.CONDUIT_CARRIER and r2 == FlowRole.CONDUIT_CARRIER)
                        or (dist <= 2.5 and {r1, r2} <= {FlowRole.CONTROL, FlowRole.TERMINAL})
                    )
                    if valid_connection:
                        adj[n1.entity_id].add(n2.entity_id)
                        adj[n2.entity_id].add(n1.entity_id)
                        n1.connections.append(n2.entity_id)
                        n2.connections.append(n1.entity_id)

                        # Add RelationshipKind.CONNECTS to WorldIR entities
                        e1 = world.entities[n1.entity_id]
                        e2 = world.entities[n2.entity_id]
                        e1.relationships.append(Relationship(
                            kind=RelationshipKind.CONNECTS,
                            target_id=e2.id,
                            confidence=0.80,
                            provenance=Provenance.INFERRED,
                            metadata={"utility_domain": domain.value, "distance_m": round(dist, 3)},
                        ))
                        relationships_added += 1

        # Find connected components (circuits)
        visited: Set[str] = set()
        c_idx = 1
        for n in domain_nodes:
            if n.entity_id in visited:
                continue
            # BFS traverse component
            comp: List[str] = []
            queue = deque([n.entity_id])
            visited.add(n.entity_id)
            total_len = 0.0

            while queue:
                curr = queue.popleft()
                comp.append(curr)
                curr_pos = np.array(nodes[curr].position)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        total_len += float(np.linalg.norm(curr_pos - np.array(nodes[neighbor].position)))

            # Identify circuit source
            src_id = None
            for eid in comp:
                if nodes[eid].flow_role == FlowRole.SOURCE:
                    src_id = eid
                    break

            circuits.append(UtilityCircuit(
                circuit_id=f"ckt-{domain.value[:4]}-{c_idx:02d}",
                name=f"{domain.value.title()} Network Branch {c_idx}",
                domain=domain,
                source_entity_id=src_id,
                member_entity_ids=comp,
                total_length_m=total_len,
            ))
            c_idx += 1

    # Find isolated entities
    isolated = [nid for nid, node in nodes.items() if not node.connections]

    report = UtilityNetworkReport(
        total_utility_entities=len(nodes),
        circuits=circuits,
        isolated_entities=isolated,
        relationships_added=relationships_added,
        domain_counts=dict(domain_counts),
    )

    if world.metadata is None:
        world.metadata = {}
    world.metadata["utility_network"] = report.to_dict()

    return report
