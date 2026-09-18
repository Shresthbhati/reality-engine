"""WorldIR 2.0 additive extensions (P9-01 open items: topology,
temporal history, external references).

Position: the v1 canonical schema (schema_v1.py) already carries
Relationship/TemporalEvent/Provenance per entity. What the ledger
records as missing is the WORLD-LEVEL structure:

  - TopologyGraph: containment + connectivity DERIVED from the
    entities' existing relationships (never stored separately, never
    invented -- a relationship naming a missing entity is reported,
    not dropped and not fabricated into a node).
  - TemporalHistory: the world's ordered event log (the entity-level
    TemporalEvents aggregated and ORDERED, with exact range and
    per-entity queries). The log adds no provenance of its own; events
    carry theirs.
  - ExternalReference: a TYPED link from a world statement to a
    resource the world does not own (image file, COLMAP model,
    external report). The checksum is RECORDED, never computed here --
    this module has no file access by design.

Additive contract (constitution, P9-01): v1 dicts without the new
fields load unchanged; every new field defaults to empty/None; nothing
previously serialized changes meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .entity import Entity
from provenance import Provenance
from .schema_v1 import TemporalEvent

#: Relationship kinds interpreted as containment (parent side is the
#: target). Extend only by ADDING a kind here -- never by redefining.
PART_OF_KINDS = frozenset({"part_of"})

#: Relationship kinds interpreted as connectivity seams: an entity
#: (typically a wall/door/floor) adjoining two or more spaces makes
#: those spaces neighbors.
ADJACENCY_KINDS = frozenset({"adjoins", "connects"})

#: Accepted checksum shapes: "<algorithm>:<hex>" per the repo's
#: artifact-store convention.
_CHECKSUM_ALGOS = ("sha256", "sha1", "md5", "blake3")


@dataclass(frozen=True)
class ExternalReference:
    """A typed link to a resource OUTSIDE the world (the world points
    at evidence it does not own and does not load lazily)."""

    kind: str            # e.g. "image", "colmap_model", "report"
    uri: str
    role: str            # e.g. "supporting_evidence"
    checksum: Optional[str] = None  # "<algo>:<hex>", recorded never computed
    provenance: Provenance = Provenance.UNKNOWN

    def __post_init__(self):
        if not self.kind or not self.uri or not self.role:
            raise ValueError(
                "ExternalReference requires kind, uri and role -- an "
                "untyped link cannot be traced"
            )
        if self.checksum is not None:
            algo, _, _digest = self.checksum.partition(":")
            if algo not in _CHECKSUM_ALGOS or not _digest:
                raise ValueError(
                    f"checksum must be '<algo>:<hex>' with algo in "
                    f"{_CHECKSUM_ALGOS}; got {self.checksum!r}"
                )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "uri": self.uri,
            "role": self.role,
            "checksum": self.checksum,
            "provenance": self.provenance.value,
        }

    @staticmethod
    def from_dict(data: dict) -> "ExternalReference":
        raw_prov = data.get("provenance", Provenance.UNKNOWN.value)
        return ExternalReference(
            kind=data["kind"],
            uri=data["uri"],
            role=data["role"],
            checksum=data.get("checksum"),
            provenance=Provenance(raw_prov),
        )


@dataclass(frozen=True)
class TopologyGraph:
    """Containment + connectivity derived from relationship evidence.

    `reported_dangling` carries (entity_id, kind, missing_target_id)
    triples -- evidence problems are reported, never healed silently.
    """

    #: entity_id -> parent entity_id (containment only).
    parents: Dict[str, str] = field(default_factory=dict)
    #: entity_id -> sorted neighbor entity_ids (connectivity only).
    neighbors: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    reported_dangling: Tuple[Tuple[str, str, str], ...] = ()
    #: every entity id seen in the source registry (query validation).
    known_ids: frozenset = frozenset()

    @staticmethod
    def from_registry(entities: Sequence[Entity]) -> "TopologyGraph":
        ids = {e.id for e in entities}
        parents: Dict[str, str] = {}
        adjacency: Dict[str, set] = {}
        dangling: List[Tuple[str, str, str]] = []

        for e in entities:
            adjacency_targets = []
            for rel in e.relationships:
                if rel.target_id not in ids:
                    dangling.append((e.id, rel.kind, rel.target_id))
                    continue
                if rel.kind in PART_OF_KINDS:
                    # One containment parent per entity: a second
                    # part_of target would make "parent" ambiguous; the
                    # first (deterministic: relationship list order)
                    # wins.
                    if e.id not in parents:
                        parents[e.id] = rel.target_id
                elif rel.kind in ADJACENCY_KINDS:
                    adjacency_targets.append(rel.target_id)
            # Seam semantics: an entity adjoining N >= 2 targets
            # (a wall between two rooms, a door connecting them)
            # connects those targets to EACH OTHER. The seam itself is
            # a leaf unless another seam connects it.
            if len(adjacency_targets) >= 2:
                for a in adjacency_targets:
                    for b in adjacency_targets:
                        if a != b:
                            adjacency.setdefault(a, set()).add(b)

        neighbors = {
            node: tuple(sorted(others - {node}))
            for node, others in adjacency.items()
        }
        return TopologyGraph(
            parents=dict(parents),
            neighbors=neighbors,
            reported_dangling=tuple(dangling),
            known_ids=frozenset(ids),
        )

    def parent_of(self, entity_id: str) -> Optional[str]:
        if entity_id not in self.known_ids:
            raise KeyError(f"unknown entity '{entity_id}' in topology graph")
        return self.parents.get(entity_id)

    def children_of(self, entity_id: str) -> List[str]:
        return sorted(k for k, v in self.parents.items() if v == entity_id)

    def ancestors_of(self, entity_id: str) -> List[str]:
        """Chain from the entity's parent up to the root (nearest
        first). Cycle-safe: a malformed containment cycle stops the
        walk rather than looping forever."""
        chain: List[str] = []
        seen = {entity_id}
        current = self.parents.get(entity_id)
        while current is not None and current not in seen:
            chain.append(current)
            seen.add(current)
            current = self.parents.get(current)
        return chain

    def neighbors_of(self, entity_id: str) -> List[str]:
        return list(self.neighbors.get(entity_id, ()))


@dataclass(frozen=True)
class TemporalHistory:
    """The world's ordered event log. Events are supplied already
    carrying their provenance (schema_v1.TemporalEvent); ordering is
    stable by (timestamp, original arrival order)."""

    events: Tuple[TemporalEvent, ...] = ()

    @staticmethod
    def from_events(events: Sequence[TemporalEvent]) -> "TemporalHistory":
        ordered = sorted(
            enumerate(events), key=lambda pair: (pair[1].timestamp, pair[0])
        )
        return TemporalHistory(events=tuple(e for _, e in ordered))

    def events_in_range(self, start: float, end: float) -> List[TemporalEvent]:
        """Inclusive on both ends; O(log n) via bisect on the ordered
        timestamps."""
        import bisect

        stamps = [e.timestamp for e in self.events]
        lo = bisect.bisect_left(stamps, start)
        hi = bisect.bisect_right(stamps, end)
        return list(self.events[lo:hi])

    def events_for(self, entity_id: str) -> List[TemporalEvent]:
        return [e for e in self.events if e.entity_id == entity_id]

    def to_dict(self) -> list:
        return [e.to_dict() for e in self.events]

    @staticmethod
    def from_dict(data: list) -> "TemporalHistory":
        return TemporalHistory.from_events(
            [TemporalEvent.from_dict(e) for e in data]
        )
