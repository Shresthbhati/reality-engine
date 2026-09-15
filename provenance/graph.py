"""Queryable provenance graph (P10-01 remainder).

Per-artifact provenance records existed; the ledger's named gap was
the LINKED graph (artifact -> artifact edges) and a query interface.
This module adds both, over the content-addressed ArtifactStore:

- nodes: artifacts keyed by their sha256 digest (immutable by
  construction -- a digest names exactly one byte sequence);
- edges: derived_from (artifact -> artifact), added only;
- queries: parents/children (direct), ancestors/descendants
  (transitive, iterative DFS -- no recursion limit on deep chains),
  and `why_exists` (the ordered producing-stage chain back to
  captures, answering the spec's user question);
- verify_all: every node's digest re-checked against the store's
  actual bytes -- an edited or deleted artifact is DETECTABLE, never
  silently trusted;
- tombstones: deleting a source marks the node; history (the node and
  its edges) is never rewritten;
- cycles: rejected at edge-addition time (a provenance DAG by
  construction; a cycle would make ancestry walks and incremental
  recompilation ill-defined).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from world_ir.artifact_store import ArtifactStore, ArtifactNotFoundError


class ProvenanceGraphError(ValueError):
    """Bad graph operation (unknown node, cycle, ...)."""


class ProvenanceGraph:
    def __init__(self, store: ArtifactStore):
        self._store = store
        # digest -> {"kind", "producer", "stage", "tombstone": Optional[str]}
        self._nodes: Dict[str, Dict] = {}
        # digest -> set of parent digests (derived_from targets)
        self._edges: Dict[str, set] = {}

    # ---- construction ----

    def add_node(self, digest: str, *, kind: str, producer: str, stage: str) -> None:
        if digest in self._nodes:
            return  # immutable: re-adding the same node is a no-op
        self._nodes[digest] = {
            "kind": kind, "producer": producer, "stage": stage,
            "tombstone": None,
        }
        self._edges.setdefault(digest, set())

    def add_edge(self, child: str, parent: str) -> None:
        """Record `child` derived_from `parent`."""
        for digest in (child, parent):
            if digest not in self._nodes:
                raise ProvenanceGraphError(f"unknown artifact node: {digest}")
        if child == parent:
            raise ProvenanceGraphError("self-edge: an artifact cannot derive from itself")
        if parent in self._edges[child]:
            return  # duplicate edge: immutable graph, adding again is a no-op
        # A cycle forms iff the CHILD is already an ancestor of the
        # PARENT (parent would then descend from child while child
        # derives from parent).
        if child in self._ancestors_of(parent):
            raise ProvenanceGraphError(
                f"cycle rejected: {child} is already an ancestor of {parent}"
            )
        self._edges[child].add(parent)

    def tombstone(self, digest: str, *, reason: str) -> None:
        """Mark a node's source deleted. History is preserved: the
        node and its edges stay queryable."""
        if digest not in self._nodes:
            raise ProvenanceGraphError(f"unknown artifact node: {digest}")
        self._nodes[digest]["tombstone"] = reason

    def is_tombstoned(self, digest: str) -> bool:
        return self._nodes[digest]["tombstone"] is not None

    # ---- queries ----

    def parents(self, digest: str) -> List[str]:
        self._require(digest)
        return sorted(self._edges[digest])

    def children(self, digest: str) -> List[str]:
        self._require(digest)
        return sorted(c for c, ps in self._edges.items() if digest in ps)

    def ancestors(self, digest: str) -> set:
        return self._ancestors_of(digest)

    def descendants(self, digest: str) -> set:
        self._require(digest)
        seen: set = set()
        stack = [digest]
        while stack:
            current = stack.pop()
            for child, ps in self._edges.items():
                if current in ps and child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    def why_exists(self, digest: str) -> List[dict]:
        """The ordered producing-stage chain from captures to this
        artifact: the answer to "why does the system believe this
        exists?" Roots-first (longest provenance path), each step
        carrying stage, producer, and digest."""
        chain: List[dict] = []
        self._why_walk(digest, chain, set())
        return [
            {
                "stage": self._nodes[d]["stage"],
                "producer": self._nodes[d]["producer"],
                "artifact": d,
                "tombstoned": self._nodes[d]["tombstone"] is not None,
            }
            for d in chain
        ]

    def _why_walk(self, digest: str, chain: List[dict], seen: set) -> None:
        """Post-order DFS: parents first (roots earliest), then the node
        -- so the reversed chain reads captures -> ... -> artifact."""
        if digest in seen:
            return
        seen.add(digest)
        for parent in sorted(self._edges[digest]):
            self._why_walk(parent, chain, seen)
        chain.append(digest)

    def _ancestors_of(self, digest: str) -> set:
        self._require(digest)
        seen: set = set()
        stack = [digest]
        while stack:
            current = stack.pop()
            for parent in self._edges[current]:
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen

    def _require(self, digest: str) -> None:
        if digest not in self._nodes:
            raise ProvenanceGraphError(f"unknown artifact node: {digest}")

    # ---- integrity ----

    def verify_all(self) -> List[dict]:
        """Re-digest every node's bytes against the store. Returns the
        list of mismatches (empty when everything verifies). An edited
        artifact no longer hashes to its node key; a deleted artifact
        is not found -- both are detections, reported with reasons."""
        failures: List[dict] = []
        for digest, node in self._nodes.items():
            try:
                data = self._store.get(self._store.uri_for(digest))
            except (ArtifactNotFoundError, KeyError):
                failures.append({
                    "artifact": digest,
                    "reason": "artifact missing from store (deleted or lost)",
                })
                continue
            from world_ir.artifact_store import _digest
            if _digest(data) != digest:
                failures.append({
                    "artifact": digest,
                    "reason": "hash mismatch: store bytes no longer match "
                              "the recorded artifact digest (tampered or corrupted)",
                })
        return failures

    # NOTE on `why_exists` ordering: post-order DFS appends a node only
    # after all its parents, so `chain` is already root-first; the
    # caller-facing method reverses nothing. Sorting parents by name
    # keeps the walk deterministic for a node with multiple parents.
