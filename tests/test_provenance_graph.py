"""Tests for provenance/graph.py (P10-01 remainder): the queryable
linked provenance graph over content-addressed artifacts.

Spec (docs/future/provenance/PROVENANCE_GRAPH.md): nodes are artifacts
(by hash), edges are derived_from; deleting a source tombstones, never
rewrites history; the graph must answer "why does the system believe
this exists" by walking to origins. Tamper detection: every node's
hash must still match the artifact store's bytes -- an edited artifact
is detectable, not silently trusted.
"""

import pytest

from world_ir.artifact_store import MemoryArtifactStore
from provenance.graph import (
    ProvenanceGraph,
    ProvenanceGraphError,
)


@pytest.fixture()
def store():
    return MemoryArtifactStore()


@pytest.fixture()
def compiled_fixture(store):
    """A compiled-style chain: raw capture bytes -> dense cloud ->
    mesh -> world version, each derived from the previous."""
    g = ProvenanceGraph(store)
    # put() returns (uri, digest); nodes are keyed by DIGEST.
    _, captures = store.put(b"raw-capture-bytes-img1")
    _, captures2 = store.put(b"raw-capture-bytes-img2")
    _, dense = store.put(b"dense-pointcloud-ply-bytes")
    _, mesh = store.put(b"mesh-ply-bytes")
    _, world = store.put(b"worldir-v1-serialized")
    g.add_node(captures, kind="capture", producer="sensor", stage="ingest")
    g.add_node(captures2, kind="capture", producer="sensor", stage="ingest")
    g.add_node(dense, kind="pointcloud", producer="colmap", stage="dense_mvs")
    g.add_node(mesh, kind="mesh", producer="poisson", stage="meshing")
    g.add_node(world, kind="world_version", producer="reality-compile", stage="compile")
    g.add_edge(dense, captures)
    g.add_edge(dense, captures2)
    g.add_edge(mesh, dense)
    g.add_edge(world, mesh)
    return g, {
        "captures": captures, "captures2": captures2,
        "dense": dense, "mesh": mesh, "world": world,
    }


class TestAncestryWalk:
    def test_full_ancestor_chain_from_world_version(self, compiled_fixture):
        g, ids = compiled_fixture
        ancestors = g.ancestors(ids["world"])
        # image -> feature/match -> pose -> depth -> point -> mesh ->
        # world version, compressed here to captures -> dense -> mesh
        # -> world: the FULL transitive closure must be reachable.
        assert ids["mesh"] in ancestors
        assert ids["dense"] in ancestors
        assert ids["captures"] in ancestors
        assert ids["captures2"] in ancestors
        # The node itself is not its own ancestor.
        assert ids["world"] not in ancestors

    def test_direct_parents_vs_transitive(self, compiled_fixture):
        g, ids = compiled_fixture
        parents = g.parents(ids["world"])
        assert parents == [ids["mesh"]]
        assert ids["captures"] not in parents

    def test_descendants_walk(self, compiled_fixture):
        g, ids = compiled_fixture
        descendants = g.descendants(ids["dense"])
        assert ids["mesh"] in descendants
        assert ids["world"] in descendants
        # Captures are upstream, not downstream.
        assert ids["captures"] not in descendants

    def test_why_exists_reports_stage_chain(self, compiled_fixture):
        # The spec's user question: "why does the system believe this
        # exists?" -- answer = the ordered producing stages back to
        # captures.
        g, ids = compiled_fixture
        chain = g.why_exists(ids["mesh"])
        stages = [step["stage"] for step in chain]
        # Two capture nodes feed the dense cloud, so "ingest" appears
        # once per capture ancestor, root-first, then the producers.
        assert stages == ["ingest", "ingest", "dense_mvs", "meshing"]

    def test_unknown_node_raises(self, compiled_fixture):
        g, _ = compiled_fixture
        with pytest.raises(ProvenanceGraphError, match="unknown"):
            g.ancestors("sha256-deadbeef")


class TestTamperDetection:
    def test_unmodified_artifacts_verify(self, compiled_fixture):
        g, _ = compiled_fixture
        assert g.verify_all() == []

    def test_edited_artifact_is_detected(self, store, compiled_fixture):
        g, ids = compiled_fixture
        # Overwrite the mesh artifact's bytes in the store (tampering):
        # same URI, different content.
        uri = store.uri_for(ids["mesh"])
        store._blobs[ids["mesh"]] = b"TAMPERED-mesh-ply-bytes"
        failures = g.verify_all()
        assert len(failures) == 1
        assert failures[0]["artifact"] == ids["mesh"]
        assert "hash mismatch" in failures[0]["reason"]

    def test_deleted_artifact_is_detected(self, store, compiled_fixture):
        g, ids = compiled_fixture
        del store._blobs[ids["dense"]]
        failures = g.verify_all()
        assert any(f["artifact"] == ids["dense"] for f in failures)


class TestImmutability:
    def test_duplicate_edge_is_noop(self, compiled_fixture):
        g, ids = compiled_fixture
        before = len(g.parents(ids["world"]))
        g.add_edge(ids["world"], ids["mesh"])
        assert len(g.parents(ids["world"])) == before

    def test_tombstone_preserves_history(self, compiled_fixture):
        # Deleting a source tombstones the node -- the node and its
        # edges remain queryable (history is never rewritten), with
        # the tombstone recorded.
        g, ids = compiled_fixture
        g.tombstone(ids["captures2"], reason="source file deleted by user")
        assert ids["captures2"] in g.ancestors(ids["world"])
        assert g.is_tombstoned(ids["captures2"])
        assert not g.is_tombstoned(ids["captures"])

    def test_cycles_rejected(self, store):
        g = ProvenanceGraph(store)
        a, _ = store.put(b"a")
        b, _ = store.put(b"b")
        g.add_node(a, kind="artifact", producer="p", stage="s")
        g.add_node(b, kind="artifact", producer="p", stage="s")
        g.add_edge(b, a)  # b derived from a
        with pytest.raises(ProvenanceGraphError, match="cycle"):
            g.add_edge(a, b)  # would make a descend from its own parent
