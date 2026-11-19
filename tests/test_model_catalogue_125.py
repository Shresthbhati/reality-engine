"""Tests for the 125-Slot Target Model Ecosystem Catalogue."""


from engine.multiengine.catalogue import (
    ModelDomain,
    get_catalogue,
)


def test_catalogue_has_exactly_125_slots() -> None:
    cat = get_catalogue()
    assert len(cat) == 125


def test_catalogue_domain_distribution() -> None:
    cat = get_catalogue()
    summary = cat.summary()
    domain_counts = summary["domain_counts"]

    assert domain_counts[ModelDomain.GEOMETRY_3D.value] == 15
    assert domain_counts[ModelDomain.DEPTH_METRIC.value] == 15
    assert domain_counts[ModelDomain.DETECTION_RECOGNITION.value] == 20
    assert domain_counts[ModelDomain.SEGMENTATION.value] == 15
    assert domain_counts[ModelDomain.FEATURE_TRACKING.value] == 12
    assert domain_counts[ModelDomain.POINTCLOUD_3D.value] == 12
    assert domain_counts[ModelDomain.ROOM_ARCHITECTURE.value] == 12
    assert domain_counts[ModelDomain.MATERIAL_SURFACE.value] == 8
    assert domain_counts[ModelDomain.MESH_TOPOLOGY.value] == 8
    assert domain_counts[ModelDomain.ENHANCEMENT.value] == 8


def test_catalogue_commercial_viability_filter() -> None:
    cat = get_catalogue()
    commercial = cat.find(commercial_viable_only=True)
    all_entries = cat.find()

    assert len(commercial) > 0
    assert len(commercial) < len(all_entries)

    for entry in commercial:
        assert entry.is_commercial_viable is True
        # Ensure CC-BY-NC is never in commercial viable entries
        assert "CC-BY-NC" not in entry.weights_license


def test_catalogue_entry_retrieval_and_fields() -> None:
    cat = get_catalogue()
    colmap = cat.get("geom-01")
    assert colmap is not None
    assert colmap.architecture_name == "COLMAP"
    assert colmap.domain == ModelDomain.GEOMETRY_3D
    assert colmap.code_license == "BSD-3-Clause"
    assert colmap.is_commercial_viable is True

    dust3r = cat.get("dust3r")
    assert dust3r is not None
    assert dust3r.domain == ModelDomain.GEOMETRY_3D
    assert dust3r.weights_license == "CC-BY-NC-4.0"
    assert dust3r.is_commercial_viable is False
