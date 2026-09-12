"""End-to-end test: evidence -> reconstruction (fake backend) -> promote -> WorldIR.

Exercises the full Goal C/D chain against IReconstructionBackend before any
real CV dependency exists, per docs/CAPABILITY_MATRIX.md's next-blocker note.
"""

import pytest

from evidence.session import EvidenceItem, EvidenceKind
from evidence.promote_reconstruction import (
    EmptyReconstructionError,
    promote_reconstruction_to_entity,
    promote_validated_reconstruction_to_entity,
)
from provenance import Provenance, Uncertainty
from reconstruction.backend.fake import FakeReconstructionBackend
from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult
from world_ir import EntityType, WorldIR


def _photo(id_):
    return EvidenceItem(id=id_, kind=EvidenceKind.PHOTO, source_uri=f"file://{id_}.jpg")


def test_fake_backend_refuses_with_fewer_than_two_photos():
    backend = FakeReconstructionBackend()
    result = backend.reconstruct([_photo("p1")])
    assert result.registration_status == "failed"
    assert result.points == []


def test_fake_backend_produces_canned_result_for_matching_evidence():
    point = ReconstructedPoint(position=(1.0, 2.0, 3.0), track_id="t1",
                                source_evidence_ids=["p1", "p2"],
                                uncertainty=Uncertainty(confidence=0.8))
    pose1 = ReconstructedCameraPose(evidence_id="p1", position=(0, 0, 0), rotation=(1, 0, 0, 0))
    pose2 = ReconstructedCameraPose(evidence_id="p2", position=(1, 0, 0), rotation=(1, 0, 0, 0))
    backend = FakeReconstructionBackend(canned_points=[point], canned_poses=[pose1, pose2])

    result = backend.reconstruct([_photo("p1"), _photo("p2")])
    assert result.registration_status == "success"
    assert result.points == [point]
    assert len(result.camera_poses) == 2


def test_fake_backend_reports_failed_when_no_canned_data_matches():
    backend = FakeReconstructionBackend()
    result = backend.reconstruct([_photo("p1"), _photo("p2")])
    assert result.registration_status == "failed"


def test_promote_reconstruction_creates_real_entity_and_pointcloud_geometry():
    point = ReconstructedPoint(position=(1.0, 2.0, 3.0), track_id="t1",
                                source_evidence_ids=["p1", "p2"],
                                uncertainty=Uncertainty(confidence=0.7))
    pose1 = ReconstructedCameraPose(evidence_id="p1", position=(0, 0, 0), rotation=(1, 0, 0, 0))
    pose2 = ReconstructedCameraPose(evidence_id="p2", position=(1, 0, 0), rotation=(1, 0, 0, 0))
    backend = FakeReconstructionBackend(canned_points=[point], canned_poses=[pose1, pose2])
    result = backend.reconstruct([_photo("p1"), _photo("p2")])

    world = WorldIR(id="w1")
    entity = promote_reconstruction_to_entity(result, world, entity_id="wall_recon", entity_type=EntityType.STRUCTURE)

    assert entity.id in world.entities
    assert entity.provenance == Provenance.RECONSTRUCTED
    assert entity.confidence == 0.7
    geometry = world.geometries[entity.geometry_ids[0]]
    assert geometry.vertex_count == 1
    assert geometry.provenance == Provenance.RECONSTRUCTED


def test_promote_reconstruction_refuses_failed_result():
    world = WorldIR(id="w1")
    backend = FakeReconstructionBackend()
    result = backend.reconstruct([_photo("p1")])  # too few photos -> failed
    with pytest.raises(EmptyReconstructionError):
        promote_reconstruction_to_entity(result, world, entity_id="wall_recon")


def test_promoted_reconstruction_roundtrips_through_worldir():
    point = ReconstructedPoint(position=(1.0, 2.0, 3.0), track_id="t1",
                                source_evidence_ids=["p1", "p2"])
    pose1 = ReconstructedCameraPose(evidence_id="p1", position=(0, 0, 0), rotation=(1, 0, 0, 0))
    pose2 = ReconstructedCameraPose(evidence_id="p2", position=(1, 0, 0), rotation=(1, 0, 0, 0))
    backend = FakeReconstructionBackend(canned_points=[point], canned_poses=[pose1, pose2])
    result = backend.reconstruct([_photo("p1"), _photo("p2")])

    world = WorldIR(id="w1")
    promote_reconstruction_to_entity(result, world, entity_id="wall_recon", entity_name="Recon Wall")

    restored = WorldIR.from_dict(world.to_dict())
    assert restored.entities["wall_recon"].name == "Recon Wall"
    assert restored.geometries["geom-wall_recon"].type.value == "pointcloud"


def test_promote_validated_reconstruction_stays_reconstructed_when_reference_agrees():
    point = ReconstructedPoint(position=(1.0, 2.0, 3.0), track_id="t1", source_evidence_ids=["p1", "p2"])
    result = ReconstructionResult(points=[point], camera_poses=[], registration_status="success")
    reference = ReconstructionResult(
        points=[ReconstructedPoint(position=(1.0, 2.0, 3.0001), track_id="r1", source_evidence_ids=["p1", "p2"])],
        camera_poses=[], registration_status="success",
    )

    world = WorldIR(id="w1")
    entity, report = promote_validated_reconstruction_to_entity(
        result, reference, world, entity_id="wall_recon", distance_threshold=0.01,
    )

    assert not report.has_disagreements()
    assert entity.provenance == Provenance.RECONSTRUCTED
    assert world.geometries[entity.geometry_ids[0]].provenance == Provenance.RECONSTRUCTED


def test_promote_validated_reconstruction_downgrades_to_conflict_on_disagreement():
    point = ReconstructedPoint(position=(0.0, 0.0, 0.0), track_id="t1", source_evidence_ids=["p1", "p2"],
                                uncertainty=Uncertainty(confidence=0.9))
    result = ReconstructionResult(points=[point], camera_poses=[], registration_status="success")
    reference = ReconstructionResult(
        points=[ReconstructedPoint(position=(100.0, 100.0, 100.0), track_id="r1", source_evidence_ids=["p1", "p2"])],
        camera_poses=[], registration_status="success",
    )

    world = WorldIR(id="w1")
    entity, report = promote_validated_reconstruction_to_entity(
        result, reference, world, entity_id="wall_recon", distance_threshold=1.0,
    )

    assert report.has_disagreements()
    assert entity.provenance == Provenance.CONFLICT
    assert not entity.is_canonical() if hasattr(entity, "is_canonical") else True
    geometry = world.geometries[entity.geometry_ids[0]]
    assert geometry.provenance == Provenance.CONFLICT
    assert geometry.confidence == 0.0
    assert entity.confidence == 0.0
    assert geometry.observations[0].sensor_type == "reconstruction_validation"
    assert geometry.observations[0].metadata["disagreements"] == 1


def test_promote_validated_reconstruction_refuses_failed_result():
    world = WorldIR(id="w1")
    failed = ReconstructionResult(points=[], camera_poses=[], registration_status="failed")
    reference = ReconstructionResult(
        points=[ReconstructedPoint(position=(0, 0, 0), track_id="r1", source_evidence_ids=["p1"])],
        camera_poses=[], registration_status="success",
    )
    with pytest.raises(EmptyReconstructionError):
        promote_validated_reconstruction_to_entity(failed, reference, world, entity_id="x", distance_threshold=1.0)
