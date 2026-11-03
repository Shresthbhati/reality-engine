"""Tests for Metric Scale Fusion Engine (Universal Intelligence Expansion Epic E02-1).

Validates:
1. Operator baseline constraint extraction and scaling.
2. GNSS great-circle baseline constraint extraction and scaling.
3. Architectural physical entity prior extraction (doors, switchboards, stairs).
4. Robust 1D RANSAC consensus with outlier rejection (<1.5% scale error).
5. Honest degradation to RELATIVE on conflicting / insufficient evidence.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from evidence.session import EvidenceItem, EvidenceKind
from provenance import Provenance, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.scale import (
    MetricFusionDiagnostics,
    MetricScaleFusionEngine,
    ScaleConstraint,
    ScaleConstraintType,
    ScaleReference,
    ScaleState,
)
from world_ir import Entity, EntityType, WorldIR


def _make_synth_recon(true_scale: float = 2.5) -> ReconstructionResult:
    """Synthetic reconstruction in model units where 1 model unit = true_scale meters."""
    # Camera 0 at (0, 0, 1), Camera 1 at (4/true_scale, 0, 1), Camera 2 at (0, 3/true_scale, 1)
    # True metric distances: Cam0-Cam1 = 4.0m, Cam0-Cam2 = 3.0m, Cam1-Cam2 = 5.0m
    poses = [
        ReconstructedCameraPose(
            evidence_id="cam_0",
            position=(0.0, 0.0, 1.0 / true_scale),
            rotation=(1.0, 0.0, 0.0, 0.0),
        ),
        ReconstructedCameraPose(
            evidence_id="cam_1",
            position=(4.0 / true_scale, 0.0, 1.0 / true_scale),
            rotation=(1.0, 0.0, 0.0, 0.0),
        ),
        ReconstructedCameraPose(
            evidence_id="cam_2",
            position=(0.0, 3.0 / true_scale, 1.0 / true_scale),
            rotation=(1.0, 0.0, 0.0, 0.0),
        ),
    ]

    pts = [
        ReconstructedPoint(
            position=(float(x) / true_scale, float(y) / true_scale, 0.0),
            track_id=f"p_{x}_{y}",
            source_evidence_ids=["cam_0", "cam_1", "cam_2"],
        )
        for x in (0, 1, 2, 3, 4)
        for y in (0, 1, 2, 3)
    ]

    return ReconstructionResult(
        points=pts,
        camera_poses=poses,
        registration_status="success",
    )


class TestMetricScaleFusionEngine:
    def test_single_operator_baseline_anchoring(self):
        true_scale = 1.80  # 1.80 meters per model unit
        recon = _make_synth_recon(true_scale)

        # Operator measured distance between cam_0 and cam_1 is 4.0m
        ref = ScaleReference(evidence_id_a="cam_0", evidence_id_b="cam_1", distance_m=4.0)

        engine = MetricScaleFusionEngine()
        scaled, diag = engine.fuse(recon, references=[ref])

        assert diag.consensus_reached is True
        assert diag.scale_state == "metric"
        assert diag.fused_meters_per_unit == pytest.approx(true_scale, rel=1e-4)
        assert diag.estimated_error_pct <= 1.5

        # Check scaled camera positions
        p0 = next(p.position for p in scaled.camera_poses if p.evidence_id == "cam_0")
        p1 = next(p.position for p in scaled.camera_poses if p.evidence_id == "cam_1")
        baseline_m = math.dist(p0, p1)
        assert baseline_m == pytest.approx(4.0, rel=1e-4)

    def test_gnss_baseline_fusion(self):
        true_scale = 2.0
        recon = _make_synth_recon(true_scale)

        lat0, lon0 = 37.774900, -122.419400
        # Cam1 is approx 4.0 meters east
        delta_lon = 4.0 / (6371000.0 * math.cos(math.radians(lat0)) * (math.pi / 180.0))
        lat1, lon1 = lat0, lon0 + delta_lon
        # Cam2 is approx 3.0 meters north
        delta_lat = 3.0 / (6371000.0 * (math.pi / 180.0))
        lat2, lon2 = lat0 + delta_lat, lon0

        items = [
            EvidenceItem(
                id="cam_0",
                kind=EvidenceKind.PHOTO,
                source_uri="/photos/cam_0.jpg",
                metadata={"gps": (lat0, lon0)},
                provenance=Provenance.OBSERVED,
            ),
            EvidenceItem(
                id="cam_1",
                kind=EvidenceKind.PHOTO,
                source_uri="/photos/cam_1.jpg",
                metadata={"gps": (lat1, lon1)},
                provenance=Provenance.OBSERVED,
            ),
            EvidenceItem(
                id="cam_2",
                kind=EvidenceKind.PHOTO,
                source_uri="/photos/cam_2.jpg",
                metadata={"gps": (lat2, lon2)},
                provenance=Provenance.OBSERVED,
            ),
        ]

        engine = MetricScaleFusionEngine(min_inlier_weight=0.5)
        scaled, diag = engine.fuse(recon, evidence_items=items)

        assert diag.consensus_reached is True
        assert diag.scale_state == "metric"
        assert diag.fused_meters_per_unit == pytest.approx(true_scale, rel=0.015)
        assert diag.estimated_error_pct <= 1.5
        assert diag.inlier_count >= 2

    def test_entity_priors_fusion_doors_and_switchboards(self):
        true_scale = 1.50
        recon = _make_synth_recon(true_scale)

        world = WorldIR(id="w-priors", name="Test Priors")
        # Standard door has nominal height 2.036m
        # In model units, door height is 2.036 / true_scale
        world.entities["door_1"] = Entity(
            id="door_1",
            name="Bedroom Door",
            type=EntityType.DOOR,
            custom_properties={"height_m": 2.036 / true_scale},
            provenance=Provenance.INFERRED,
        )
        world.entities["door_2"] = Entity(
            id="door_2",
            name="Bathroom Door",
            type=EntityType.DOOR,
            custom_properties={"height_m": 2.040 / true_scale},
            provenance=Provenance.INFERRED,
        )
        # Standard switchboard width 0.086m
        world.entities["sw_1"] = Entity(
            id="sw_1",
            name="Light Switchboard",
            type=EntityType.INFRASTRUCTURE,
            custom_properties={"width_m": 0.086 / true_scale},
            provenance=Provenance.INFERRED,
        )

        engine = MetricScaleFusionEngine(min_inlier_weight=0.7)
        scaled, diag = engine.fuse(recon, world=world)

        assert diag.consensus_reached is True
        assert diag.scale_state == "metric"
        assert diag.fused_meters_per_unit == pytest.approx(true_scale, rel=0.015)
        assert diag.estimated_error_pct <= 1.5
        assert diag.inlier_count >= 2

    def test_ransac_consensus_rejects_spurious_outlier(self):
        """Prove RANSAC rejects a 10x corrupted prior and fuses true inliers."""
        true_scale = 2.0
        recon = _make_synth_recon(true_scale)

        world = WorldIR(id="w-outlier", name="Test Outlier")
        # Inlier door 1
        world.entities["door_1"] = Entity(
            id="door_1",
            name="Good Door 1",
            type=EntityType.DOOR,
            custom_properties={"height_m": 2.036 / true_scale},
            provenance=Provenance.INFERRED,
        )
        # Inlier door 2
        world.entities["door_2"] = Entity(
            id="door_2",
            name="Good Door 2",
            type=EntityType.DOOR,
            custom_properties={"height_m": 2.030 / true_scale},
            provenance=Provenance.INFERRED,
        )
        # Spurious corrupted entity with 10x wrong scale
        world.entities["door_fake"] = Entity(
            id="door_fake",
            name="Corrupted Door",
            type=EntityType.DOOR,
            custom_properties={"height_m": 2.036 / (true_scale * 10.0)},
            provenance=Provenance.INFERRED,
        )

        # Operator ground truth reference
        ref = ScaleReference(evidence_id_a="cam_0", evidence_id_b="cam_1", distance_m=4.0)

        engine = MetricScaleFusionEngine()
        scaled, diag = engine.fuse(recon, world=world, references=[ref])

        assert diag.consensus_reached is True
        assert diag.scale_state == "metric"
        assert diag.fused_meters_per_unit == pytest.approx(true_scale, rel=0.015)
        # Check that corrupted entity was excluded from inliers
        assert not any("door_fake" in src for src in diag.inlier_sources)
        assert any("cam_0" in src for src in diag.inlier_sources)

    def test_conflicting_unverified_priors_degrades_to_relative(self):
        """When evidence cannot agree, never fabricate scale; degrade honestly to RELATIVE."""
        recon = _make_synth_recon(1.0)
        world = WorldIR(id="w-conflict", name="Conflicting Priors")
        # Two entities claiming wildly different scales (ratio 0.5 vs ratio 2.0)
        world.entities["ent_a"] = Entity(
            id="ent_a",
            name="Entity A",
            type=EntityType.DOOR,
            custom_properties={"height_m": 4.0},  # implies scale 2.036 / 4 = 0.509
            provenance=Provenance.INFERRED,
        )
        world.entities["ent_b"] = Entity(
            id="ent_b",
            name="Entity B",
            type=EntityType.DOOR,
            custom_properties={"height_m": 1.0},  # implies scale 2.036 / 1 = 2.036
            provenance=Provenance.INFERRED,
        )

        engine = MetricScaleFusionEngine(inlier_tolerance=0.03, min_inlier_weight=1.5)
        scaled, diag = engine.fuse(recon, world=world)

        assert diag.consensus_reached is False
        assert diag.scale_state == "relative"
        assert diag.fused_meters_per_unit is None
