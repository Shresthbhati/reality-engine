"""WorldIR integration for the detail spine (directive §8/§11: every
result evidence-grounded in the world model; the handoff's recorded
open item after PR #41).

The chain quality -> budget -> discovery -> ROI -> refinement ends in
work-order records and pipeline facts -- with no world-model consumer
the refined outcomes dead-end. This layer closes that loop:

  - REFINED outcomes become REAL WorldIR statements: one Geometry
    record (typed by the winning backend) + one Entity record linked
    via geometry_ids, both carrying the repo's provenance conventions
    (RECONSTRUCTED provenance, measured quality as confidence, an
    Observation recording the ALGORITHM + its measured residuals +
    ROI provenance, DERIVED statement state via the existing
    classifier).
  - REFUSED outcomes record a FACT, never geometry: nothing is added
    to entities/geometries (a refusal is not a shape); the diagnostic
    is preserved in the report's `refusals` field.
  - Reuses the established conventions unchanged -- promote_planes'
    Geometry+Observation pattern, the mesh stage's artifact-store and
    validate-rollback pattern, statement_state's classifier -- no
    parallel schema, no parallel ingestion path.
  - Deterministic: same input -> byte-identical report; no wall
    clock, no RNG.
"""

from __future__ import annotations

import math

import pytest


def _pt(xyz, tid, eid="c0"):
    from reconstruction.backend.interface import ReconstructedPoint

    return ReconstructedPoint(
        position=xyz, track_id=tid, source_evidence_ids=[eid]
    )


def _shell(cx, cy):
    from tests.test_detail_discovery import _cyl_shell

    return _cyl_shell(cx, cy)


def _roi_from_shell(world=None):
    """Run the real chain: points -> quality -> discovery -> ROI ->
    refinement, returning (result, cameras, rois, outcomes)."""
    from perception.detail.discovery import discover_detail
    from perception.detail.refinement import refine_rois
    from perception.detail.roi import generate_rois
    from tests.test_detail_discovery import (
        _quality_report, _report_cameras,
    )
    from tests.test_evidence_quality import _result

    pts = _shell(0.0, 0.0)
    cams = _report_cameras()
    result = _result(pts, cams)
    report = _quality_report(pts)
    cands = discover_detail(result, report)
    rois = generate_rois(cands, voxel_size=1.0)
    registry = {p.track_id: p.position for p in pts}
    outcomes = refine_rois(
        rois, point_lookup=lambda pid: registry.get(pid)
    )
    return result, cams, rois, outcomes


class TestIntegrationContract:
    def test_refined_outcome_becomes_entity_plus_geometry(self):
        from perception.detail.worldir import integrate_detail_outcomes

        _, _, rois, outcomes = _roi_from_shell()
        world = _empty_world()
        report = integrate_detail_outcomes(rois, outcomes, world)
        assert report.refusal_count == 0
        assert report.entity_ids and report.geometry_ids
        for eid in report.entity_ids:
            assert eid in world.entities
        for gid in report.geometry_ids:
            assert gid in world.geometries
        e = world.entities[report.entity_ids[0]]
        g = world.geometries[report.geometry_ids[0]]
        # Entity links its geometry (no dangling references).
        assert g.id in e.geometry_ids
        # Measured quality as confidence (bounded, derived from the
        # refinement's measured rms).
        assert 0.0 < e.confidence <= 1.0

    def test_geometry_typed_by_winning_backend(self):
        from perception.detail.worldir import integrate_detail_outcomes
        from world_ir import GeometryType

        _, _, rois, outcomes = _roi_from_shell()
        world = _empty_world()
        report = integrate_detail_outcomes(rois, outcomes, world)
        g = world.geometries[report.geometry_ids[0]]
        # The shell's winning backend (measured rms decides) is
        # cylinder or plane or sphere -- the record must match it,
        # never a fabricated type.
        expected = {
            "plane": GeometryType.PLANE,
            "cylinder": GeometryType.CYLINDER,
            "sphere": GeometryType.SPHERE,
        }[outcomes[0].refinement.backend]
        assert g.type is expected
        # lod_level records the budget's justified level (L2 here).
        assert g.lod_level == 2

    def test_provenance_answers_what_produced_this(self):
        from perception.detail.worldir import integrate_detail_outcomes
        from provenance import Provenance
        from world_ir import GeometryType

        _, _, rois, outcomes = _roi_from_shell()
        world = _empty_world()
        report = integrate_detail_outcomes(rois, outcomes, world)
        g = world.geometries[report.geometry_ids[0]]
        e = world.entities[report.entity_ids[0]]
        # Provenance is the repo's RECONSTRUCTED convention.
        assert g.provenance is Provenance.RECONSTRUCTED
        assert e.provenance is Provenance.RECONSTRUCTED
        obs = g.observations[0]
        # The observation answers "what produced this": algorithm,
        # measured residuals, quality, ROI + cell provenance.
        assert obs.sensor_type == "detail_refinement"
        md = obs.metadata
        assert md["backend"] == outcomes[0].refinement.backend
        assert md["rms_residual_m"] == pytest.approx(
            outcomes[0].refinement.rms_residual_m
        )
        assert md["quality"] == pytest.approx(
            outcomes[0].refinement.quality
        )
        assert md["roi_id"] == rois[0].roi_id
        assert md["discovery"] == "perception.detail.discovery.discover_detail"
        assert md["point_ids"] == list(rois[0].point_ids)
        # Statement state via the existing classifier: RECONSTRUCTED ->
        # DERIVED (never fabricated OBSERVED).
        from world_ir.statement_state import StatementState

        assert e.statement_state is StatementState.DERIVED

    def test_refused_roi_records_fact_not_geometry(self):
        from perception.detail.worldir import integrate_detail_outcomes

        # A refusal outcome: evidence unresolvable.
        from perception.detail.refinement import RefinementOutcome
        from perception.detail.roi import RegionOfInterest
        from perception.quality.detail_budget import DetailBudget

        _, _, rois, _ = _roi_from_shell()
        world = _empty_world()
        n_ent, n_geom = len(world.entities), len(world.geometries)
        refused = RefinementOutcome(
            roi_id=rois[0].roi_id,
            status="refused",
            reason=(
                "evidence unresolvable: 2 of 2 point_ids missing "
                "(first: 'ghost-1')"
            ),
            refinement=None,
            fits_attempted=0,
            fits_succeeded=0,
            backends_run=(),
        )
        report = integrate_detail_outcomes(
            [rois[0]], [refused], world
        )
        assert report.refusal_count == 1
        assert report.entity_ids == [] and report.geometry_ids == []
        # Nothing fabricated.
        assert len(world.entities) == n_ent
        assert len(world.geometries) == n_geom
        # The refusal is a recorded fact with its diagnostic.
        assert report.refusals == [
            {"roi_id": rois[0].roi_id, "reason": refused.reason}
        ]

    def test_deterministic_and_validates_clean(self):
        from perception.detail.worldir import integrate_detail_outcomes
        from world_ir.validation import validate_world_ir

        def run():
            world = _empty_world()
            _, _, rois, outcomes = _roi_from_shell()
            report = integrate_detail_outcomes(rois, outcomes, world)
            vr = validate_world_ir(world)
            return report.to_dict(), vr.errors

        d1, e1 = run()
        d2, e2 = run()
        assert d1 == d2
        # Integrated detail statements keep the world valid.
        assert e1 == [] and e2 == []

    def test_mixed_refined_and_refused(self):
        from perception.detail.worldir import integrate_detail_outcomes
        from perception.detail.refinement import RefinementOutcome

        _, _, rois, outcomes = _roi_from_shell()
        world = _empty_world()
        # One extra refused ROI with a distinct id.
        from perception.detail.roi import RegionOfInterest

        other = RegionOfInterest(
            roi_id="roi-9-9-9",
            parent_entity_id=None,
            bounds=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            n_cells=1,
            n_points=2,
            detail_cells=("9-9-9",),
            point_ids=("ghost-a", "ghost-b"),
            budget=rois[0].budget,
            max_curvature=0.2,
            status="pending",
            provenance={"voxel_size": 1.0},
        )
        refused = RefinementOutcome(
            roi_id="roi-9-9-9",
            status="refused",
            reason="evidence unresolvable: 2 of 2 point_ids missing "
                   "(first: 'ghost-a')",
            refinement=None,
            fits_attempted=0,
            fits_succeeded=0,
            backends_run=(),
        )
        report = integrate_detail_outcomes(
            rois + [other], outcomes + [refused], world
        )
        assert report.refusal_count == 1
        assert len(report.entity_ids) == len(
            [o for o in outcomes if o.status == "refined"]
        )
        # Refused id must not appear as an entity.
        assert not any(
            "roi-9-9-9" in str(getattr(world.entities[e], "custom_properties", {}))
            for e in world.entities
        )


class TestPipelineWiring:
    def test_pipeline_driver_gains_world_ir_output(self):
        from perception.detail.pipeline import run_detail_pipeline

        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        pts = _shell(0.0, 0.0)
        cams = _report_cameras()
        result = _result(pts, cams)
        rep = run_detail_pipeline(result, cams, build_world_ir=True)
        assert rep.world_ir is not None
        assert rep.summary["n_entities"] >= 1
        assert rep.summary["n_refused"] == rep.refusal_count
        # The world carried the integrated statements.
        assert rep.world_ir.entities and rep.world_ir.geometries

    def test_pipeline_without_world_ir_unchanged(self):
        from perception.detail.pipeline import run_detail_pipeline

        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        pts = _shell(0, 0)
        cams = _report_cameras()
        result = _result(pts, cams)
        rep = run_detail_pipeline(result, cams)
        assert rep.world_ir is None
        assert rep.summary.get("n_entities", 0) == 0


def _empty_world():
    from world_ir import WorldIR

    return WorldIR()


class TestVerticalSliceWiring:
    def test_slice_stage_3_8_integrates_into_world(self):
        from engine.pipeline.vertical_slice import _detail_stage

        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        pts = _shell(0.0, 0.0)
        cams = _report_cameras()
        result = _result(pts, cams)

        from perception.detail.pipeline import run_detail_pipeline
        from world_ir import WorldIR

        world = WorldIR()
        facts = _detail_stage(result, world, _slice_options())
        assert facts["status"] == "ran"
        assert facts["summary"]["n_entities"] >= 1
        # The world now carries the detail statements.
        assert world.entities and world.geometries
        # Refused ROIs did not fabricate entities.
        assert world.metadata["detail"]["summary"]["n_refused"] == 0


def _slice_options():
    from engine.pipeline.vertical_slice import VerticalSliceOptions

    return VerticalSliceOptions(
        intrinsics=(640.0, 640.0, 320.0, 0.0 + 240.0),
        image_size=(640, 480),
    )
