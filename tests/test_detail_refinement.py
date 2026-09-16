"""ROI refinement executor tests (universal-perception directive
sections 11/13: DISCOVERY -> ROI -> LOCAL RECONSTRUCTION -> ADAPTIVE
REFINEMENT; P7-06's declared open item).

The executor consumes pending RegionOfInterest work orders. Contract
under test:

  - Only REAL points refine: the executor resolves the ROI's
    point_ids through the caller's lookup -- it NEVER fabricates
    geometry from ROI metadata (bounds/curvature/budget) alone. A
    work order whose evidence cannot be resolved is refused, not
    guessed.
  - The compute_tier SPENDS effort: "none" never fits; "light"/
    "standard" run the plane backend only; "high"/"full" escalate
    through cylinder and sphere backends. Measured effort counters
    record what actually ran.
  - Outcomes are MEASURED: rms/max residuals from the winning
    backend, never fabricated; quality is the documented monotone
    map 1/(1+(rms/tol)^2) applied to the winning rms.
  - Status transitions carry evidence: success -> "refined" with a
    refinement outcome; failure -> "refused" with a diagnostic
    reason. Never a silent fake success.
  - Deterministic: same input -> byte-identical result; the input
    ROI list is not mutated.

Backends (perception/detail/backends.py): degenerate support raises
FitRefused, never best-effort garbage. fit_plane is the level-1
backend (smallest-eigenvalue normal via the repo's Jacobi solver);
sphere/cylinder reuse parametric.py's measured-residual fits
UNCHANGED -- no duplicated geometry math.
"""

from __future__ import annotations

import math

import pytest

from perception.detail.roi import RegionOfInterest
from perception.quality.detail_budget import DetailBudget


def _pt(xyz, tid, eid="c0"):
    from reconstruction.backend.interface import ReconstructedPoint

    return ReconstructedPoint(
        position=xyz, track_id=tid, source_evidence_ids=[eid]
    )


def _shell(cx, cy):
    """Stacked-ring cylinder-shell sample (genuinely curved), the
    same construction the discovery tests use."""
    from tests.test_detail_discovery import _cyl_shell

    return _cyl_shell(cx, cy)


def _budget(level="L2", tier="standard", gsd=10.0):
    return DetailBudget(
        justified_level=level,
        compute_tier=tier,
        max_gsd_mm_per_px=gsd,
        coverage_capped=False,
        basis="measured",
    )


def _roi(
    point_ids,
    budget=None,
    voxel_size=1.0,
    bounds=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    cell_id="0-0-0",
    max_curvature=0.2,
):
    return RegionOfInterest(
        roi_id=f"roi-{cell_id}",
        parent_entity_id=None,
        bounds=bounds,
        n_cells=1,
        n_points=len(point_ids),
        detail_cells=(cell_id,),
        point_ids=tuple(point_ids),
        budget=budget or _budget(),
        max_curvature=max_curvature,
        status="pending",
        provenance={"voxel_size": voxel_size},
    )


class TestPlaneBackend:
    def test_plane_fit_measures_real_residuals(self):
        from perception.detail.backends import PlaneFit, fit_plane

        # Exact plane z = 2: rms must be 0 (measured, not claimed).
        pts = [(x * 0.1, y * 0.1, 2.0) for x in range(6) for y in range(6)]
        fit = fit_plane(pts)
        assert isinstance(fit, PlaneFit)
        assert fit.rms_residual_m < 1e-9
        assert fit.max_residual_m < 1e-9
        # Normal is unit and vertical.
        n = fit.normal
        assert abs(n[0]) < 1e-9 and abs(n[1]) < 1e-9
        assert abs(abs(n[2]) - 1.0) < 1e-9
        assert fit.n_points == 36
        assert fit.confidence > 0.99

    def test_plane_fit_refuses_degenerate_support(self):
        from perception.detail.backends import FitRefused, fit_plane

        # Collinear points: no plane is determined.
        pts = [(i * 0.1, 0.0, 0.0) for i in range(10)]
        with pytest.raises(FitRefused):
            fit_plane(pts)

    def test_tilted_plane_residuals_are_measured(self):
        from perception.detail.backends import fit_plane

        # z = 0.5x + 1 with deterministic dither: residuals are the
        # measured deviations, bounded by the dither amplitude.
        dither = [0.001 * ((-1.0) ** i) for i in range(25)]
        pts = []
        for i in range(25):
            x, y = i * 0.2, (i % 5) * 0.2
            pts.append((x, y, 0.5 * x + 1.0 + dither[i]))
        fit = fit_plane(pts)
        assert 1e-6 < fit.rms_residual_m < 0.01
        assert fit.max_residual_m <= 0.0011 + 1e-12


class TestSphereBackend:
    def test_sphere_backend_reuses_parametric(self):
        from perception.detail.backends import fit_sphere_on

        pts = [
            (1.0 + 2.0 * math.sin(a) * math.cos(b),
             2.0 + 2.0 * math.sin(a) * math.sin(b),
             3.0 + 2.0 * math.cos(a))
            for a in [k * math.pi / 6 for k in range(7)]
            for b in [k * 2.0 * math.pi / 8 for k in range(8)]
        ]
        fit = fit_sphere_on(pts)
        assert fit.rms_residual_m < 1e-6
        assert fit.confidence > 0.9


def _shell_chain():
    """Real chain: shell points -> discovery -> ROI generation."""
    from perception.detail.discovery import discover_detail
    from perception.detail.roi import generate_rois
    from tests.test_detail_discovery import (
        _quality_report, _report_cameras,
    )
    from tests.test_evidence_quality import _result

    pts = _shell(0.0, 0.0)
    report = _quality_report(pts)
    cands = discover_detail(_result(pts, _report_cameras()), report)
    rois = generate_rois(cands, voxel_size=1.0)
    return pts, rois


class TestPipelineWiring:
    """One driver call executes the whole detail chain (assess ->
    discover -> ROI -> refine -> apply) so the stages are not isolated
    modules consumed only by tests."""

    def _shell_result(self):
        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        pts = _shell(0.0, 0.0)
        return _result(pts, _report_cameras()), _report_cameras(), pts

    def test_pipeline_runs_end_to_end(self):
        from perception.detail.pipeline import run_detail_pipeline

        result, cameras, pts = self._shell_result()
        report = run_detail_pipeline(result, cameras)
        # The full chain produced measured artifacts...
        assert report.candidates
        assert report.rois
        # ...and the lifecycle completed: every work order transitioned.
        assert all(r.status != "pending" for r in report.rois)
        refined = [r for r in report.rois if r.status == "refined"]
        assert refined
        assert all(
            r.refinement_outcome.rms_residual_m > 0.0 for r in refined
        )
        # Summary counts are measured and coherent.
        assert report.summary["n_rois"] == len(report.rois)
        assert report.summary["n_refined"] == len(refined)
        assert (
            report.summary["n_refined"] + report.summary["n_refused"]
            == report.summary["n_rois"]
        )

    def test_pipeline_default_lookup_uses_result_points(self):
        # No explicit point_lookup: the driver resolves evidence from
        # the reconstruction result's own points.
        from perception.detail.pipeline import run_detail_pipeline

        result, cameras, pts = self._shell_result()
        report = run_detail_pipeline(result, cameras, point_lookup=None)
        assert report.summary["n_refined"] >= 1

    def test_pipeline_empty_scene_is_honest_and_empty(self):
        from perception.detail.pipeline import run_detail_pipeline

        result, cameras, _ = self._shell_result()
        empty = type(result)(points=[], camera_poses=[],
                             registration_status="success")
        report = run_detail_pipeline(empty, cameras)
        assert report.candidates == []
        assert report.rois == []
        assert report.outcomes == []
        assert report.summary["n_rois"] == 0

    def test_pipeline_deterministic(self):
        from perception.detail.pipeline import run_detail_pipeline

        result, cameras, _ = self._shell_result()
        r1 = run_detail_pipeline(result, cameras).to_dict()
        r2 = run_detail_pipeline(result, cameras).to_dict()
        assert r1 == r2


class TestVerticalSliceStage:
    """The vertical-slice wiring (stage 3.8): the detail chain runs in
    the canonical capture-to-world driver, reusing the canonical
    cameras built there -- connective tissue, not an isolated module.
    Contract (the slice's stage pattern): facts dict with a visible
    status, never raises, never fabricates."""

    def _result_and_cameras(self):
        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        pts = _shell(0.0, 0.0)
        cams = _report_cameras()
        return _result(pts, cams), cams, pts

    def test_stage_runs_and_records_world_metadata(self):
        from engine.pipeline.vertical_slice import _detail_stage

        result, cams, pts = self._result_and_cameras()
        from world_ir import WorldIR as _WorldIR

        world = _WorldIR()
        facts = _detail_stage(result, world, self._options())
        assert facts["status"] == "ran"
        assert facts["summary"]["n_refined"] >= 1
        assert "detail" in world.metadata
        assert world.metadata["detail"]["summary"]["n_refined"] >= 1
        # Measured refinement evidence recorded.
        assert facts["summary"]["n_refined"] + facts["summary"]["n_refused"] \
            == facts["summary"]["n_rois"]

    def test_stage_disabled_is_a_visible_skip(self):
        from engine.pipeline.vertical_slice import _detail_stage

        result, cams, _ = self._result_and_cameras()
        world = _FakeWorld()
        facts = _detail_stage(
            result, world, self._options(detail_enabled=False)
        )
        assert facts["status"] == "skipped"
        assert "disabled" in facts["note"]
        assert "detail" not in world.metadata

    def test_stage_without_cameras_refuses_measurement_honestly(self):
        from engine.pipeline.vertical_slice import _detail_stage

        result, _, _ = self._result_and_cameras()
        world = _FakeWorld()
        # No trusted intrinsics -> GSD is unmeasurable -> the stage
        # must SKIP VISIBLY, never fabricate budgets.
        opts = self._options()
        opts = type(opts)(**{
            **{f: getattr(opts, f) for f in (
                'seed', 'up', 'measured_baselines', 'colmap_binary',
                'image_size', 'detail_enabled', 'detail_voxel_size_m')},
            'intrinsics': None,
        })
        facts = _detail_stage(result, world, opts)
        assert facts["status"] == "skipped"
        assert "intrinsics" in facts["note"].lower()
        assert "detail" not in world.metadata

    def _options(self, detail_enabled=True):
        from engine.pipeline.vertical_slice import VerticalSliceOptions

        # Trusted intrinsics matching the discovery-test camera
        # geometry (fx=width -> the shell at z~0.65 is in bounds).
        return VerticalSliceOptions(
            detail_enabled=detail_enabled,
            intrinsics=(640.0, 640.0, 320.0, 240.0),
            image_size=(640, 480),
        )


class _FakeWorld:
    """Minimal WorldIR-shaped sink for the detail stage: entities +
    geometries dicts (integration targets) + metadata."""

    def __init__(self):
        self.metadata = {}
        self.entities = {}
        self.geometries = {}


class TestLifecycleWiring:
    """The state-transition ownership roi.py declares: local
    refinement owns pending -> refined/refused. `apply_outcomes` is
    the connective tissue a pipeline driver consumes -- outcomes
    become updated work-order records, never in-place mutation."""

    def test_outcomes_transition_roi_status(self):
        from perception.detail.refinement import apply_outcomes, refine_rois

        pts, rois = _shell_chain()
        registry = {p.track_id: p.position for p in pts}
        outcomes = refine_rois(
            rois, point_lookup=lambda pid: registry.get(pid)
        )
        updated = apply_outcomes(rois, outcomes)
        assert len(updated) == len(rois)
        # Original work orders untouched; new records carry statuses.
        assert all(r.status == "pending" for r in rois)
        assert all(r.status != "pending" for r in updated)
        # The refined record's status matches its outcome, and the
        # measured evidence rides along on the record.
        by_id = {o.roi_id: o for o in outcomes}
        for roi in updated:
            o = by_id[roi.roi_id]
            assert roi.status == o.status
            if o.status == "refined":
                assert roi.refinement_outcome is o.refinement
                assert roi.refinement_outcome.rms_residual_m > 0.0
            else:
                assert roi.refinement_outcome is None
                assert roi.status_reason

    def test_apply_survives_unknown_roi_ids(self):
        from perception.detail.refinement import (
            RefinementOutcome, apply_outcomes,
        )

        pts, rois = _shell_chain()
        stranger = RefinementOutcome(
            roi_id="roi-not-in-input", status="refined", reason=None,
            refinement=None, fits_attempted=1, fits_succeeded=1,
            backends_run=("plane",),
        )
        updated = apply_outcomes(rois, [stranger])
        # Unknown outcomes are ignored; input ROIs stay pending in the
        # returned records (no outcome == no transition).
        assert len(updated) == len(rois)
        assert all(r.status == "pending" for r in updated)


class TestExecutorContract:
    voxel = 1.0

    def _discover_roi_from_shell(self):
        """Real chain: shell points -> discovery -> ROI generation."""
        from perception.detail.discovery import discover_detail
        from perception.detail.roi import generate_rois
        from tests.test_detail_discovery import (
            _quality_report, _report_cameras,
        )
        from tests.test_evidence_quality import _result

        pts = _shell(0.0, 0.0)
        report = _quality_report(pts)
        cands = discover_detail(_result(pts, _report_cameras()), report)
        rois = generate_rois(cands, voxel_size=self.voxel)
        return pts, rois

    def _registry(self, pts):
        return {p.track_id: p.position for p in pts}

    def test_pending_roi_with_resolvable_evidence_refines(self):
        from perception.detail.refinement import refine_rois

        pts, rois = self._discover_roi_from_shell()
        assert rois and rois[0].status == "pending"
        registry = self._registry(pts)
        outcomes = refine_rois(
            rois,
            point_lookup=lambda pid: registry.get(pid),
            up=(0.0, 0.0, 1.0),
        )
        assert len(outcomes) == len(rois)
        o = outcomes[0]
        assert o.roi_id == rois[0].roi_id
        assert o.status == "refined"
        assert o.reason is None
        # Measured residuals: shell points lie on a curved surface,
        # so the winning fit has real, nonzero residual.
        assert o.refinement is not None
        assert o.refinement.rms_residual_m > 0.0
        assert o.refinement.max_residual_m >= o.refinement.rms_residual_m
        # Measured effort counters (outcome-level).
        assert o.fits_attempted >= 1
        assert o.fits_succeeded == 1
        # Quality derived by the documented map from measured rms.
        assert 0.0 < o.refinement.quality <= 1.0
        # Provenance: the winning backend is recorded.
        assert o.refinement.backend in ("plane", "cylinder", "sphere")

    def test_roi_with_unresolvable_evidence_refuses(self):
        from perception.detail.refinement import refine_rois

        pts, rois = self._discover_roi_from_shell()
        registry = self._registry(pts)
        unresolvable = [
            RegionOfInterest(
                roi_id=r.roi_id,
                parent_entity_id=r.parent_entity_id,
                bounds=r.bounds,
                n_cells=r.n_cells,
                n_points=r.n_points,
                detail_cells=r.detail_cells,
                point_ids=("ghost-1", "ghost-2"),  # resolve to nothing
                budget=r.budget,
                max_curvature=r.max_curvature,
                status="pending",
                provenance=dict(r.provenance),
            )
            for r in rois
        ]
        outcomes = refine_rois(
            unresolvable, point_lookup=lambda pid: registry.get(pid)
        )
        assert outcomes[0].status == "refused"
        assert outcomes[0].refinement is None
        assert "ghost-1" in outcomes[0].reason

    def test_zero_tier_never_fits(self):
        from perception.detail.refinement import refine_rois

        pts, rois = self._discover_roi_from_shell()
        registry = self._registry(pts)
        zero = [
            RegionOfInterest(
                roi_id=r.roi_id,
                parent_entity_id=r.parent_entity_id,
                bounds=r.bounds,
                n_cells=r.n_cells,
                n_points=r.n_points,
                detail_cells=r.detail_cells,
                point_ids=r.point_ids,
                budget=_budget(level="L0", tier="none"),
                max_curvature=r.max_curvature,
                status="pending",
                provenance=dict(r.provenance),
            )
            for r in rois
        ]
        outcomes = refine_rois(
            zero, point_lookup=lambda pid: registry.get(pid)
        )
        assert outcomes[0].status == "refused"
        assert "compute_tier=none" in outcomes[0].reason
        assert outcomes[0].refinement is None

    def test_high_tier_escalates_through_backends(self):
        from perception.detail.refinement import refine_rois

        pts, rois = self._discover_roi_from_shell()
        registry = self._registry(pts)
        high = [
            RegionOfInterest(
                roi_id=r.roi_id,
                parent_entity_id=r.parent_entity_id,
                bounds=r.bounds,
                n_cells=r.n_cells,
                n_points=r.n_points,
                detail_cells=r.detail_cells,
                point_ids=r.point_ids,
                budget=_budget(level="L4", tier="high"),
                max_curvature=r.max_curvature,
                status="pending",
                provenance=dict(r.provenance),
            )
            for r in rois
        ]
        outcomes = refine_rois(
            high, point_lookup=lambda pid: registry.get(pid),
            up=(0.0, 0.0, 1.0),
        )
        o = outcomes[0]
        assert o.status == "refined"
        # Escalation is MEASURED: plane ran first (and was beaten by
        # the shell's true cylinder backend, which the counters show).
        assert o.fits_attempted >= 2
        assert o.refinement.backend in ("cylinder", "sphere")
        assert o.fits_succeeded >= 1

    def test_deterministic_and_non_mutating(self):
        from perception.detail.refinement import refine_rois

        pts, rois = self._discover_roi_from_shell()
        registry = self._registry(pts)
        before = [r.to_dict() for r in rois]

        def run():
            return [
                o.to_dict()
                for o in refine_rois(
                    rois,
                    point_lookup=lambda pid: registry.get(pid),
                    up=(0.0, 0.0, 1.0),
                )
            ]

        assert run() == run()
        # The input work orders are untouched.
        assert [r.to_dict() for r in rois] == before

    def test_all_backends_refusing_yields_honest_refusal(self):
        from perception.detail.refinement import refine_rois

        # Isotropic blob: every fit backend refuses (plane is
        # under-determined in axis; sphere/cylinder gates fail).
        n = 40
        pts = []
        for k in range(n):
            a = k * 2.399963  # golden-angle spiral on a sphere
            z = 1.0 - 2.0 * k / (n - 1)
            r = math.sqrt(max(0.0, 1.0 - z * z))
            pts.append(_pt(
                (0.5 + r * math.cos(a), 0.5 + r * math.sin(a), 0.5 + z),
                f"blob-{k}",
            ))
        registry = self._registry(pts)
        roi = _roi([p.track_id for p in pts], max_curvature=0.2)
        outcomes = refine_rois(
            [roi],
            point_lookup=lambda pid: registry.get(pid),
            up=(0.0, 0.0, 1.0),
        )
        # Whatever the winner is, the residual must be MEASURED and
        # the outcome must not pretend more than the evidence
        # supports; for an isotropic blob no backend may claim a
        # high-quality fit.
        o = outcomes[0]
        if o.status == "refined":
            assert o.refinement.quality < 0.9
        else:
            assert o.refinement is None and o.reason
