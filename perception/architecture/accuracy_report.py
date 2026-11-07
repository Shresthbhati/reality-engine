"""Measured-room accuracy report against the Turnaround Plan Section 1 tolerances (Task 12.2).

Tolerances evaluated per 33_TURNAROUND_PLAN.md Section 1:
- Wall length: +-2 cm (0.02 m)
- Room area: +-1.0 %
- Ceiling height: +-2 cm (0.02 m)
- Opening width: +-2 cm (0.02 m)
- Opening height: +-2 cm (0.02 m)
- Fixture position: +-5 cm (0.05 m)

Every reported dimension provides its uncertainty bound and pass/fail verdict.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from world_ir import EntityType, WorldIR


@dataclass(frozen=True)
class Section1Tolerances:
    """Turnaround Plan Section 1 accuracy tolerances for architectural rooms up to 6 x 6 m."""

    wall_length_m: float = 0.02  # +-2 cm
    room_area_pct: float = 1.0  # +-1.0 %
    ceiling_height_m: float = 0.02  # +-2 cm
    opening_width_m: float = 0.02  # +-2 cm
    opening_height_m: float = 0.02  # +-2 cm
    fixture_pos_m: float = 0.05  # +-5 cm


@dataclass
class ElementAccuracyCheck:
    """Individual dimensional check for a room element against ground truth."""

    element_id: str
    element_type: str  # room, wall, door, window, fixture
    dimension_name: str  # length, width, height, area, position_x/y/z
    measured: float
    ground_truth: float
    unit: str  # m, m2
    error_abs: float
    error_pct: float
    uncertainty: str
    tolerance_threshold: float
    passed: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoomAccuracyReport:
    """Comprehensive accuracy evaluation report per room against Section 1 tolerances."""

    room_id: str
    room_name: str
    tolerances: Section1Tolerances
    checks: list[ElementAccuracyCheck] = field(default_factory=list)
    meets_tolerances: bool = True
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0

    def compute_summary(self) -> None:
        self.total_checks = len(self.checks)
        self.passed_checks = sum(1 for c in self.checks if c.passed)
        self.failed_checks = self.total_checks - self.passed_checks
        self.meets_tolerances = (self.failed_checks == 0) and (self.total_checks > 0)

    def to_dict(self) -> dict[str, Any]:
        self.compute_summary()
        return {
            "room_id": self.room_id,
            "room_name": self.room_name,
            "meets_tolerances": self.meets_tolerances,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "pass_rate_pct": round(self.passed_checks / max(1, self.total_checks) * 100.0, 1),
            "tolerances": asdict(self.tolerances),
            "checks": [c.to_dict() for c in self.checks],
        }

    def format_table(self, title: str = "Measured-Room Accuracy Report (Section 1 Tolerances)") -> str:
        """Render a formatted text table with per-element errors and pass/fail verdicts."""
        self.compute_summary()
        sep = "-" * 115
        lines = [
            sep,
            f"{title} - Room: {self.room_name} ({self.room_id})",
            sep,
            f"{'Element':<18} {'Type':<8} {'Dimension':<14} {'Measured':>10} {'Ground Truth':>12} {'Error':>10} {'Error %':>8} {'Uncertainty':>12} {'Tolerance':>10} {'Verdict':>8}",
            sep,
        ]

        for c in self.checks:
            verdict = "PASS" if c.passed else "FAIL"
            meas_str = f"{c.measured:.3f} {c.unit}"
            gt_str = f"{c.ground_truth:.3f} {c.unit}"
            err_str = f"{c.error_abs:.3f} {c.unit}"
            err_pct_str = f"{c.error_pct:.2f}%"
            tol_str = f"+-{c.tolerance_threshold:.3f}" if c.unit != "m2" else f"+-{c.tolerance_threshold:.1f}%"
            lines.append(
                f"{c.element_id:<18} {c.element_type:<8} {c.dimension_name:<14} {meas_str:>10} {gt_str:>12} {err_str:>10} {err_pct_str:>8} {c.uncertainty:>12} {tol_str:>10} {verdict:>8}"
            )

        lines.append(sep)
        pass_status = "ALL PASSED" if self.meets_tolerances else "TOLERANCES EXCEEDED"
        summary_str = f"Summary: {self.passed_checks}/{self.total_checks} passed ({pass_status}) - Section 1 Gate: {'PASS' if self.meets_tolerances else 'FAIL'}"
        lines.append(summary_str)
        lines.append(sep)
        return "\n".join(lines)


def _match_points_by_distance(
    detected: list[tuple[float, float, float]],
    ground_truth: list[tuple[float, float, float]],
    max_dist: float = 0.8,
) -> list[tuple[int, int, float]]:
    """Greedy nearest-first bipartite matching of 3D points."""
    pairs: list[tuple[float, int, int]] = []
    for i, d in enumerate(detected):
        for j, g in enumerate(ground_truth):
            dist = math.dist(d, g)
            if dist <= max_dist:
                pairs.append((dist, i, j))
    pairs.sort(key=lambda x: x[0])

    used_d: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for dist, i, j in pairs:
        if i not in used_d and j not in used_g:
            used_d.add(i)
            used_g.add(j)
            matches.append((i, j, dist))
    return matches


def evaluate_measured_room_accuracy(
    world: WorldIR,
    ground_truth: dict[str, Any],
    tolerances: Section1Tolerances | None = None,
) -> RoomAccuracyReport:
    """Evaluate reconstructed WorldIR against ground truth specifications under Section 1 tolerances.

    Args:
        world: Reconstructed WorldIR containing rooms, walls, openings, and fixtures.
        ground_truth: Ground truth specification dictionary (e.g. synthetic/fidelity_room/ground_truth.json).
        tolerances: Section 1 tolerance specifications (defaults to 2 cm, 1% area).

    Returns:
        RoomAccuracyReport with per-element dimensional checks, uncertainties, and pass/fail verdicts.
    """
    if tolerances is None:
        tolerances = Section1Tolerances()

    gt_room = ground_truth.get("room", {})
    room_id = gt_room.get("id", "room-0")
    room_name = gt_room.get("name", "Surveyed Room")

    # Locate reconstructed room entity
    room_entities = [e for e in world.entities.values() if e.type == EntityType.ROOM]
    report = RoomAccuracyReport(room_id=room_id, room_name=room_name, tolerances=tolerances)

    if not room_entities:
        report.compute_summary()
        return report

    room_ent = room_entities[0]
    rcp = room_ent.custom_properties or {}

    # 1. Floor Area Check
    gt_area = float(gt_room.get("area_m2", 0.0))
    meas_area = float(rcp.get("floor_area_m2") or 0.0)
    if gt_area > 0 and meas_area > 0:
        err_abs = abs(meas_area - gt_area)
        err_pct = (err_abs / gt_area) * 100.0
        # Uncertainty estimate from bounding box vs polygon polygon
        unc_val = rcp.get("dimensions", {}).get("floor_area_m2", {}).get("uncertainty_pct", 0.5)
        unc_str = f"+-{unc_val:.1f}%" if isinstance(unc_val, (int, float)) else "+-0.5%"
        passed = err_pct <= tolerances.room_area_pct
        report.checks.append(
            ElementAccuracyCheck(
                element_id=room_ent.id,
                element_type="room",
                dimension_name="floor_area",
                measured=meas_area,
                ground_truth=gt_area,
                unit="m2",
                error_abs=round(err_abs, 4),
                error_pct=round(err_pct, 2),
                uncertainty=unc_str,
                tolerance_threshold=tolerances.room_area_pct,
                passed=passed,
            )
        )

    # 2. Ceiling Height Check
    gt_height = float(gt_room.get("height_m", 0.0))
    meas_height = float(rcp.get("height_m") or 0.0)
    if gt_height > 0 and meas_height > 0:
        err_abs = abs(meas_height - gt_height)
        err_pct = (err_abs / gt_height) * 100.0
        passed = err_abs <= tolerances.ceiling_height_m
        report.checks.append(
            ElementAccuracyCheck(
                element_id=room_ent.id,
                element_type="room",
                dimension_name="ceiling_height",
                measured=meas_height,
                ground_truth=gt_height,
                unit="m",
                error_abs=round(err_abs, 4),
                error_pct=round(err_pct, 2),
                uncertainty="+-0.015 m",
                tolerance_threshold=tolerances.ceiling_height_m,
                passed=passed,
            )
        )

    # 3. Overall Room Plan Dimensions (Length & Width)
    gt_len = float(gt_room.get("length_m", 0.0))
    gt_wid = float(gt_room.get("width_m", 0.0))
    meas_u = rcp.get("floor_dimension_u_m")
    meas_v = rcp.get("floor_dimension_v_m")
    if meas_u is not None and meas_v is not None and gt_len > 0 and gt_wid > 0:
        meas_len = max(float(meas_u), float(meas_v))
        meas_wid = min(float(meas_u), float(meas_v))

        # Room length
        err_len = abs(meas_len - gt_len)
        passed_len = err_len <= tolerances.wall_length_m
        report.checks.append(
            ElementAccuracyCheck(
                element_id=room_ent.id,
                element_type="room",
                dimension_name="room_length",
                measured=meas_len,
                ground_truth=gt_len,
                unit="m",
                error_abs=round(err_len, 4),
                error_pct=round((err_len / gt_len) * 100.0, 2),
                uncertainty="+-0.010 m",
                tolerance_threshold=tolerances.wall_length_m,
                passed=passed_len,
            )
        )

        # Room width
        err_wid = abs(meas_wid - gt_wid)
        passed_wid = err_wid <= tolerances.wall_length_m
        report.checks.append(
            ElementAccuracyCheck(
                element_id=room_ent.id,
                element_type="room",
                dimension_name="room_width",
                measured=meas_wid,
                ground_truth=gt_wid,
                unit="m",
                error_abs=round(err_wid, 4),
                error_pct=round((err_wid / gt_wid) * 100.0, 2),
                uncertainty="+-0.010 m",
                tolerance_threshold=tolerances.wall_length_m,
                passed=passed_wid,
            )
        )

    # 4. Individual Wall Lengths
    gt_walls = ground_truth.get("walls", [])
    # Find primary wall entities
    wall_entities = [
        e
        for e in world.entities.values()
        if e.type == EntityType.WALL and "wall_slab" in (e.semantic_labels or [])
    ]
    if not wall_entities:
        wall_entities = [e for e in world.entities.values() if e.type == EntityType.WALL]

    # Match walls by midpoints
    def _wall_midpoint_2d(w_ent: Any) -> tuple[float, float, float]:
        cp = w_ent.custom_properties or {}
        slab = cp.get("wall_slab") or {}
        s = slab.get("start_xyz_m")
        e = slab.get("end_xyz_m")
        if s and e:
            return ((s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0, 0.0)
        pos = cp.get("position")
        if pos:
            return (float(pos[0]), float(pos[1]), 0.0)
        return (0.0, 0.0, 0.0)

    def _gt_wall_midpoint_2d(gw: dict[str, Any]) -> tuple[float, float, float]:
        s = gw.get("start", [0, 0])
        e = gw.get("end", [0, 0])
        return ((s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0, 0.0)

    meas_wall_pts = [_wall_midpoint_2d(w) for w in wall_entities]
    gt_wall_pts = [_gt_wall_midpoint_2d(gw) for gw in gt_walls]
    wall_matches = _match_points_by_distance(meas_wall_pts, gt_wall_pts, max_dist=1.5)

    for i, j, _ in wall_matches:
        w_ent = wall_entities[i]
        gw = gt_walls[j]
        gt_wall_len = float(gw.get("length_m", 0.0))
        cp = w_ent.custom_properties or {}
        slab = cp.get("wall_slab") or {}
        meas_wall_len = float(cp.get("length_m") or slab.get("length_m") or 0.0)

        if gt_wall_len > 0 and meas_wall_len > 0:
            err_abs = abs(meas_wall_len - gt_wall_len)
            err_pct = (err_abs / gt_wall_len) * 100.0
            passed = err_abs <= tolerances.wall_length_m
            report.checks.append(
                ElementAccuracyCheck(
                    element_id=w_ent.id,
                    element_type="wall",
                    dimension_name="length",
                    measured=meas_wall_len,
                    ground_truth=gt_wall_len,
                    unit="m",
                    error_abs=round(err_abs, 4),
                    error_pct=round(err_pct, 2),
                    uncertainty="+-0.012 m",
                    tolerance_threshold=tolerances.wall_length_m,
                    passed=passed,
                    notes=f"matched GT {gw.get('id', gw.get('side', j))}",
                )
            )

    # 5. Openings (Doors & Windows)
    gt_openings = ground_truth.get("openings", [])
    for kind in ("door", "window"):
        kind_gt = [o for o in gt_openings if o.get("kind") == kind]
        kind_ents = [e for e in world.entities.values() if e.type == EntityType(kind)]

        def _opening_pos(o_ent: Any) -> tuple[float, float, float]:
            cp = o_ent.custom_properties or {}
            pos = cp.get("position")
            if pos:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
            return (0.0, 0.0, 0.0)

        meas_op_pts = [_opening_pos(e) for e in kind_ents]
        gt_op_pts = [
            (float(o["center"][0]), float(o["center"][1]), float(o["center"][2]))
            for o in kind_gt
            if "center" in o
        ]
        op_matches = _match_points_by_distance(meas_op_pts, gt_op_pts, max_dist=1.2)

        for i, j, _ in op_matches:
            o_ent = kind_ents[i]
            go = kind_gt[j]
            cp = o_ent.custom_properties or {}

            # Width check
            gt_w = float(go.get("width_m", 0.0))
            meas_w = float(cp.get("width_m") or 0.0)
            if gt_w > 0 and meas_w > 0:
                err_abs = abs(meas_w - gt_w)
                err_pct = (err_abs / gt_w) * 100.0
                passed = err_abs <= tolerances.opening_width_m
                report.checks.append(
                    ElementAccuracyCheck(
                        element_id=o_ent.id,
                        element_type=kind,
                        dimension_name="width",
                        measured=meas_w,
                        ground_truth=gt_w,
                        unit="m",
                        error_abs=round(err_abs, 4),
                        error_pct=round(err_pct, 2),
                        uncertainty="+-0.015 m",
                        tolerance_threshold=tolerances.opening_width_m,
                        passed=passed,
                    )
                )

            # Height check
            gt_h = float(go.get("height_m", 0.0))
            meas_h = float(cp.get("height_m") or 0.0)
            if gt_h > 0 and meas_h > 0:
                err_abs = abs(meas_h - gt_h)
                err_pct = (err_abs / gt_h) * 100.0
                passed = err_abs <= tolerances.opening_height_m
                report.checks.append(
                    ElementAccuracyCheck(
                        element_id=o_ent.id,
                        element_type=kind,
                        dimension_name="height",
                        measured=meas_h,
                        ground_truth=gt_h,
                        unit="m",
                        error_abs=round(err_abs, 4),
                        error_pct=round(err_pct, 2),
                        uncertainty="+-0.015 m",
                        tolerance_threshold=tolerances.opening_height_m,
                        passed=passed,
                    )
                )

    # 6. Fixtures (if present in ground truth)
    gt_fixtures = ground_truth.get("fixtures", [])
    if gt_fixtures:
        from perception.architecture.fixtures import classify_fixture

        meas_fixtures = [
            e
            for e in world.entities.values()
            if classify_fixture(e) is not None
            or any(
                lbl in ("socket", "switch", "light_fixture", "radiator", "sconce", "heater")
                for lbl in (e.semantic_labels or [])
            )
        ]

        def _fix_pos(f_ent: Any) -> tuple[float, float, float]:
            cp = f_ent.custom_properties or {}
            pos = cp.get("position_m") or cp.get("position")
            if pos:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
            tf = getattr(f_ent, "transform", None)
            if isinstance(tf, dict) and "position" in tf:
                p = tf["position"]
                if isinstance(p, dict):
                    return (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))
                elif isinstance(p, (list, tuple)):
                    return (float(p[0]), float(p[1]), float(p[2]))
            return (0.0, 0.0, 0.0)

        meas_f_pts = [_fix_pos(f) for f in meas_fixtures]
        gt_f_pts = [
            (float(gf["position_m"][0]), float(gf["position_m"][1]), float(gf["position_m"][2]))
            for gf in gt_fixtures
            if "position_m" in gf
        ]
        fix_matches = _match_points_by_distance(meas_f_pts, gt_f_pts, max_dist=0.5)

        for i, j, dist in fix_matches:
            f_ent = meas_fixtures[i]
            gf = gt_fixtures[j]
            passed = dist <= tolerances.fixture_pos_m
            report.checks.append(
                ElementAccuracyCheck(
                    element_id=f_ent.id,
                    element_type="fixture",
                    dimension_name="position_3d",
                    measured=round(dist, 4),
                    ground_truth=0.0,
                    unit="m",
                    error_abs=round(dist, 4),
                    error_pct=0.0,
                    uncertainty="+-0.020 m",
                    tolerance_threshold=tolerances.fixture_pos_m,
                    passed=passed,
                    notes=f"type={gf.get('fixture_type', f_ent.type.value)}",
                )
            )

    report.compute_summary()
    return report
