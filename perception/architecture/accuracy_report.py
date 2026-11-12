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

import numpy as np
from scipy.optimize import linear_sum_assignment

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
class ClassScore:
    """Detection quality for one element class (0/0 is defined as 0.0)."""

    element_type: str
    n_ground_truth: int
    n_detected: int
    n_matched: int
    precision: float
    recall: float
    f1: float


def _class_score(element_type: str, n_gt: int, n_det: int, n_matched: int) -> ClassScore:
    precision = n_matched / n_det if n_det else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ClassScore(
        element_type, n_gt, n_det, n_matched, round(precision, 4), round(recall, 4), round(f1, 4)
    )


# A ground-truth element of these types with no match fails the gate outright (decision D4).
GATE_CRITICAL_TYPES = frozenset({"wall", "door", "window", "opening"})


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
    class_scores: list[ClassScore] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)  # {"element_type", "gt_id"}
    extra: list[dict[str, Any]] = field(default_factory=list)  # {"element_type", "element_id"}
    room_found: bool = True

    def compute_summary(self) -> None:
        self.total_checks = len(self.checks)
        self.passed_checks = sum(1 for c in self.checks if c.passed)
        self.failed_checks = self.total_checks - self.passed_checks
        critical_missing = any(m.get("element_type") in GATE_CRITICAL_TYPES for m in self.missing)
        self.meets_tolerances = (
            self.failed_checks == 0
            and self.total_checks > 0
            and self.room_found
            and not critical_missing
        )

    def to_dict(self) -> dict[str, Any]:
        self.compute_summary()
        return {
            "room_id": self.room_id,
            "room_name": self.room_name,
            "meets_tolerances": self.meets_tolerances,
            "room_found": self.room_found,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "pass_rate_pct": round(self.passed_checks / max(1, self.total_checks) * 100.0, 1),
            "tolerances": asdict(self.tolerances),
            "checks": [c.to_dict() for c in self.checks],
            "class_scores": [asdict(c) for c in self.class_scores],
            "missing": [dict(m) for m in self.missing],
            "extra": [dict(e) for e in self.extra],
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
        if not self.room_found:
            lines.append("NO ROOM ENTITY FOUND")
        if self.class_scores:
            lines.append("Class scores:")
            lines.append(
                f"  {'Class':<10} {'GT':>4} {'Det':>4} {'Match':>6} {'Precision':>10} {'Recall':>8} {'F1':>8}"
            )
            for cs in self.class_scores:
                lines.append(
                    f"  {cs.element_type:<10} {cs.n_ground_truth:>4} {cs.n_detected:>4} {cs.n_matched:>6} "
                    f"{cs.precision:>10.3f} {cs.recall:>8.3f} {cs.f1:>8.3f}"
                )
        for m in self.missing:
            lines.append(f"MISSING {m['element_type']}: ground truth {m['gt_id']} has no match")
        for e in self.extra:
            lines.append(f"EXTRA   {e['element_type']}: detection {e['element_id']} has no ground truth")
        if self.class_scores or self.missing or self.extra:
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
    """One-to-one minimum-total-distance assignment of 3D points (Hungarian).

    Pairs farther apart than ``max_dist`` are rejected *after* the assignment, so a far pair can
    never displace a legal one. Returns ``(detected_idx, ground_truth_idx, distance)``.
    """
    if not detected or not ground_truth:
        return []
    cost = np.array([[math.dist(d, g) for g in ground_truth] for d in detected], dtype=float)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i, j])) for i, j in zip(rows, cols) if cost[i, j] <= max_dist]


def _add_dim_check(
    report: RoomAccuracyReport,
    *,
    element_id: str,
    element_type: str,
    dimension_name: str,
    measured: float,
    ground_truth: float,
    unit: str,
    tolerance: float,
    uncertainty: str,
    pct_tolerance: bool = False,
    notes: str = "",
) -> None:
    """Append a dimensional check. A zero or absent measured value is a failure, never a skip."""
    err_abs = abs(measured - ground_truth)
    err_pct = (err_abs / ground_truth) * 100.0
    within = (err_pct if pct_tolerance else err_abs) <= tolerance
    absent = measured <= 0.0
    report.checks.append(
        ElementAccuracyCheck(
            element_id=element_id,
            element_type=element_type,
            dimension_name=dimension_name,
            measured=measured,
            ground_truth=ground_truth,
            unit=unit,
            error_abs=round(err_abs, 4),
            error_pct=round(err_pct, 2),
            uncertainty=uncertainty,
            tolerance_threshold=tolerance,
            passed=within and not absent,
            notes=("measured value absent or zero" if absent else notes),
        )
    )


def _record_matching(
    report: RoomAccuracyReport,
    element_type: str,
    gt_ids: list[str],
    det_ids: list[str],
    matches: list[tuple[int, int, float]],
) -> None:
    """Fill class score, missing (unmatched ground truth) and extra (unmatched detections)."""
    matched_d = {i for i, _, _ in matches}
    matched_g = {j for _, j, _ in matches}
    report.class_scores.append(_class_score(element_type, len(gt_ids), len(det_ids), len(matches)))
    report.missing += [
        {"element_type": element_type, "gt_id": gid}
        for j, gid in enumerate(gt_ids)
        if j not in matched_g
    ]
    report.extra += [
        {"element_type": element_type, "element_id": did}
        for i, did in enumerate(det_ids)
        if i not in matched_d
    ]


def _match_with_optional_positions(
    det_pts: list[tuple[float, float, float]],
    gt_pts: list[tuple[float, float, float] | None],
    max_dist: float,
) -> list[tuple[int, int, float]]:
    """Match against ground truth where some items may have no position (they stay unmatched)."""
    usable = [j for j, p in enumerate(gt_pts) if p is not None]
    sub = _match_points_by_distance(det_pts, [gt_pts[j] for j in usable], max_dist)  # type: ignore[misc]
    return [(i, usable[k], d) for i, k, d in sub]


def evaluate_measured_room_accuracy(
    world: WorldIR,
    ground_truth: dict[str, Any],
    tolerances: Section1Tolerances | None = None,
) -> RoomAccuracyReport:
    """Evaluate reconstructed WorldIR against ground truth specifications under Section 1 tolerances.

    Every ground-truth element is accounted for: unmatched ones land in ``missing``, unmatched
    detections in ``extra``, and per-class precision/recall/F1 in ``class_scores``.

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

    room_entities = [e for e in world.entities.values() if e.type == EntityType.ROOM]
    report = RoomAccuracyReport(room_id=room_id, room_name=room_name, tolerances=tolerances)
    report.room_found = bool(room_entities)

    if room_entities:
        room_ent = room_entities[0]
        rcp = room_ent.custom_properties or {}
        room_kw = {"element_id": room_ent.id, "element_type": "room"}

        # 1. Floor area
        gt_area = float(gt_room.get("area_m2", 0.0))
        if gt_area > 0:
            unc_val = rcp.get("dimensions", {}).get("floor_area_m2", {}).get("uncertainty_pct", 0.5)
            unc_str = f"+-{unc_val:.1f}%" if isinstance(unc_val, (int, float)) else "+-0.5%"
            _add_dim_check(
                report, **room_kw, dimension_name="floor_area",
                measured=float(rcp.get("floor_area_m2") or 0.0), ground_truth=gt_area, unit="m2",
                tolerance=tolerances.room_area_pct, uncertainty=unc_str, pct_tolerance=True,
            )

        # 2. Ceiling height
        gt_height = float(gt_room.get("height_m", 0.0))
        if gt_height > 0:
            _add_dim_check(
                report, **room_kw, dimension_name="ceiling_height",
                measured=float(rcp.get("height_m") or 0.0), ground_truth=gt_height, unit="m",
                tolerance=tolerances.ceiling_height_m, uncertainty="+-0.015 m",
            )

        # 3. Overall room plan dimensions (length and width)
        gt_len = float(gt_room.get("length_m", 0.0))
        gt_wid = float(gt_room.get("width_m", 0.0))
        if gt_len > 0 and gt_wid > 0:
            meas_u = float(rcp.get("floor_dimension_u_m") or 0.0)
            meas_v = float(rcp.get("floor_dimension_v_m") or 0.0)
            for name, meas, gt_v in (
                ("room_length", max(meas_u, meas_v), gt_len),
                ("room_width", min(meas_u, meas_v), gt_wid),
            ):
                _add_dim_check(
                    report, **room_kw, dimension_name=name, measured=meas, ground_truth=gt_v,
                    unit="m", tolerance=tolerances.wall_length_m, uncertainty="+-0.010 m",
                )

    # 4. Individual wall lengths
    gt_walls = ground_truth.get("walls", [])
    wall_entities = [
        e
        for e in world.entities.values()
        if e.type == EntityType.WALL and "wall_slab" in (e.semantic_labels or [])
    ]
    if not wall_entities:
        wall_entities = [e for e in world.entities.values() if e.type == EntityType.WALL]

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
    _record_matching(
        report,
        "wall",
        [str(gw.get("id", gw.get("side", j))) for j, gw in enumerate(gt_walls)],
        [w.id for w in wall_entities],
        wall_matches,
    )

    for i, j, _ in wall_matches:
        w_ent = wall_entities[i]
        gw = gt_walls[j]
        gt_wall_len = float(gw.get("length_m", 0.0))
        cp = w_ent.custom_properties or {}
        slab = cp.get("wall_slab") or {}
        if gt_wall_len > 0:
            _add_dim_check(
                report, element_id=w_ent.id, element_type="wall", dimension_name="length",
                measured=float(cp.get("length_m") or slab.get("length_m") or 0.0),
                ground_truth=gt_wall_len, unit="m", tolerance=tolerances.wall_length_m,
                uncertainty="+-0.012 m", notes=f"matched GT {gw.get('id', gw.get('side', j))}",
            )

    # 5. Openings (doors and windows)
    gt_openings = ground_truth.get("openings", [])

    def _opening_pos(o_ent: Any) -> tuple[float, float, float]:
        pos = (o_ent.custom_properties or {}).get("position")
        if pos:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        return (0.0, 0.0, 0.0)

    for kind in ("door", "window"):
        kind_gt = [o for o in gt_openings if o.get("kind") == kind]
        kind_ents = [e for e in world.entities.values() if e.type == EntityType(kind)]
        meas_op_pts = [_opening_pos(e) for e in kind_ents]
        gt_op_pts = [
            (float(o["center"][0]), float(o["center"][1]), float(o["center"][2]))
            if "center" in o
            else None
            for o in kind_gt
        ]
        op_matches = _match_with_optional_positions(meas_op_pts, gt_op_pts, max_dist=1.2)
        _record_matching(
            report,
            kind,
            [str(o.get("id", f"{kind}-{j}")) for j, o in enumerate(kind_gt)],
            [e.id for e in kind_ents],
            op_matches,
        )

        for i, j, _ in op_matches:
            o_ent = kind_ents[i]
            go = kind_gt[j]
            cp = o_ent.custom_properties or {}
            for dim, key, tol in (
                ("width", "width_m", tolerances.opening_width_m),
                ("height", "height_m", tolerances.opening_height_m),
            ):
                gt_v = float(go.get(key, 0.0))
                if gt_v > 0:
                    _add_dim_check(
                        report, element_id=o_ent.id, element_type=kind, dimension_name=dim,
                        measured=float(cp.get(key) or 0.0), ground_truth=gt_v, unit="m",
                        tolerance=tol, uncertainty="+-0.015 m",
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
            if "position_m" in gf
            else None
            for gf in gt_fixtures
        ]
        fix_matches = _match_with_optional_positions(meas_f_pts, gt_f_pts, max_dist=0.5)
        _record_matching(
            report,
            "fixture",
            [str(gf.get("id", f"fixture-{j}")) for j, gf in enumerate(gt_fixtures)],
            [f.id for f in meas_fixtures],
            fix_matches,
        )

        for i, j, dist in fix_matches:
            f_ent = meas_fixtures[i]
            gf = gt_fixtures[j]
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
                    passed=dist <= tolerances.fixture_pos_m,
                    notes=f"type={gf.get('fixture_type', f_ent.type.value)}",
                )
            )

    report.compute_summary()
    return report
