"""Multi-storey building vertical alignment and structural load path validator (Mandate Phase 6).

Verifies vertical geometric consistency of multi-storey buildings:
1. Floor-to-ceiling elevation monotonicity and code-compliant clearance (>= 2.10m).
2. Inter-storey vertical circulation continuity (stair spans between adjacent storeys).
3. Structural load path alignment (column-to-column vertical axis offset <= 0.25m).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType


@dataclass(frozen=True)
class StoreyElevationCheck:
    """Evaluation of an individual building storey level."""

    storey_id: str
    floor_height_m: float
    ceiling_height_m: float
    clearance_m: float
    has_valid_clearance: bool


@dataclass
class MultiStoreyValidationReport:
    """Comprehensive validation report for multi-storey vertical coherence."""

    total_storeys: int
    is_vertically_consistent: bool
    elevation_monotonic: bool
    clearance_checks_passed: bool
    stair_circulation_connected: bool
    load_bearing_overlap_pct: float
    storey_checks: list[StoreyElevationCheck] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    verdict: str = "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_storeys": self.total_storeys,
            "is_vertically_consistent": self.is_vertically_consistent,
            "elevation_monotonic": self.elevation_monotonic,
            "clearance_checks_passed": self.clearance_checks_passed,
            "stair_circulation_connected": self.stair_circulation_connected,
            "load_bearing_overlap_pct": round(self.load_bearing_overlap_pct, 1),
            "storey_checks": [asdict(c) for c in self.storey_checks],
            "violations": self.violations,
            "verdict": self.verdict,
        }


class MultiStoreyValidator:
    """Validates vertical elevation, stair connectivity, and column load paths across storeys."""

    MIN_CLEARANCE_HEIGHT_M = 2.10  # Minimum architectural code clearance
    MAX_COLUMN_AXIS_OFFSET_M = 0.25  # Max horizontal misalignment for continuous load paths

    @classmethod
    def validate(cls, world: WorldIR) -> MultiStoreyValidationReport:
        """Inspect world entities and validate multi-storey architectural hierarchy."""
        storeys: list[tuple[float, Entity]] = []

        # Find all storey/level entities
        for ent in world.entities.values():
            if ent.type in (EntityType.STOREY, getattr(EntityType, "LEVEL", EntityType.STOREY)):
                cp = ent.custom_properties or {}
                fh = cp.get("floor_height_m") if cp.get("floor_height_m") is not None else cp.get("height_m")
                if fh is None and ent.geometry_ids:
                    geom = world.geometries.get(ent.geometry_ids[0])
                    if geom and geom.bounds_min:
                        fh = geom.bounds_min.z
                if fh is not None:
                    storeys.append((float(fh), ent))

        # If no explicit STOREY entities, check for rooms with distinct floor heights
        if not storeys:
            room_floors: dict[str, tuple[float, Entity]] = {}
            for ent in world.entities.values():
                if ent.type == EntityType.ROOM:
                    cp = ent.custom_properties or {}
                    raw_h = cp.get("floor_height_m")
                    fh = float(raw_h) if raw_h is not None else 0.0
                    room_floors[ent.id] = (fh, ent)
            # Group into unique elevation clusters (tolerance 0.25m)
            sorted_rooms = sorted(room_floors.values(), key=lambda x: x[0])
            for fh, rent in sorted_rooms:
                if not storeys or abs(fh - storeys[-1][0]) > 0.50:
                    storeys.append((fh, rent))

        # Single-storey or zero storeys: trivially consistent
        if len(storeys) <= 1:
            return MultiStoreyValidationReport(
                total_storeys=len(storeys),
                is_vertically_consistent=True,
                elevation_monotonic=True,
                clearance_checks_passed=True,
                stair_circulation_connected=True,
                load_bearing_overlap_pct=100.0,
                storey_checks=[
                    StoreyElevationCheck(
                        storey_id=storeys[0][1].id if storeys else "ground",
                        floor_height_m=storeys[0][0] if storeys else 0.0,
                        ceiling_height_m=(storeys[0][0] + 2.7) if storeys else 2.7,
                        clearance_m=2.7,
                        has_valid_clearance=True,
                    )
                ] if storeys else [],
                verdict="PASS",
            )

        # Sort storeys bottom-up
        storeys.sort(key=lambda s: s[0])
        n = len(storeys)

        violations: list[str] = []
        storey_checks: list[StoreyElevationCheck] = []
        is_monotonic = True
        clearance_ok = True

        for i in range(n):
            curr_h, curr_ent = storeys[i]
            cp = curr_ent.custom_properties or {}
            nominal_ceil = cp.get("ceiling_height_m")

            if i < n - 1:
                next_h, next_ent = storeys[i + 1]
                if next_h <= curr_h:
                    is_monotonic = False
                    violations.append(
                        f"Non-monotonic elevation: Storey '{next_ent.id}' ({next_h:.2f}m) <= '{curr_ent.id}' ({curr_h:.2f}m)"
                    )
                clearance = next_h - curr_h
                ceil_h = next_h
            else:
                clearance = float(nominal_ceil or 2.70)
                ceil_h = curr_h + clearance

            valid_cl = clearance >= cls.MIN_CLEARANCE_HEIGHT_M
            if not valid_cl:
                clearance_ok = False
                violations.append(
                    f"Insufficient storey clearance: Storey '{curr_ent.id}' clearance is {clearance:.2f}m (< {cls.MIN_CLEARANCE_HEIGHT_M:.2f}m)"
                )

            storey_checks.append(
                StoreyElevationCheck(
                    storey_id=curr_ent.id,
                    floor_height_m=round(curr_h, 3),
                    ceiling_height_m=round(ceil_h, 3),
                    clearance_m=round(clearance, 3),
                    has_valid_clearance=valid_cl,
                )
            )

        # 2. Check vertical circulation (stairs)
        stairs = [e for e in world.entities.values() if e.type == EntityType.STAIRS]
        stair_ok = True
        if n >= 2:
            spanned_levels = set()
            for st in stairs:
                cp = st.custom_properties or {}
                # Check custom properties for storey links
                spanned = cp.get("spanned_storeys") or cp.get("storey_ids")
                if spanned and isinstance(spanned, (list, tuple)):
                    spanned_levels.update(spanned)
                elif st.geometry_ids:
                    geom = world.geometries.get(st.geometry_ids[0])
                    if geom and geom.bounds_min and geom.bounds_max:
                        z_min, z_max = geom.bounds_min.z, geom.bounds_max.z
                        for s_idx in range(n - 1):
                            if z_min <= storeys[s_idx][0] + 0.3 and z_max >= storeys[s_idx + 1][0] - 0.3:
                                spanned_levels.add(storeys[s_idx][1].id)
                                spanned_levels.add(storeys[s_idx + 1][1].id)

            # If there are stairs, verify they span all intermediate storeys
            if stairs and len(spanned_levels) < n:
                stair_ok = False
                violations.append(
                    f"Stair circulation incomplete: {len(spanned_levels)}/{n} storeys connected by vertical stairs"
                )

        # 3. Check structural load path alignment (columns)
        columns = [e for e in world.entities.values() if e.type == EntityType.COLUMN]
        load_path_pct = 100.0
        if columns and n >= 2:
            # Group columns by storey elevation
            cols_by_storey: dict[int, list[tuple[float, float]]] = {}
            for col in columns:
                if col.geometry_ids:
                    g = world.geometries.get(col.geometry_ids[0])
                    if g and g.bounds_min and g.bounds_max:
                        cz = (g.bounds_min.z + g.bounds_max.z) / 2.0
                        cx = (g.bounds_min.x + g.bounds_max.x) / 2.0
                        cy = (g.bounds_min.y + g.bounds_max.y) / 2.0
                        # Assign to closest storey
                        s_idx = min(range(n), key=lambda idx: abs(storeys[idx][0] - cz))
                        cols_by_storey.setdefault(s_idx, []).append((cx, cy))

            # Check overlap between adjacent storeys
            matched_pairs = 0
            total_upper = 0
            for s_idx in range(1, n):
                lower_cols = cols_by_storey.get(s_idx - 1, [])
                upper_cols = cols_by_storey.get(s_idx, [])
                total_upper += len(upper_cols)
                for ux, uy in upper_cols:
                    if any(math.hypot(ux - lx, uy - ly) <= cls.MAX_COLUMN_AXIS_OFFSET_M for lx, ly in lower_cols):
                        matched_pairs += 1

            if total_upper > 0:
                load_path_pct = (matched_pairs / total_upper) * 100.0
                if load_path_pct < 70.0:
                    violations.append(
                        f"Poor column load path alignment: only {load_path_pct:.1f}% upper columns align with lower columns"
                    )

        consistent = is_monotonic and clearance_ok and stair_ok

        return MultiStoreyValidationReport(
            total_storeys=n,
            is_vertically_consistent=consistent,
            elevation_monotonic=is_monotonic,
            clearance_checks_passed=clearance_ok,
            stair_circulation_connected=stair_ok,
            load_bearing_overlap_pct=load_path_pct,
            storey_checks=storey_checks,
            violations=violations,
            verdict="PASS" if consistent else "FAIL",
        )
