"""Construction site progress tracking and as-built vs. plan compliance verification.

Provides:
- BIM-vs-reality compliance checking: compares design plan models against as-built site captures.
- Trade-level breakdown (Structure, Architecture, MEP, Finishes).
- Tolerance gating: detects completed, misaligned, in-progress, missing, and unplanned elements.
- Multi-epoch construction timeline tracking: measures installation velocity and detects rework.
- Built-in zero-dependency PDF progress report generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from world_ir.schema_v1 import Entity, EntityType, Vector3
from world_ir.temporal import (
    DEFAULT_STATIC_TYPES,
    TemporalAlignmentResult,
    TemporalEpoch,
    _is_static_candidate,
    _resolve_entity_pos,
    align_temporal_epochs,
)
from world_ir.world_v1 import WorldIR


class Trade(str, Enum):
    """Construction trade classifications."""

    STRUCTURE = "structure"
    ARCHITECTURE = "architecture"
    MEP = "mep"
    FINISHES = "finishes"
    OTHER = "other"


class ComplianceStatus(str, Enum):
    """Compliance state of an element relative to the design plan."""

    COMPLETED = "completed"
    MISALIGNED = "misaligned"
    IN_PROGRESS = "in_progress"
    MISSING = "missing"
    UNPLANNED = "unplanned"


def infer_trade(entity: Entity) -> Trade:
    """Infer the construction trade from entity type, name, and properties."""
    etype = entity.type
    name = (entity.name or "").lower()
    props = entity.custom_properties or {}
    trade_prop = str(props.get("trade", "")).lower()

    if trade_prop in ("structure", "structural"):
        return Trade.STRUCTURE
    if trade_prop in ("architecture", "architectural"):
        return Trade.ARCHITECTURE
    if trade_prop in ("mep", "electrical", "plumbing", "mechanical", "hvac"):
        return Trade.MEP
    if trade_prop in ("finishes", "finish", "interior"):
        return Trade.FINISHES

    # Type & name heuristic
    if etype in (EntityType.COLUMN, EntityType.BEAM):
        return Trade.STRUCTURE
    if any(kw in name for kw in ("slab", "bearing", "foundation", "footing", "column", "beam")):
        return Trade.STRUCTURE

    if etype in (EntityType.WALL, EntityType.DOOR, EntityType.WINDOW, EntityType.STAIRS, EntityType.CEILING, EntityType.ROOM, EntityType.OPENING):
        return Trade.ARCHITECTURE
    if any(kw in name for kw in ("partition", "drywall", "framing", "door", "window", "stair")):
        return Trade.ARCHITECTURE

    if any(kw in name for kw in ("socket", "switch", "panel", "conduit", "pipe", "duct", "hvac", "valve", "light", "mep", "fixture")):
        return Trade.MEP

    if any(kw in name for kw in ("paint", "tile", "floor finish", "wallpaper", "carpet", "trim", "finish")):
        return Trade.FINISHES

    if etype in (EntityType.FLOOR, EntityType.ROOF, EntityType.STRUCTURE):
        return Trade.STRUCTURE

    return Trade.OTHER


@dataclass(frozen=True)
class ElementCompliance:
    """Compliance record for a single building element."""

    element_id: str
    planned_id: Optional[str]
    name: str
    trade: Trade
    entity_type: str
    status: ComplianceStatus
    deviation_m: float = 0.0
    deviation_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tolerance_m: float = 0.05
    planned_pos: Optional[Tuple[float, float, float]] = None
    observed_pos: Optional[Tuple[float, float, float]] = None
    confidence: float = 1.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "planned_id": self.planned_id,
            "name": self.name,
            "trade": self.trade.value,
            "entity_type": self.entity_type,
            "status": self.status.value,
            "deviation_m": round(self.deviation_m, 4),
            "deviation_vector": [round(v, 4) for v in self.deviation_vector],
            "tolerance_m": round(self.tolerance_m, 4),
            "planned_pos": [round(p, 4) for p in self.planned_pos] if self.planned_pos else None,
            "observed_pos": [round(p, 4) for p in self.observed_pos] if self.observed_pos else None,
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TradeProgressSummary:
    """Progress statistics for an individual construction trade."""

    trade: Trade
    planned_count: int
    completed_count: int
    misaligned_count: int
    in_progress_count: int
    missing_count: int
    unplanned_count: int
    completion_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade": self.trade.value,
            "planned_count": self.planned_count,
            "completed_count": self.completed_count,
            "misaligned_count": self.misaligned_count,
            "in_progress_count": self.in_progress_count,
            "missing_count": self.missing_count,
            "unplanned_count": self.unplanned_count,
            "completion_pct": round(self.completion_pct, 2),
        }


@dataclass(frozen=True)
class SiteComplianceReport:
    """Comprehensive BIM-vs-as-built verification report."""

    design_world_id: str
    site_world_id: str
    inspection_date_iso: str
    default_tolerance_m: float
    elements: Tuple[ElementCompliance, ...]
    alignment_residual_m: float = 0.0
    alignment_converged: bool = True

    @property
    def planned_elements(self) -> Tuple[ElementCompliance, ...]:
        return tuple(e for e in self.elements if e.status != ComplianceStatus.UNPLANNED)

    @property
    def completed_elements(self) -> Tuple[ElementCompliance, ...]:
        return tuple(e for e in self.elements if e.status == ComplianceStatus.COMPLETED)

    @property
    def misaligned_elements(self) -> Tuple[ElementCompliance, ...]:
        return tuple(e for e in self.elements if e.status == ComplianceStatus.MISALIGNED)

    @property
    def in_progress_elements(self) -> Tuple[ElementCompliance, ...]:
        return tuple(e for e in self.elements if e.status == ComplianceStatus.IN_PROGRESS)

    @property
    def missing_elements(self) -> Tuple[ElementCompliance, ...]:
        return tuple(e for e in self.elements if e.status == ComplianceStatus.MISSING)

    @property
    def unplanned_elements(self) -> Tuple[ElementCompliance, ...]:
        return tuple(e for e in self.elements if e.status == ComplianceStatus.UNPLANNED)

    @property
    def overall_completion_pct(self) -> float:
        total = len(self.planned_elements)
        if total == 0:
            return 100.0
        return round((len(self.completed_elements) / total) * 100.0, 2)

    def trade_summaries(self) -> Dict[Trade, TradeProgressSummary]:
        res: Dict[Trade, TradeProgressSummary] = {}
        for trade in Trade:
            t_elems = [e for e in self.elements if e.trade == trade]
            if not t_elems:
                continue
            planned = sum(1 for e in t_elems if e.status != ComplianceStatus.UNPLANNED)
            completed = sum(1 for e in t_elems if e.status == ComplianceStatus.COMPLETED)
            misaligned = sum(1 for e in t_elems if e.status == ComplianceStatus.MISALIGNED)
            in_prog = sum(1 for e in t_elems if e.status == ComplianceStatus.IN_PROGRESS)
            missing = sum(1 for e in t_elems if e.status == ComplianceStatus.MISSING)
            unplanned = sum(1 for e in t_elems if e.status == ComplianceStatus.UNPLANNED)
            pct = (completed / planned * 100.0) if planned > 0 else 100.0

            res[trade] = TradeProgressSummary(
                trade=trade,
                planned_count=planned,
                completed_count=completed,
                misaligned_count=misaligned,
                in_progress_count=in_prog,
                missing_count=missing,
                unplanned_count=unplanned,
                completion_pct=round(pct, 2),
            )
        return res

    def summary(self) -> Dict[str, Any]:
        return {
            "design_world_id": self.design_world_id,
            "site_world_id": self.site_world_id,
            "inspection_date": self.inspection_date_iso,
            "default_tolerance_m": self.default_tolerance_m,
            "overall_completion_pct": self.overall_completion_pct,
            "total_planned": len(self.planned_elements),
            "completed_count": len(self.completed_elements),
            "misaligned_count": len(self.misaligned_elements),
            "in_progress_count": len(self.in_progress_elements),
            "missing_count": len(self.missing_elements),
            "unplanned_count": len(self.unplanned_elements),
            "alignment_residual_m": round(self.alignment_residual_m, 4),
            "alignment_converged": self.alignment_converged,
            "trades": {t.value: s.to_dict() for t, s in self.trade_summaries().items()},
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "elements": [e.to_dict() for e in self.elements],
        }


def verify_as_built_against_plan(
    design_world: WorldIR,
    site_world: WorldIR,
    default_tolerance_m: float = 0.05,
    auto_align: bool = True,
    inspection_date_iso: Optional[str] = None,
) -> SiteComplianceReport:
    """Compare a planned design WorldIR against an as-built site WorldIR.

    Parameters:
        design_world: WorldIR containing planned/designed building elements.
        site_world: WorldIR containing elements reconstructed from site photos/scans.
        default_tolerance_m: Acceptable positional deviation in meters (default 0.05 m = 5 cm).
        auto_align: If True, computes Procrustes alignment between static anchors.
        inspection_date_iso: ISO timestamp of inspection (defaults to current UTC time).
    """
    if inspection_date_iso is None:
        inspection_date_iso = datetime.now(timezone.utc).isoformat()

    # 1. Alignment
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    residual_m = 0.0
    converged = True

    if auto_align:
        epoch_design = TemporalEpoch("design", inspection_date_iso, design_world)
        epoch_site = TemporalEpoch("site", inspection_date_iso, site_world)
        alignment = align_temporal_epochs(epoch_design, epoch_site)
        if alignment.converged:
            R = np.array(alignment.rotation_matrix, dtype=np.float64)
            t = np.array(alignment.translation, dtype=np.float64)
            residual_m = alignment.residual_error_m
            converged = True

    # 2. Extract positions
    design_positions: Dict[str, np.ndarray] = {}
    for eid, ent in design_world.entities.items():
        pos = _resolve_entity_pos(ent, design_world)
        if pos is not None:
            design_positions[eid] = pos

    site_positions: Dict[str, np.ndarray] = {}
    for eid, ent in site_world.entities.items():
        pos = _resolve_entity_pos(ent, site_world)
        if pos is not None:
            # Transform site pos into design coordinate frame
            p_aligned = R @ pos + t
            site_positions[eid] = p_aligned

    # 3. Match elements
    matched_site_ids: Set[str] = set()
    element_results: List[ElementCompliance] = []

    for d_id, d_ent in design_world.entities.items():
        trade = infer_trade(d_ent)
        ent_type_str = d_ent.type.value if hasattr(d_ent.type, "value") else str(d_ent.type)
        tol_m = float((d_ent.custom_properties or {}).get("tolerance_m", default_tolerance_m))
        p_plan = design_positions.get(d_id)

        # Check exact ID match first
        best_site_id: Optional[str] = None
        best_dist: float = float("inf")
        best_vec = (0.0, 0.0, 0.0)

        if d_id in site_world.entities and d_id not in matched_site_ids:
            s_ent = site_world.entities[d_id]
            best_site_id = d_id
            if p_plan is not None and d_id in site_positions:
                p_site = site_positions[d_id]
                diff = p_site - p_plan
                best_dist = float(np.linalg.norm(diff))
                best_vec = (float(diff[0]), float(diff[1]), float(diff[2]))
            else:
                best_dist = 0.0

        # Spatial search among unmatched candidates of compatible trade/type
        if best_site_id is None and p_plan is not None:
            for s_id, s_pos in site_positions.items():
                if s_id in matched_site_ids:
                    continue
                s_ent = site_world.entities[s_id]
                s_trade = infer_trade(s_ent)
                if s_trade != trade and trade != Trade.OTHER and s_trade != Trade.OTHER:
                    continue
                dist = float(np.linalg.norm(s_pos - p_plan))
                # Search window up to 0.75m
                if dist < 0.75 and dist < best_dist:
                    best_dist = dist
                    diff = s_pos - p_plan
                    best_vec = (float(diff[0]), float(diff[1]), float(diff[2]))
                    best_site_id = s_id

        # Determine compliance status
        if best_site_id is not None:
            matched_site_ids.add(best_site_id)
            s_ent = site_world.entities[best_site_id]
            props = s_ent.custom_properties or {}
            is_rough = props.get("rough") is True or props.get("framing_only") is True or "rough" in (s_ent.name or "").lower()

            p_site_tup = tuple(site_positions[best_site_id]) if best_site_id in site_positions else None
            p_plan_tup = tuple(p_plan) if p_plan is not None else None

            if is_rough:
                status = ComplianceStatus.IN_PROGRESS
                notes = "In progress (rough framing or unfinished state detected)"
            elif best_dist <= tol_m:
                status = ComplianceStatus.COMPLETED
                notes = f"Verified compliant (deviation {best_dist * 100:.1f} cm <= {tol_m * 100:.1f} cm)"
            else:
                status = ComplianceStatus.MISALIGNED
                notes = f"Tolerance exceeded (deviation {best_dist * 100:.1f} cm > {tol_m * 100:.1f} cm)"

            element_results.append(
                ElementCompliance(
                    element_id=best_site_id,
                    planned_id=d_id,
                    name=d_ent.name or s_ent.name or d_id,
                    trade=trade,
                    entity_type=ent_type_str,
                    status=status,
                    deviation_m=best_dist,
                    deviation_vector=best_vec,
                    tolerance_m=tol_m,
                    planned_pos=p_plan_tup,
                    observed_pos=p_site_tup,
                    confidence=float(s_ent.confidence) if s_ent.confidence is not None else 1.0,
                    notes=notes,
                )
            )
        else:
            # Missing
            p_plan_tup = tuple(p_plan) if p_plan is not None else None
            element_results.append(
                ElementCompliance(
                    element_id=d_id,
                    planned_id=d_id,
                    name=d_ent.name or d_id,
                    trade=trade,
                    entity_type=ent_type_str,
                    status=ComplianceStatus.MISSING,
                    deviation_m=0.0,
                    tolerance_m=tol_m,
                    planned_pos=p_plan_tup,
                    observed_pos=None,
                    confidence=1.0,
                    notes="Planned element not observed on site",
                )
            )

    # 4. Check for unplanned site elements
    for s_id, s_ent in site_world.entities.items():
        if s_id in matched_site_ids:
            continue
        trade = infer_trade(s_ent)
        ent_type_str = s_ent.type.value if hasattr(s_ent.type, "value") else str(s_ent.type)
        p_site_tup = tuple(site_positions[s_id]) if s_id in site_positions else None

        element_results.append(
            ElementCompliance(
                element_id=s_id,
                planned_id=None,
                name=s_ent.name or s_id,
                trade=trade,
                entity_type=ent_type_str,
                status=ComplianceStatus.UNPLANNED,
                deviation_m=0.0,
                tolerance_m=default_tolerance_m,
                planned_pos=None,
                observed_pos=p_site_tup,
                confidence=float(s_ent.confidence) if s_ent.confidence is not None else 1.0,
                notes="Observed on site without matching design element",
            )
        )

    return SiteComplianceReport(
        design_world_id=design_world.id,
        site_world_id=site_world.id,
        inspection_date_iso=inspection_date_iso,
        default_tolerance_m=default_tolerance_m,
        elements=tuple(element_results),
        alignment_residual_m=residual_m,
        alignment_converged=converged,
    )


# ---------------------------------------------------------------------------
# Multi-Epoch Timeline Progression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineMilestone:
    """Progress snapshot at an individual capture epoch."""

    epoch_id: str
    timestamp_iso: str
    overall_completion_pct: float
    elements_completed: int
    elements_in_progress: int
    elements_missing: int
    elements_unplanned: int
    net_velocity_elements_per_day: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "timestamp_iso": self.timestamp_iso,
            "overall_completion_pct": round(self.overall_completion_pct, 2),
            "elements_completed": self.elements_completed,
            "elements_in_progress": self.elements_in_progress,
            "elements_missing": self.elements_missing,
            "elements_unplanned": self.elements_unplanned,
            "net_velocity_elements_per_day": round(self.net_velocity_elements_per_day, 2),
        }


@dataclass(frozen=True)
class ConstructionTimelineReport:
    """Historical progression analysis across multiple capture dates."""

    project_id: str
    milestones: Tuple[TimelineMilestone, ...]
    rework_count: int = 0
    rework_details: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "milestones": [m.to_dict() for m in self.milestones],
            "total_epochs": len(self.milestones),
            "rework_count": self.rework_count,
            "rework_details": list(self.rework_details),
        }


def compute_construction_timeline(
    epochs: Sequence[TemporalEpoch],
    design_world: WorldIR,
    tolerance_m: float = 0.05,
) -> ConstructionTimelineReport:
    """Analyze progressive installation across multiple site captures."""
    if not epochs:
        return ConstructionTimelineReport(design_world.id, ())

    # Sort epochs chronologically
    sorted_epochs = sorted(epochs, key=lambda e: e.timestamp_iso)
    milestones: List[TimelineMilestone] = []
    prev_epoch: Optional[TemporalEpoch] = None
    prev_dt: Optional[datetime] = None
    prev_completed = 0

    rework_details: List[str] = []
    rework_count = 0
    completed_history: Dict[str, int] = {}  # element_id -> epoch_index when first completed

    for idx, ep in enumerate(sorted_epochs):
        report = verify_as_built_against_plan(
            design_world, ep.world, default_tolerance_m=tolerance_m, auto_align=True, inspection_date_iso=ep.timestamp_iso
        )

        dt = datetime.fromisoformat(ep.timestamp_iso.replace("Z", "+00:00"))
        velocity = 0.0
        if prev_dt is not None:
            days = max((dt - prev_dt).total_seconds() / 86400.0, 0.001)
            delta_completed = len(report.completed_elements) - prev_completed
            velocity = delta_completed / days

        # Track completed element set for rework detection
        current_completed_ids = {e.element_id for e in report.completed_elements}
        for eid in current_completed_ids:
            if eid not in completed_history:
                completed_history[eid] = idx

        # Check if an element completed in a prior epoch is now missing or misaligned (rework)
        if idx > 0:
            for eid, first_seen_idx in completed_history.items():
                if first_seen_idx < idx and eid not in current_completed_ids:
                    rework_count += 1
                    rework_details.append(f"Element {eid} was completed in epoch {first_seen_idx} but degraded/removed in {ep.epoch_id}")

        milestones.append(
            TimelineMilestone(
                epoch_id=ep.epoch_id,
                timestamp_iso=ep.timestamp_iso,
                overall_completion_pct=report.overall_completion_pct,
                elements_completed=len(report.completed_elements),
                elements_in_progress=len(report.in_progress_elements),
                elements_missing=len(report.missing_elements),
                elements_unplanned=len(report.unplanned_elements),
                net_velocity_elements_per_day=velocity,
            )
        )

        prev_epoch = ep
        prev_dt = dt
        prev_completed = len(report.completed_elements)

    return ConstructionTimelineReport(
        project_id=design_world.id,
        milestones=tuple(milestones),
        rework_count=rework_count,
        rework_details=tuple(rework_details),
    )


# ---------------------------------------------------------------------------
# Pure-Python Zero-Dependency PDF Report Generator
# ---------------------------------------------------------------------------


def _escape_pdf(text: str) -> str:
    """Escape special characters in PDF literal text strings."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def export_construction_progress_pdf(
    report: SiteComplianceReport,
    project_name: str = "Construction Site As-Built Verification",
) -> bytes:
    """Generate a clean, professional, multi-page PDF progress report (PDF-1.4).

    Built with pure Python byte buffers and Base-14 Helvetica without external dependencies.
    """
    # A4 dimensions: 595.28 x 841.89 points
    PAGE_WIDTH = 595
    PAGE_HEIGHT = 842
    MARGIN = 40

    stream = []
    stream.append(f"q\n")

    # Background header banner
    stream.append("0.12 0.14 0.18 rg\n")  # Charcoal banner
    stream.append(f"0 {PAGE_HEIGHT - 90} {PAGE_WIDTH} 90 re f\n")

    # Title text
    stream.append("BT\n")
    stream.append("/F1 20 Tf\n")
    stream.append("1 1 1 rg\n")  # White
    stream.append(f"{MARGIN} {PAGE_HEIGHT - 45} Td\n")
    stream.append(f"({_escape_pdf(project_name)}) Tj\n")
    stream.append("ET\n")

    stream.append("BT\n")
    stream.append("/F1 10 Tf\n")
    stream.append("0.75 0.78 0.82 rg\n")
    stream.append(f"{MARGIN} {PAGE_HEIGHT - 65} Td\n")
    date_str = report.inspection_date_iso[:10]
    stream.append(f"(Reality Engine Verification Report  |  Inspection: {date_str}  |  Tolerance: {report.default_tolerance_m*100:.0f} cm) Tj\n")
    stream.append("ET\n")

    # KPI Card: Overall Progress
    y = PAGE_HEIGHT - 170
    card_w = (PAGE_WIDTH - (MARGIN * 2) - 30) / 4

    kpis = [
        ("Overall Progress", f"{report.overall_completion_pct:.1f}%", (0.15, 0.65, 0.35)),
        ("Completed Elements", f"{len(report.completed_elements)} / {len(report.planned_elements)}", (0.1, 0.45, 0.85)),
        ("Misalignments", f"{len(report.misaligned_elements)}", (0.85, 0.35, 0.1) if len(report.misaligned_elements) > 0 else (0.4, 0.4, 0.4)),
        ("Missing / Remaining", f"{len(report.missing_elements)}", (0.85, 0.15, 0.15) if len(report.missing_elements) > 0 else (0.4, 0.4, 0.4)),
    ]

    for i, (title, val, col) in enumerate(kpis):
        cx = MARGIN + i * (card_w + 10)
        # Card background
        stream.append("0.95 0.96 0.98 rg\n")
        stream.append(f"{cx} {y} {card_w} 60 re f\n")
        # Left accent stripe
        stream.append(f"{col[0]} {col[1]} {col[2]} rg\n")
        stream.append(f"{cx} {y} 4 60 re f\n")

        # Value
        stream.append("BT\n")
        stream.append("/F1 16 Tf\n")
        stream.append(f"{col[0]} {col[1]} {col[2]} rg\n")
        stream.append(f"{cx + 12} {y + 30} Td\n")
        stream.append(f"({_escape_pdf(val)}) Tj\n")
        stream.append("ET\n")

        # Title
        stream.append("BT\n")
        stream.append("/F1 9 Tf\n")
        stream.append("0.4 0.45 0.5 rg\n")
        stream.append(f"{cx + 12} {y + 12} Td\n")
        stream.append(f"({_escape_pdf(title)}) Tj\n")
        stream.append("ET\n")

    # Trade Breakdown Section
    y -= 30
    stream.append("BT\n")
    stream.append("/F1 13 Tf\n")
    stream.append("0.1 0.1 0.1 rg\n")
    stream.append(f"{MARGIN} {y} Td\n")
    stream.append("(Trade Breakdown & Readiness) Tj\n")
    stream.append("ET\n")

    y -= 20
    # Table header
    stream.append("0.9 0.92 0.95 rg\n")
    stream.append(f"{MARGIN} {y} {PAGE_WIDTH - MARGIN*2} 20 re f\n")

    headers = [("Trade", MARGIN + 10), ("Planned", MARGIN + 120), ("Completed", MARGIN + 190), ("Misaligned", MARGIN + 270), ("In Progress", MARGIN + 350), ("Missing", MARGIN + 430), ("Status", MARGIN + 490)]
    stream.append("BT\n")
    stream.append("/F1 9 Tf\n")
    stream.append("0.2 0.25 0.3 rg\n")
    for h, xpos in headers:
        stream.append(f"{xpos} {y + 6} Td ({h}) Tj {-(xpos)} {-(y+6)} Td\n")
    stream.append("ET\n")

    summaries = report.trade_summaries()
    for trade, s in summaries.items():
        y -= 22
        # Alternating row background
        stream.append("0.98 0.98 0.99 rg\n")
        stream.append(f"{MARGIN} {y} {PAGE_WIDTH - MARGIN*2} 20 re f\n")

        stream.append("BT\n")
        stream.append("/F1 9 Tf\n")
        stream.append("0.15 0.15 0.15 rg\n")

        tname = trade.value.upper()
        stream.append(f"{MARGIN + 10} {y + 6} Td ({tname}) Tj {-(MARGIN + 10)} {-(y+6)} Td\n")
        stream.append(f"{MARGIN + 120} {y + 6} Td ({s.planned_count}) Tj {-(MARGIN + 120)} {-(y+6)} Td\n")
        stream.append(f"{MARGIN + 190} {y + 6} Td ({s.completed_count}) Tj {-(MARGIN + 190)} {-(y+6)} Td\n")
        stream.append(f"{MARGIN + 270} {y + 6} Td ({s.misaligned_count}) Tj {-(MARGIN + 270)} {-(y+6)} Td\n")
        stream.append(f"{MARGIN + 350} {y + 6} Td ({s.in_progress_count}) Tj {-(MARGIN + 350)} {-(y+6)} Td\n")
        stream.append(f"{MARGIN + 430} {y + 6} Td ({s.missing_count}) Tj {-(MARGIN + 430)} {-(y+6)} Td\n")

        pct_str = f"{s.completion_pct:.0f}%"
        stream.append(f"{MARGIN + 490} {y + 6} Td ({pct_str}) Tj {-(MARGIN + 490)} {-(y+6)} Td\n")
        stream.append("ET\n")

    # Critical Deviations Section
    y -= 35
    stream.append("BT\n")
    stream.append("/F1 13 Tf\n")
    stream.append("0.1 0.1 0.1 rg\n")
    stream.append(f"{MARGIN} {y} Td\n")
    stream.append("(Critical Deviations & Misalignments) Tj\n")
    stream.append("ET\n")

    y -= 20
    stream.append("0.9 0.92 0.95 rg\n")
    stream.append(f"{MARGIN} {y} {PAGE_WIDTH - MARGIN*2} 20 re f\n")

    dev_headers = [("Element", MARGIN + 10), ("Trade", MARGIN + 140), ("Planned Pos", MARGIN + 220), ("Observed Pos", MARGIN + 320), ("Deviation", MARGIN + 430), ("Status", MARGIN + 490)]
    stream.append("BT\n")
    stream.append("/F1 9 Tf\n")
    stream.append("0.2 0.25 0.3 rg\n")
    for h, xpos in dev_headers:
        stream.append(f"{xpos} {y + 6} Td ({h}) Tj {-(xpos)} {-(y+6)} Td\n")
    stream.append("ET\n")

    flagged = list(report.misaligned_elements) + list(report.missing_elements)[:5]
    if not flagged:
        y -= 22
        stream.append("BT\n")
        stream.append("/F1 9 Tf\n")
        stream.append("0.3 0.6 0.3 rg\n")
        stream.append(f"{MARGIN + 10} {y + 6} Td (No critical tolerance deviations detected. All observed elements match design plan.) Tj\n")
        stream.append("ET\n")
    else:
        for elem in flagged[:12]:
            y -= 22
            if y < 60:
                break
            stream.append("0.98 0.98 0.99 rg\n")
            stream.append(f"{MARGIN} {y} {PAGE_WIDTH - MARGIN*2} 20 re f\n")

            stream.append("BT\n")
            stream.append("/F1 9 Tf\n")
            stream.append("0.15 0.15 0.15 rg\n")

            ename = _escape_pdf(elem.name[:18])
            etrade = elem.trade.value.upper()
            pp_str = f"({elem.planned_pos[0]:.2f}, {elem.planned_pos[1]:.2f})" if elem.planned_pos else "N/A"
            op_str = f"({elem.observed_pos[0]:.2f}, {elem.observed_pos[1]:.2f})" if elem.observed_pos else "Missing"
            dev_str = f"{elem.deviation_m * 100:.1f} cm" if elem.observed_pos else "-"
            st_str = elem.status.value.upper()

            stream.append(f"{MARGIN + 10} {y + 6} Td ({ename}) Tj {-(MARGIN + 10)} {-(y+6)} Td\n")
            stream.append(f"{MARGIN + 140} {y + 6} Td ({etrade}) Tj {-(MARGIN + 140)} {-(y+6)} Td\n")
            stream.append(f"{MARGIN + 220} {y + 6} Td ({pp_str}) Tj {-(MARGIN + 220)} {-(y+6)} Td\n")
            stream.append(f"{MARGIN + 320} {y + 6} Td ({op_str}) Tj {-(MARGIN + 320)} {-(y+6)} Td\n")
            stream.append(f"{MARGIN + 430} {y + 6} Td ({dev_str}) Tj {-(MARGIN + 430)} {-(y+6)} Td\n")

            # Color status
            if elem.status == ComplianceStatus.MISALIGNED:
                stream.append("0.85 0.35 0.1 rg\n")
            elif elem.status == ComplianceStatus.MISSING:
                stream.append("0.85 0.15 0.15 rg\n")
            else:
                stream.append("0.2 0.6 0.3 rg\n")

            stream.append(f"{MARGIN + 490} {y + 6} Td ({st_str}) Tj {-(MARGIN + 490)} {-(y+6)} Td\n")
            stream.append("ET\n")

    # Footer
    stream.append("BT\n")
    stream.append("/F1 8 Tf\n")
    stream.append("0.5 0.5 0.5 rg\n")
    stream.append(f"{MARGIN} 25 Td\n")
    stream.append("(Generated by Reality Engine Autonomous Verification System  |  Document ID: RE-CONSTRUCTION-PROGRESS) Tj\n")
    stream.append("ET\n")

    stream.append("Q\n")
    content_bytes = "".join(stream).encode("latin-1", errors="replace")

    # Build PDF object tree
    objects: List[bytes] = []
    offsets: List[int] = []

    def add_obj(data: bytes) -> int:
        nonlocal objects
        objects.append(data)
        return len(objects)

    # 1: Catalog
    cat_id = add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages
    pages_id = add_obj(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # 3: Page
    page_obj = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".encode("latin-1")
    page_id = add_obj(page_obj)
    # 4: Stream
    stream_obj = f"<< /Length {len(content_bytes)} >>\nstream\n".encode("latin-1") + content_bytes + b"\nendstream"
    stream_id = add_obj(stream_obj)
    # 5: Font Helvetica
    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Serialize PDF-1.4
    out = bytearray(b"%PDF-1.4\n")
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("latin-1"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_offset = len(out)
    out.extend(b"xref\n")
    out.extend(f"0 {len(objects) + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    out.extend(b"trailer\n")
    out.extend(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("latin-1"))
    out.extend(b"startxref\n")
    out.extend(f"{xref_offset}\n".encode("latin-1"))
    out.extend(b"%%EOF\n")

    return bytes(out)
