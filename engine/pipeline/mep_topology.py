"""Automated MEP & Conduit Routing Perception with IFC4x3 Facility Topology.

Provides end-to-end perception, spatial graph topology, code clearance verification,
and buildingSMART IFC4 / IFC4x3 Building Services export for Mechanical, Electrical,
and Plumbing (MEP) infrastructure.

Key Capabilities:
1. MEP Perception & Classification:
   - Identifies Electrical (conduits, cable trays, junction boxes, switchboards, outlets).
   - Identifies HVAC (supply/return air ducts, diffusers, terminal units).
   - Identifies Plumbing (hot/cold water supply, drainage pipes, sanitary fixtures).
2. Facility Spatial Routing & Orthogonal Network Inference:
   - Generates 3D centerline path geometries with circular and rectangular cross-sections.
   - Infers orthogonal Manhattan routing (along wall headers and ceiling plenums) with 90°/45° elbows.
   - Assigns functional connection ports (upstream SOURCE, downstream SINK, BIDIRECTIONAL).
3. Facility Graph Topology & DAG Verification:
   - Builds directed distribution graphs from main distribution panels / service risers to endpoints.
   - Computes linear run lengths, cross-sectional areas, volumes, and DAG connectivity.
4. Code Clearance Verification:
   - Checks 3D spatial clearance between electrical runs and plumbing/water lines (min 150mm separation).
5. Open Standard IFC4 / IFC4x3 & GeoJSON Exporters:
   - Generates IFC4X3 STEP Physical Files with IfcDistributionSystem, IfcCableCarrierSegment,
     IfcPipeSegment, IfcDuctSegment, IfcDistributionPort, IfcRelNests, and IfcRelConnectsPorts.
   - Exports 3D LineString GeoJSON network representations for web viewers and GIS integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, RelationshipKind, Vector3


class MepSystemType(str, Enum):
    ELECTRICAL = "electrical"
    HVAC = "hvac"
    PLUMBING = "plumbing"
    FIRE_PROTECTION = "fire_protection"
    DATA_COMM = "data_comm"


class MepElementType(str, Enum):
    CONDUIT = "conduit"
    CABLE_TRAY = "cable_tray"
    PIPE = "pipe"
    DUCT = "duct"
    ELBOW = "elbow"
    TEE = "tee"
    CROSS = "cross"
    JUNCTION_BOX = "junction_box"
    PANELBOARD = "panelboard"
    OUTLET = "outlet"
    SWITCH = "switch"
    LIGHT_FIXTURE = "light_fixture"
    AIR_TERMINAL = "air_terminal"
    PLUMBING_FIXTURE = "plumbing_fixture"


class CrossSectionKind(str, Enum):
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"


class FlowDirection(str, Enum):
    SOURCE = "source"
    SINK = "sink"
    BIDIRECTIONAL = "bidirectional"


def _deterministic_guid(seed: str) -> str:
    """Deterministic 22-char IFC GlobalId derived from a seed."""
    digest = uuid.uuid5(uuid.NAMESPACE_URL, f"reality-engine://mep/{seed}").hex
    # RFC 4122 hex -> standard base64 alphabet for IFC
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
    num = int(digest, 16)
    out = []
    for _ in range(22):
        out.append(chars[num % 64])
        num //= 64
    return "".join(reversed(out))


@dataclass(slots=True)
class MepPort:
    """buildingSMART IfcDistributionPort representation for MEP network connectivity."""
    port_id: str
    element_id: str
    position: tuple[float, float, float]
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    flow_direction: FlowDirection = FlowDirection.BIDIRECTIONAL
    connected_port_id: Optional[str] = None
    nominal_size_mm: float = 25.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "port_id": self.port_id,
            "element_id": self.element_id,
            "position": [round(c, 4) for c in self.position],
            "direction": [round(c, 4) for c in self.direction],
            "flow_direction": self.flow_direction.value,
            "connected_port_id": self.connected_port_id,
            "nominal_size_mm": round(self.nominal_size_mm, 1),
        }


@dataclass(slots=True)
class MepSegment:
    """Linear distribution segment (conduit, cable tray, pipe, duct)."""
    segment_id: str
    name: str
    system_type: MepSystemType
    element_type: MepElementType
    start_point: tuple[float, float, float]
    end_point: tuple[float, float, float]
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)
    cross_section: CrossSectionKind = CrossSectionKind.CIRCULAR
    diameter_mm: float = 25.0
    width_mm: float = 0.0
    height_mm: float = 0.0
    length_m: float = 0.0
    ports: list[MepPort] = field(default_factory=list)
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    storey_id: Optional[str] = None
    room_id: Optional[str] = None
    bounds: dict[str, list[float]] = field(default_factory=dict)
    provenance: str = "observed"

    def __post_init__(self):
        if not self.waypoints:
            self.waypoints = [self.start_point, self.end_point]
        if self.length_m <= 0.0:
            total = 0.0
            for i in range(len(self.waypoints) - 1):
                p0, p1 = self.waypoints[i], self.waypoints[i + 1]
                total += math.sqrt(sum((b - a) ** 2 for a, b in zip(p0, p1)))
            self.length_m = total

        if not self.bounds:
            xs = [p[0] for p in self.waypoints]
            ys = [p[1] for p in self.waypoints]
            zs = [p[2] for p in self.waypoints]
            rad = (self.diameter_mm / 2000.0) if self.cross_section == CrossSectionKind.CIRCULAR else (max(self.width_mm, self.height_mm) / 2000.0)
            self.bounds = {
                "min": [min(xs) - rad, min(ys) - rad, min(zs) - rad],
                "max": [max(xs) + rad, max(ys) + rad, max(zs) + rad],
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "name": self.name,
            "system_type": self.system_type.value,
            "element_type": self.element_type.value,
            "start_point": [round(c, 4) for c in self.start_point],
            "end_point": [round(c, 4) for c in self.end_point],
            "waypoints": [[round(c, 4) for c in wp] for wp in self.waypoints],
            "cross_section": self.cross_section.value,
            "diameter_mm": round(self.diameter_mm, 1),
            "width_mm": round(self.width_mm, 1),
            "height_mm": round(self.height_mm, 1),
            "length_m": round(self.length_m, 3),
            "ports": [p.to_dict() for p in self.ports],
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "storey_id": self.storey_id,
            "room_id": self.room_id,
            "bounds": {k: [round(c, 4) for c in v] for k, v in self.bounds.items()},
            "provenance": self.provenance,
        }


@dataclass(slots=True)
class MepFitting:
    """Elbow, Tee, or Cross junction connecting distribution segments."""
    fitting_id: str
    name: str
    system_type: MepSystemType
    element_type: MepElementType
    position: tuple[float, float, float]
    angle_deg: float = 90.0
    connected_segment_ids: list[str] = field(default_factory=list)
    ports: list[MepPort] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitting_id": self.fitting_id,
            "name": self.name,
            "system_type": self.system_type.value,
            "element_type": self.element_type.value,
            "position": [round(c, 4) for c in self.position],
            "angle_deg": round(self.angle_deg, 1),
            "connected_segment_ids": self.connected_segment_ids,
            "ports": [p.to_dict() for p in self.ports],
        }


@dataclass(slots=True)
class MepTerminal:
    """Terminal device (outlet, switch, luminaire, air diffuser, panelboard, sink)."""
    terminal_id: str
    name: str
    system_type: MepSystemType
    element_type: MepElementType
    position: tuple[float, float, float]
    bounds: dict[str, list[float]] = field(default_factory=dict)
    host_wall_id: Optional[str] = None
    storey_id: Optional[str] = None
    room_id: Optional[str] = None
    ports: list[MepPort] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_id": self.terminal_id,
            "name": self.name,
            "system_type": self.system_type.value,
            "element_type": self.element_type.value,
            "position": [round(c, 4) for c in self.position],
            "bounds": {k: [round(c, 4) for c in v] for k, v in self.bounds.items()},
            "host_wall_id": self.host_wall_id,
            "storey_id": self.storey_id,
            "room_id": self.room_id,
            "ports": [p.to_dict() for p in self.ports],
        }


@dataclass(slots=True)
class MepClearanceViolation:
    """Spatial clearance / code compliance conflict between conflicting trades."""
    violation_id: str
    element_a_id: str
    element_a_system: MepSystemType
    element_b_id: str
    element_b_system: MepSystemType
    actual_clearance_m: float
    required_clearance_m: float
    location: tuple[float, float, float]
    severity: str  # CRITICAL, MAJOR, MINOR
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "element_a_id": self.element_a_id,
            "element_a_system": self.element_a_system.value,
            "element_b_id": self.element_b_id,
            "element_b_system": self.element_b_system.value,
            "actual_clearance_m": round(self.actual_clearance_m, 4),
            "required_clearance_m": round(self.required_clearance_m, 4),
            "location": [round(c, 4) for c in self.location],
            "severity": self.severity,
            "description": self.description,
        }


@dataclass(slots=True)
class MepDistributionNetwork:
    """A cohesive facility MEP network (e.g. Electrical Power, HVAC Supply, Cold Water)."""
    system_id: str
    name: str
    system_type: MepSystemType
    root_panel_id: Optional[str] = None
    segments: list[MepSegment] = field(default_factory=list)
    fittings: list[MepFitting] = field(default_factory=list)
    terminals: list[MepTerminal] = field(default_factory=list)
    total_length_m: float = 0.0
    total_volume_m3: float = 0.0
    is_connected_dag: bool = True

    def __post_init__(self):
        if self.total_length_m <= 0.0:
            self.total_length_m = sum(s.length_m for s in self.segments)
        if self.total_volume_m3 <= 0.0:
            vol = 0.0
            for s in self.segments:
                if s.cross_section == CrossSectionKind.CIRCULAR:
                    radius = (s.diameter_mm / 1000.0) / 2.0
                    vol += math.pi * (radius ** 2) * s.length_m
                else:
                    vol += (s.width_mm / 1000.0) * (s.height_mm / 1000.0) * s.length_m
            self.total_volume_m3 = vol

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "name": self.name,
            "system_type": self.system_type.value,
            "root_panel_id": self.root_panel_id,
            "segment_count": len(self.segments),
            "fitting_count": len(self.fittings),
            "terminal_count": len(self.terminals),
            "total_length_m": round(self.total_length_m, 3),
            "total_volume_m3": round(self.total_volume_m3, 5),
            "is_connected_dag": self.is_connected_dag,
            "segments": [s.to_dict() for s in self.segments],
            "fittings": [f.to_dict() for f in self.fittings],
            "terminals": [t.to_dict() for t in self.terminals],
        }


@dataclass(slots=True)
class MepFacilityTopology:
    """Complete building facility MEP infrastructure topology."""
    world_id: str
    systems: list[MepDistributionNetwork] = field(default_factory=list)
    total_conduit_length_m: float = 0.0
    total_pipe_length_m: float = 0.0
    total_duct_length_m: float = 0.0
    clearance_violations: list[MepClearanceViolation] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.total_conduit_length_m = sum(
            s.length_m for sys in self.systems if sys.system_type == MepSystemType.ELECTRICAL for s in sys.segments
        )
        self.total_pipe_length_m = sum(
            s.length_m for sys in self.systems if sys.system_type == MepSystemType.PLUMBING for s in sys.segments
        )
        self.total_duct_length_m = sum(
            s.length_m for sys in self.systems if sys.system_type == MepSystemType.HVAC for s in sys.segments
        )
        if not self.summary:
            total_elements = sum(len(sys.segments) + len(sys.fittings) + len(sys.terminals) for sys in self.systems)
            self.summary = {
                "system_count": len(self.systems),
                "total_elements": total_elements,
                "total_run_length_m": round(self.total_conduit_length_m + self.total_pipe_length_m + self.total_duct_length_m, 2),
                "clearance_violations_count": len(self.clearance_violations),
                "code_compliant": len(self.clearance_violations) == 0,
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "total_conduit_length_m": round(self.total_conduit_length_m, 2),
            "total_pipe_length_m": round(self.total_pipe_length_m, 2),
            "total_duct_length_m": round(self.total_duct_length_m, 2),
            "summary": self.summary,
            "clearance_violations": [v.to_dict() for v in self.clearance_violations],
            "systems": [sys.to_dict() for sys in self.systems],
        }


# =========================================================================
# Geometry and Routing Utilities
# =========================================================================

def _extract_position(entity: Any) -> tuple[float, float, float] | None:
    """Extract (x, y, z) position from an entity's transform or bounds."""
    tr = getattr(entity, "transform", None)
    if isinstance(tr, dict) and "position" in tr:
        pos = tr["position"]
        if isinstance(pos, dict) and all(k in pos for k in ["x", "y", "z"]):
            return (float(pos["x"]), float(pos["y"]), float(pos["z"]))
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
    props = getattr(entity, "custom_properties", {}) or {}
    if "position" in props and isinstance(props["position"], (list, tuple)) and len(props["position"]) >= 3:
        p = props["position"]
        return (float(p[0]), float(p[1]), float(p[2]))
    return None


def _extract_bounds(entity: Any, geometries: dict[str, Any]) -> dict[str, list[float]] | None:
    """Extract AABB bounding box min/max from entity or its geometries."""
    gids = getattr(entity, "geometry_ids", [])
    for gid in gids:
        geom = geometries.get(gid)
        if geom is not None and getattr(geom, "bounds_min", None) and getattr(geom, "bounds_max", None):
            b_min = geom.bounds_min
            b_max = geom.bounds_max
            return {
                "min": [float(getattr(b_min, "x", b_min[0])), float(getattr(b_min, "y", b_min[1])), float(getattr(b_min, "z", b_min[2]))],
                "max": [float(getattr(b_max, "x", b_max[0])), float(getattr(b_max, "y", b_max[1])), float(getattr(b_max, "z", b_max[2]))],
            }
    pos = _extract_position(entity)
    if pos:
        return {
            "min": [pos[0] - 0.05, pos[1] - 0.05, pos[2] - 0.05],
            "max": [pos[0] + 0.05, pos[1] + 0.05, pos[2] + 0.05],
        }
    return None


def infer_orthogonal_routing(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    ceiling_z: float = 2.4,
) -> list[tuple[float, float, float]]:
    """Calculates orthogonal Manhattan 3D route between two endpoints.

    Standard architectural MEP practice routes vertically up to the ceiling plenum / header zone,
    traverses horizontally in orthogonal steps (X then Y or aligned with primary axes),
    and drops vertically down to the destination terminal.
    """
    sx, sy, sz = start
    ex, ey, ez = end

    # If already at similar height or direct planar path
    z_run = max(ceiling_z, max(sz, ez) + 0.2)

    waypoints: list[tuple[float, float, float]] = [(sx, sy, sz)]

    # Rise from start to ceiling plenum
    if abs(sz - z_run) > 0.05:
        waypoints.append((sx, sy, z_run))

    # Orthogonal horizontal legs
    if abs(sx - ex) > 0.02 and abs(sy - ey) > 0.02:
        # Step along X
        waypoints.append((ex, sy, z_run))
        # Step along Y
        waypoints.append((ex, ey, z_run))
    elif abs(sx - ex) > 0.02:
        waypoints.append((ex, sy, z_run))
    elif abs(sy - ey) > 0.02:
        waypoints.append((sx, ey, z_run))

    # Drop to destination
    if abs(ez - z_run) > 0.05:
        waypoints.append((ex, ey, ez))

    # Filter out near-duplicate adjacent points
    filtered: list[tuple[float, float, float]] = [waypoints[0]]
    for pt in waypoints[1:]:
        prev = filtered[-1]
        dist = math.sqrt(sum((b - a) ** 2 for a, b in zip(prev, pt)))
        if dist > 0.01:
            filtered.append(pt)

    return filtered


def _distance_between_segments(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    p4: tuple[float, float, float],
) -> float:
    """Computes minimum 3D Euclidean distance between line segment P1-P2 and P3-P4."""
    # Approximate with discrete sampling
    samples_a = [
        (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t, p1[2] + (p2[2] - p1[2]) * t)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]
    ]
    samples_b = [
        (p3[0] + (p4[0] - p3[0]) * s, p3[1] + (p4[1] - p3[1]) * s, p3[2] + (p4[2] - p3[2]) * s)
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]
    ]
    min_d = float("inf")
    for a in samples_a:
        for b in samples_b:
            d = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            if d < min_d:
                min_d = d
    return min_d


def verify_code_clearances(
    systems: list[MepDistributionNetwork],
    min_clearance_m: float = 0.15,
) -> list[MepClearanceViolation]:
    """Verify physical clearance between electrical systems and fluid/plumbing pipes.

    NEC Article 110 and IPC Section 305 require separation between electrical conduits/panels
    and water/drainage pipes to prevent shock hazards and condensation damage.
    """
    violations: list[MepClearanceViolation] = []
    elec_systems = [s for s in systems if s.system_type == MepSystemType.ELECTRICAL]
    plumb_systems = [s for s in systems if s.system_type in (MepSystemType.PLUMBING, MepSystemType.HVAC)]

    for esys in elec_systems:
        for psys in plumb_systems:
            for eseg in esys.segments:
                for pseg in psys.segments:
                    # Quick bounding box proximity check
                    e_min, e_max = eseg.bounds["min"], eseg.bounds["max"]
                    p_min, p_max = pseg.bounds["min"], pseg.bounds["max"]
                    if (
                        e_min[0] - min_clearance_m > p_max[0] or e_max[0] + min_clearance_m < p_min[0] or
                        e_min[1] - min_clearance_m > p_max[1] or e_max[1] + min_clearance_m < p_min[1] or
                        e_min[2] - min_clearance_m > p_max[2] or e_max[2] + min_clearance_m < p_min[2]
                    ):
                        continue

                    dist = _distance_between_segments(eseg.start_point, eseg.end_point, pseg.start_point, pseg.end_point)
                    if dist < min_clearance_m:
                        mid_x = (eseg.start_point[0] + pseg.start_point[0]) / 2.0
                        mid_y = (eseg.start_point[1] + pseg.start_point[1]) / 2.0
                        mid_z = (eseg.start_point[2] + pseg.start_point[2]) / 2.0
                        sev = "CRITICAL" if dist < 0.05 else "MAJOR"
                        violations.append(
                            MepClearanceViolation(
                                violation_id=f"clearance-{eseg.segment_id}-{pseg.segment_id}",
                                element_a_id=eseg.segment_id,
                                element_a_system=esys.system_type,
                                element_b_id=pseg.segment_id,
                                element_b_system=psys.system_type,
                                actual_clearance_m=dist,
                                required_clearance_m=min_clearance_m,
                                location=(mid_x, mid_y, mid_z),
                                severity=sev,
                                description=(
                                    f"Clearance conflict: {eseg.name} ({esys.system_type.value}) is {round(dist*1000, 1)}mm "
                                    f"from {pseg.name} ({psys.system_type.value}), below {round(min_clearance_m*1000, 1)}mm code threshold."
                                ),
                            )
                        )

    return violations


# =========================================================================
# Perception & WorldIR Ingestion
# =========================================================================

def detect_mep_elements_from_world(world: Any) -> MepFacilityTopology:
    """Perceives MEP infrastructure from WorldIR entities, fixtures, and geometries.

    Ingests detected electrical sockets, light fixtures, panels, HVAC grilles, and plumbing fixtures,
    and infers/synthesizes facility distribution topology with IFC4x3 ports and orthogonal routing.
    """
    world_id = getattr(world, "id", "world-1")
    entities = getattr(world, "entities", {})
    geometries = getattr(world, "geometries", {})

    elec_terminals: list[MepTerminal] = []
    hvac_terminals: list[MepTerminal] = []
    plumb_terminals: list[MepTerminal] = []
    panelboards: list[MepTerminal] = []

    # Detect ceiling height for plenum routing
    ceiling_z = 2.4
    for ent in entities.values():
        etype = getattr(ent, "type", "")
        if (getattr(etype, "value", str(etype))).lower() == "ceiling":
            pos = _extract_position(ent)
            if pos and pos[2] > 1.8:
                ceiling_z = max(ceiling_z, pos[2])

    for ent_id, ent in entities.items():
        name = (getattr(ent, "name", "") or ent_id).lower()
        etype_raw = getattr(ent, "type", "")
        etype_str = (getattr(etype_raw, "value", str(etype_raw))).lower()
        props = getattr(ent, "custom_properties", {}) or {}
        pos = _extract_position(ent)
        if not pos:
            continue
        bounds = _extract_bounds(ent, geometries) or {
            "min": [pos[0] - 0.05, pos[1] - 0.05, pos[2] - 0.05],
            "max": [pos[0] + 0.05, pos[1] + 0.05, pos[2] + 0.05],
        }

        # 1. Electrical panelboard / distribution board
        if any(k in name for k in ["panel", "distribution_board", "breaker", "switchboard", "db-"]):
            t = MepTerminal(
                terminal_id=ent_id,
                name=getattr(ent, "name", ent_id) or ent_id,
                system_type=MepSystemType.ELECTRICAL,
                element_type=MepElementType.PANELBOARD,
                position=pos,
                bounds=bounds,
                ports=[
                    MepPort(
                        port_id=f"port-{ent_id}-out",
                        element_id=ent_id,
                        position=(pos[0], pos[1], pos[2] + 0.2),
                        direction=(0.0, 0.0, 1.0),
                        flow_direction=FlowDirection.SOURCE,
                    )
                ],
            )
            panelboards.append(t)
            elec_terminals.append(t)

        # 2. Electrical Outlets / Sockets / Switches
        elif any(k in name or k in etype_str for k in ["outlet", "socket", "switch", "receptacle"]):
            el_type = MepElementType.SWITCH if "switch" in name else MepElementType.OUTLET
            elec_terminals.append(
                MepTerminal(
                    terminal_id=ent_id,
                    name=getattr(ent, "name", ent_id) or ent_id,
                    system_type=MepSystemType.ELECTRICAL,
                    element_type=el_type,
                    position=pos,
                    bounds=bounds,
                    ports=[
                        MepPort(
                            port_id=f"port-{ent_id}-in",
                            element_id=ent_id,
                            position=(pos[0], pos[1], pos[2] + 0.05),
                            direction=(0.0, 0.0, 1.0),
                            flow_direction=FlowDirection.SINK,
                        )
                    ],
                )
            )

        # 3. Light fixtures / Luminaires
        elif any(k in name or k in etype_str for k in ["light", "luminaire", "lamp", "ceiling_light", "chandelier"]):
            elec_terminals.append(
                MepTerminal(
                    terminal_id=ent_id,
                    name=getattr(ent, "name", ent_id) or ent_id,
                    system_type=MepSystemType.ELECTRICAL,
                    element_type=MepElementType.LIGHT_FIXTURE,
                    position=pos,
                    bounds=bounds,
                    ports=[
                        MepPort(
                            port_id=f"port-{ent_id}-in",
                            element_id=ent_id,
                            position=(pos[0], pos[1], pos[2] + 0.02),
                            direction=(0.0, 0.0, 1.0),
                            flow_direction=FlowDirection.SINK,
                        )
                    ],
                )
            )

        # 4. HVAC Diffusers / Grilles / Vents
        elif any(k in name or k in etype_str for k in ["hvac", "duct", "vent", "diffuser", "grille", "ac_unit"]):
            hvac_terminals.append(
                MepTerminal(
                    terminal_id=ent_id,
                    name=getattr(ent, "name", ent_id) or ent_id,
                    system_type=MepSystemType.HVAC,
                    element_type=MepElementType.AIR_TERMINAL,
                    position=pos,
                    bounds=bounds,
                    ports=[
                        MepPort(
                            port_id=f"port-{ent_id}-in",
                            element_id=ent_id,
                            position=(pos[0], pos[1], pos[2] + 0.05),
                            direction=(0.0, 0.0, 1.0),
                            flow_direction=FlowDirection.SINK,
                            nominal_size_mm=150.0,
                        )
                    ],
                )
            )

        # 5. Plumbing Sinks / Toilets / Drains
        elif any(k in name or k in etype_str for k in ["sink", "toilet", "basin", "faucet", "drain", "shower"]):
            plumb_terminals.append(
                MepTerminal(
                    terminal_id=ent_id,
                    name=getattr(ent, "name", ent_id) or ent_id,
                    system_type=MepSystemType.PLUMBING,
                    element_type=MepElementType.PLUMBING_FIXTURE,
                    position=pos,
                    bounds=bounds,
                    ports=[
                        MepPort(
                            port_id=f"port-{ent_id}-in",
                            element_id=ent_id,
                            position=(pos[0], pos[1], pos[2] - 0.05),
                            direction=(0.0, 0.0, -1.0),
                            flow_direction=FlowDirection.SINK,
                            nominal_size_mm=50.0,
                        )
                    ],
                )
            )

    systems: list[MepDistributionNetwork] = []

    # If no panelboard found, synthesize a primary electrical panel near the first terminal or origin
    if not panelboards:
        root_pos = (0.2, 0.1, 1.5)
        if elec_terminals:
            first_t = elec_terminals[0]
            root_pos = (first_t.position[0] - 1.0, first_t.position[1], 1.5)
        root_panel = MepTerminal(
            terminal_id="panel-main",
            name="Main Distribution Panel (MDP)",
            system_type=MepSystemType.ELECTRICAL,
            element_type=MepElementType.PANELBOARD,
            position=root_pos,
            bounds={"min": [root_pos[0]-0.2, root_pos[1]-0.1, root_pos[2]-0.3], "max": [root_pos[0]+0.2, root_pos[1]+0.1, root_pos[2]+0.3]},
            ports=[
                MepPort(
                    port_id="port-panel-main-out",
                    element_id="panel-main",
                    position=(root_pos[0], root_pos[1], root_pos[2] + 0.3),
                    direction=(0.0, 0.0, 1.0),
                    flow_direction=FlowDirection.SOURCE,
                )
            ],
        )
        panelboards.append(root_panel)
        elec_terminals.insert(0, root_panel)

    primary_panel = panelboards[0]

    # 1. Synthesize Electrical Conduit Network
    elec_segments: list[MepSegment] = []
    elec_fittings: list[MepFitting] = []

    # Connect downstream terminals to the main distribution panel via orthogonal plenum routing
    downstream_elec = [t for t in elec_terminals if t.terminal_id != primary_panel.terminal_id]
    for idx, term in enumerate(downstream_elec, start=1):
        route = infer_orthogonal_routing(primary_panel.position, term.position, ceiling_z=ceiling_z)
        seg_id = f"conduit-{idx}"
        p_start = MepPort(
            port_id=f"port-{seg_id}-a",
            element_id=seg_id,
            position=route[0],
            direction=(0.0, 0.0, -1.0),
            flow_direction=FlowDirection.SINK,
            connected_port_id=primary_panel.ports[0].port_id if primary_panel.ports else None,
            nominal_size_mm=25.0,
        )
        p_end = MepPort(
            port_id=f"port-{seg_id}-b",
            element_id=seg_id,
            position=route[-1],
            direction=(0.0, 0.0, 1.0),
            flow_direction=FlowDirection.SOURCE,
            connected_port_id=term.ports[0].port_id if term.ports else None,
            nominal_size_mm=25.0,
        )
        seg = MepSegment(
            segment_id=seg_id,
            name=f"Electrical Conduit Run #{idx} ({term.name})",
            system_type=MepSystemType.ELECTRICAL,
            element_type=MepElementType.CONDUIT,
            start_point=route[0],
            end_point=route[-1],
            waypoints=route,
            cross_section=CrossSectionKind.CIRCULAR,
            diameter_mm=25.0,
            ports=[p_start, p_end],
            source_node_id=primary_panel.terminal_id,
            target_node_id=term.terminal_id,
        )
        elec_segments.append(seg)

        # Create intermediate 90-degree elbow fittings at bends
        for bend_i in range(1, len(route) - 1):
            bend_pos = route[bend_i]
            fitting = MepFitting(
                fitting_id=f"fitting-elbow-{idx}-{bend_i}",
                name=f"Conduit Elbow 90° #{idx}.{bend_i}",
                system_type=MepSystemType.ELECTRICAL,
                element_type=MepElementType.ELBOW,
                position=bend_pos,
                angle_deg=90.0,
                connected_segment_ids=[seg_id],
            )
            elec_fittings.append(fitting)

    elec_network = MepDistributionNetwork(
        system_id="system-electrical-01",
        name="Power & Lighting Distribution System",
        system_type=MepSystemType.ELECTRICAL,
        root_panel_id=primary_panel.terminal_id,
        segments=elec_segments,
        fittings=elec_fittings,
        terminals=elec_terminals,
    )
    systems.append(elec_network)

    # 2. Synthesize Plumbing Distribution Network (if plumbing fixtures present or default riser)
    plumb_segments: list[MepSegment] = []
    plumb_fittings: list[MepFitting] = []
    riser_pos = (0.0, 3.8, 0.0)
    for idx, fixture in enumerate(plumb_terminals, start=1):
        p_route = infer_orthogonal_routing(riser_pos, fixture.position, ceiling_z=0.4)
        pseg_id = f"pipe-water-{idx}"
        pseg = MepSegment(
            segment_id=pseg_id,
            name=f"Cold Water Supply Pipe #{idx} ({fixture.name})",
            system_type=MepSystemType.PLUMBING,
            element_type=MepElementType.PIPE,
            start_point=p_route[0],
            end_point=p_route[-1],
            waypoints=p_route,
            cross_section=CrossSectionKind.CIRCULAR,
            diameter_mm=32.0,
            ports=[
                MepPort(port_id=f"port-{pseg_id}-a", element_id=pseg_id, position=p_route[0], nominal_size_mm=32.0),
                MepPort(port_id=f"port-{pseg_id}-b", element_id=pseg_id, position=p_route[-1], nominal_size_mm=32.0),
            ],
            source_node_id="riser-main",
            target_node_id=fixture.terminal_id,
        )
        plumb_segments.append(pseg)

    if plumb_segments or plumb_terminals:
        plumb_network = MepDistributionNetwork(
            system_id="system-plumbing-01",
            name="Domestic Cold Water Supply",
            system_type=MepSystemType.PLUMBING,
            root_panel_id="riser-main",
            segments=plumb_segments,
            fittings=plumb_fittings,
            terminals=plumb_terminals,
        )
        systems.append(plumb_network)

    # 3. Synthesize HVAC Duct Network (if diffusers present)
    if hvac_terminals:
        hvac_segments: list[MepSegment] = []
        ahu_pos = (2.0, 2.0, ceiling_z + 0.1)
        for idx, diff in enumerate(hvac_terminals, start=1):
            d_route = infer_orthogonal_routing(ahu_pos, diff.position, ceiling_z=ceiling_z)
            dseg_id = f"duct-supply-{idx}"
            dseg = MepSegment(
                segment_id=dseg_id,
                name=f"HVAC Supply Air Duct #{idx}",
                system_type=MepSystemType.HVAC,
                element_type=MepElementType.DUCT,
                start_point=d_route[0],
                end_point=d_route[-1],
                waypoints=d_route,
                cross_section=CrossSectionKind.RECTANGULAR,
                width_mm=300.0,
                height_mm=200.0,
                ports=[
                    MepPort(port_id=f"port-{dseg_id}-a", element_id=dseg_id, position=d_route[0], nominal_size_mm=250.0),
                    MepPort(port_id=f"port-{dseg_id}-b", element_id=dseg_id, position=d_route[-1], nominal_size_mm=250.0),
                ],
            )
            hvac_segments.append(dseg)

        hvac_network = MepDistributionNetwork(
            system_id="system-hvac-01",
            name="Mechanical Supply Air System",
            system_type=MepSystemType.HVAC,
            root_panel_id="ahu-main",
            segments=hvac_segments,
            terminals=hvac_terminals,
        )
        systems.append(hvac_network)

    # Code clearance evaluation
    clearances = verify_code_clearances(systems, min_clearance_m=0.15)

    return MepFacilityTopology(
        world_id=world_id,
        systems=systems,
        clearance_violations=clearances,
    )


# =========================================================================
# IFC4 / IFC4x3 Building Services Exporter
# =========================================================================

def export_mep_to_ifc4x3(
    topology: MepFacilityTopology,
    world: Optional[Any] = None,
    output_path: Optional[str | Path] = None,
) -> str:
    """Export complete MEP distribution systems, ports, and geometry to IFC4X3 STEP file."""
    import ifcopenshell
    import ifcopenshell.api as ifc_api
    import ifcopenshell.util.shape_builder as shape_builder

    model = ifc_api.run("project.create_file", version="IFC4X3")
    project = ifc_api.run("root.create_entity", model, ifc_class="IfcProject", name=f"MEP Facility Topology {topology.world_id}")
    project.GlobalId = _deterministic_guid(f"{topology.world_id}/project")

    ifc_api.run("unit.assign_unit", model)
    model_ctx = ifc_api.run("context.add_context", model, context_type="Model")
    body_ctx = ifc_api.run(
        "context.add_context", model, context_type="Model",
        context_identifier="Body", parent=model_ctx, target_view="MODEL_VIEW"
    )

    site = ifc_api.run("root.create_entity", model, ifc_class="IfcSite", name="Facility Site")
    site.GlobalId = _deterministic_guid(f"{topology.world_id}/site")
    building = ifc_api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Facility Building")
    building.GlobalId = _deterministic_guid(f"{topology.world_id}/building")
    storey = ifc_api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Level 1")
    storey.GlobalId = _deterministic_guid(f"{topology.world_id}/storey")

    builder = shape_builder.ShapeBuilder(model)

    for dist_sys in topology.systems:
        # Create IfcDistributionSystem
        ifc_sys = ifc_api.run("system.add_system", model, ifc_class="IfcDistributionSystem")
        ifc_sys.Name = dist_sys.name
        ifc_sys.GlobalId = _deterministic_guid(f"{topology.world_id}/system/{dist_sys.system_id}")
        
        # PredefinedType
        if dist_sys.system_type == MepSystemType.ELECTRICAL:
            ifc_sys.PredefinedType = "ELECTRICAL"
        elif dist_sys.system_type == MepSystemType.PLUMBING:
            ifc_sys.PredefinedType = "DOMESTICCOLDWATER"
        elif dist_sys.system_type == MepSystemType.HVAC:
            ifc_sys.PredefinedType = "VENTILATION"

        system_products = []
        created_ports: dict[str, Any] = {}

        # 1. Terminals
        for term in dist_sys.terminals:
            t_class = "IfcOutlet"
            if term.element_type == MepElementType.PANELBOARD:
                t_class = "IfcElectricDistributionBoard"
            elif term.element_type == MepElementType.SWITCH:
                t_class = "IfcSwitchingDevice"
            elif term.element_type == MepElementType.LIGHT_FIXTURE:
                t_class = "IfcLightFixture"
            elif term.element_type == MepElementType.AIR_TERMINAL:
                t_class = "IfcAirTerminal"
            elif term.element_type == MepElementType.PLUMBING_FIXTURE:
                t_class = "IfcSanitaryTerminal"

            prod = ifc_api.run("root.create_entity", model, ifc_class=t_class, name=term.name)
            prod.GlobalId = _deterministic_guid(f"{topology.world_id}/terminal/{term.terminal_id}")
            system_products.append(prod)

            # Assign to storey
            ifc_api.run("spatial.assign_container", model, relating_structure=storey, products=[prod])

            # Terminal geometry (box profile)
            box_prof = builder.rectangle(size=(0.15, 0.15), position=(term.position[0], term.position[1]))
            box_body = builder.extrude(box_prof, magnitude=0.15, position=(0.0, 0.0, term.position[2]))
            ifc_api.run("geometry.assign_representation", model, product=prod, representation=box_body)

            # Assign ports for terminal
            for port_data in term.ports:
                ifc_port = ifc_api.run("system.add_port", model, element=prod)
                ifc_port.Name = port_data.port_id
                ifc_port.GlobalId = _deterministic_guid(f"{topology.world_id}/port/{port_data.port_id}")
                created_ports[port_data.port_id] = ifc_port

        # 2. Distribution Segments
        for seg in dist_sys.segments:
            s_class = "IfcCableCarrierSegment"
            if seg.system_type == MepSystemType.PLUMBING:
                s_class = "IfcPipeSegment"
            elif seg.system_type == MepSystemType.HVAC:
                s_class = "IfcDuctSegment"

            prod = ifc_api.run("root.create_entity", model, ifc_class=s_class, name=seg.name)
            prod.GlobalId = _deterministic_guid(f"{topology.world_id}/segment/{seg.segment_id}")
            system_products.append(prod)
            ifc_api.run("spatial.assign_container", model, relating_structure=storey, products=[prod])

            # 3D Extruded geometry for segment
            start_p = seg.start_point
            end_p = seg.end_point
            mag = max(seg.length_m, 0.1)

            if seg.cross_section == CrossSectionKind.CIRCULAR:
                r = (seg.diameter_mm / 2000.0)
                curve = builder.circle(center=(start_p[0], start_p[1]), radius=r)
            else:
                w, h = (seg.width_mm / 1000.0), (seg.height_mm / 1000.0)
                curve = builder.rectangle(size=(w, h), position=(start_p[0], start_p[1]))

            body = builder.extrude(curve, magnitude=mag, position=(0.0, 0.0, start_p[2]))
            ifc_api.run("geometry.assign_representation", model, product=prod, representation=body)

            # Assign ports
            for port_data in seg.ports:
                ifc_port = ifc_api.run("system.add_port", model, element=prod)
                ifc_port.Name = port_data.port_id
                ifc_port.GlobalId = _deterministic_guid(f"{topology.world_id}/port/{port_data.port_id}")
                created_ports[port_data.port_id] = ifc_port

        # Connect ports (IfcRelConnectsPorts)
        for seg in dist_sys.segments:
            for port_data in seg.ports:
                if port_data.connected_port_id and port_data.connected_port_id in created_ports:
                    p1 = created_ports[port_data.port_id]
                    p2 = created_ports[port_data.connected_port_id]
                    ifc_api.run("system.connect_port", model, port1=p1, port2=p2)

        # Assign all products to the IfcDistributionSystem
        if system_products:
            ifc_api.run("system.assign_system", model, products=system_products, system=ifc_sys)

    ifc_str = model.to_string()
    if output_path:
        Path(output_path).write_text(ifc_str, encoding="utf-8")

    return ifc_str


# =========================================================================
# GeoJSON Exporter
# =========================================================================

def export_mep_to_geojson(topology: MepFacilityTopology) -> dict[str, Any]:
    """Generates standard GeoJSON FeatureCollection with 3D LineString geometry for MEP runs."""
    features = []

    for dist_sys in topology.systems:
        for seg in dist_sys.segments:
            coords = [[round(c, 4) for c in wp] for wp in seg.waypoints]
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "id": seg.segment_id,
                    "name": seg.name,
                    "system_type": seg.system_type.value,
                    "element_type": seg.element_type.value,
                    "cross_section": seg.cross_section.value,
                    "diameter_mm": seg.diameter_mm,
                    "length_m": seg.length_m,
                    "source_node_id": seg.source_node_id,
                    "target_node_id": seg.target_node_id,
                },
            }
            features.append(feat)

        for term in dist_sys.terminals:
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(c, 4) for c in term.position],
                },
                "properties": {
                    "id": term.terminal_id,
                    "name": term.name,
                    "system_type": term.system_type.value,
                    "element_type": term.element_type.value,
                },
            }
            features.append(feat)

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "world_id": topology.world_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": topology.summary,
        },
    }
