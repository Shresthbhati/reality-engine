"""Multi-Epoch 4D As-Built Temporal Delta & Automated Site Change Detection.

Compares two capture epochs/versions of a construction site or facility to detect:
- Added elements (newly installed construction elements, MEP, fixtures)
- Removed elements (demolished walls, removed temporary scaffolding/equipment)
- Moved elements (displaced beyond tolerance with exact 3D displacement vector)
- Renovated elements (modified dimensions, altered openings, upgraded finishes)
- Unchanged static structural anchors for metric registration validation
- Spatial point-cloud voxel occupancy delta for volumetric change heatmaps
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class TemporalChangeKind(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    RENOVATED = "renovated"
    DAMAGED = "damaged"
    UNCHANGED = "unchanged"


class TradeCategory(str, Enum):
    STRUCTURE = "structure"
    ARCHITECTURE = "architecture"
    MEP = "mep"
    FINISHES = "finishes"
    EQUIPMENT = "equipment"
    OTHER = "other"


@dataclass(slots=True)
class TemporalEntityDelta:
    """Detailed temporal delta for a single entity between two epochs."""

    entity_id: string = ""
    name: str = ""
    entity_type: str = ""
    trade: TradeCategory = TradeCategory.OTHER
    change_kind: TemporalChangeKind = TemporalChangeKind.UNCHANGED
    displacement_m: float = 0.0
    displacement_vector: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    volume_from_m3: float = 0.0
    volume_to_m3: float = 0.0
    volume_delta_m3: float = 0.0
    bounds_from: dict[str, list[float]] = field(default_factory=dict)
    bounds_to: dict[str, list[float]] = field(default_factory=dict)
    confidence: float = 1.0
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "trade": self.trade.value if isinstance(self.trade, TradeCategory) else str(self.trade),
            "change_kind": self.change_kind.value if isinstance(self.change_kind, TemporalChangeKind) else str(self.change_kind),
            "displacement_m": round(self.displacement_m, 4),
            "displacement_vector": [round(v, 4) for v in self.displacement_vector],
            "volume_from_m3": round(self.volume_from_m3, 4),
            "volume_to_m3": round(self.volume_to_m3, 4),
            "volume_delta_m3": round(self.volume_delta_m3, 4),
            "bounds_from": self.bounds_from,
            "bounds_to": self.bounds_to,
            "confidence": round(self.confidence, 3),
            "details": self.details,
        }


@dataclass(slots=True)
class SpatialChangeCluster:
    """Spatial voxel cluster representing volumetric added or removed material."""

    cluster_id: str
    change_kind: TemporalChangeKind  # ADDED or REMOVED
    centroid: list[float]
    bounds_min: list[float]
    bounds_max: list[float]
    voxel_count: int
    volume_m3: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "change_kind": self.change_kind.value,
            "centroid": [round(c, 3) for c in self.centroid],
            "bounds_min": [round(c, 3) for c in self.bounds_min],
            "bounds_max": [round(c, 3) for c in self.bounds_max],
            "voxel_count": self.voxel_count,
            "volume_m3": round(self.volume_m3, 4),
        }


@dataclass(slots=True)
class TemporalSiteDeltaSummary:
    """Complete multi-epoch temporal comparison between two site captures."""

    world_id: str
    from_epoch_id: str
    to_epoch_id: str
    from_timestamp_iso: Optional[str]
    to_timestamp_iso: Optional[str]
    is_aligned: bool
    static_anchor_count: int
    residual_alignment_error_m: float
    total_added: int
    total_removed: int
    total_moved: int
    total_renovated: int
    total_unchanged: int
    net_volume_change_m3: float
    trade_counts: dict[str, dict[str, int]]
    changes: list[TemporalEntityDelta] = field(default_factory=list)
    spatial_clusters: list[SpatialChangeCluster] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "from_epoch_id": self.from_epoch_id,
            "to_epoch_id": self.to_epoch_id,
            "from_timestamp_iso": self.from_timestamp_iso,
            "to_timestamp_iso": self.to_timestamp_iso,
            "is_aligned": self.is_aligned,
            "static_anchor_count": self.static_anchor_count,
            "residual_alignment_error_m": round(self.residual_alignment_error_m, 4),
            "counts": {
                "added": self.total_added,
                "removed": self.total_removed,
                "moved": self.total_moved,
                "renovated": self.total_renovated,
                "unchanged": self.total_unchanged,
                "total_analyzed": len(self.changes),
            },
            "net_volume_change_m3": round(self.net_volume_change_m3, 4),
            "trade_breakdown": self.trade_counts,
            "changes": [c.to_dict() for c in self.changes],
            "spatial_clusters": [cl.to_dict() for cl in self.spatial_clusters],
        }


def classify_trade(entity_type: str, custom_properties: Optional[dict] = None) -> TradeCategory:
    """Classify an entity into an architectural/construction trade category."""
    props = custom_properties or {}
    if "trade" in props:
        t_str = str(props["trade"]).lower()
        for cat in TradeCategory:
            if cat.value in t_str:
                return cat

    t_upper = str(entity_type).upper()
    if any(k in t_upper for k in ["WALL", "COLUMN", "BEAM", "SLAB", "FOUNDATION", "STAIR", "FOOTING"]):
        return TradeCategory.STRUCTURE
    if any(k in t_upper for k in ["DOOR", "WINDOW", "ROOM", "OPENING", "CEILING"]):
        return TradeCategory.ARCHITECTURE
    if any(k in t_upper for k in ["ELECTRICAL", "OUTLET", "SWITCH", "FIXTURE", "PIPE", "DUCT", "CONDUIT", "HVAC", "PLUMBING"]):
        return TradeCategory.MEP
    if any(k in t_upper for k in ["FINISH", "FLOORING", "TILE", "PAINT", "CLADDING"]):
        return TradeCategory.FINISHES
    if any(k in t_upper for k in ["FURNITURE", "EQUIPMENT", "APPLIANCE", "PROP"]):
        return TradeCategory.EQUIPMENT
    return TradeCategory.OTHER


def _extract_position(entity: Any) -> tuple[float, float, float] | None:
    """Extract (x, y, z) position tuple from entity transform or properties."""
    tr = getattr(entity, "transform", None)
    if isinstance(tr, dict) and "position" in tr:
        pos = tr["position"]
        if isinstance(pos, dict) and all(k in pos for k in ["x", "y", "z"]):
            return (float(pos["x"]), float(pos["y"]), float(pos["z"]))
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return (float(pos[0]), float(pos[1]), float(pos[2]))

    props = getattr(entity, "custom_properties", None) or {}
    if "position" in props and isinstance(props["position"], (list, tuple)) and len(props["position"]) >= 3:
        p = props["position"]
        return (float(p[0]), float(p[1]), float(p[2]))

    return None


def _extract_dimensions(entity: Any) -> tuple[float, float, float]:
    """Extract bounding dimensions (width, depth, height) in meters."""
    props = getattr(entity, "custom_properties", None) or {}
    w = float(props.get("width_m", props.get("width", 1.0)))
    d = float(props.get("depth_m", props.get("depth", 1.0)))
    h = float(props.get("height_m", props.get("height", 1.0)))
    return (max(0.01, w), max(0.01, d), max(0.01, h))


def _calculate_volume(entity: Any) -> float:
    """Estimate volume of entity in cubic meters."""
    props = getattr(entity, "custom_properties", None) or {}
    if "volume_m3" in props:
        return float(props["volume_m3"])
    w, d, h = _extract_dimensions(entity)
    return w * d * h


def _extract_bounds(entity: Any) -> dict[str, list[float]]:
    """Compute axis-aligned bounding box for entity."""
    pos = _extract_position(entity) or (0.0, 0.0, 0.0)
    w, d, h = _extract_dimensions(entity)
    hw, hd, hh = w / 2.0, d / 2.0, h / 2.0
    return {
        "min": [round(pos[0] - hw, 3), round(pos[1] - hd, 3), round(pos[2] - hh, 3)],
        "max": [round(pos[0] + hw, 3), round(pos[1] + hd, 3), round(pos[2] + hh, 3)],
    }


def compute_temporal_delta(
    world_from: Any,
    world_to: Any,
    from_epoch_id: str = "epoch_1",
    to_epoch_id: str = "epoch_2",
    movement_tolerance_m: float = 0.03,
    from_timestamp_iso: Optional[str] = None,
    to_timestamp_iso: Optional[str] = None,
) -> TemporalSiteDeltaSummary:
    """Compute comprehensive multi-epoch 4D temporal delta between two WorldIR states."""
    world_id = getattr(world_to, "id", getattr(world_from, "id", "world_default"))

    ents_from: dict[str, Any] = getattr(world_from, "entities", {}) or {}
    ents_to: dict[str, Any] = getattr(world_to, "entities", {}) or {}

    changes: list[TemporalEntityDelta] = []
    trade_counts: dict[str, dict[str, int]] = {
        t.value: {"added": 0, "removed": 0, "moved": 0, "renovated": 0, "unchanged": 0}
        for t in TradeCategory
    }

    matched_to_ids: set[str] = set()
    static_residuals: list[float] = []

    # 1. Match entities existing in from_world
    for ent_id, e_from in ents_from.items():
        name = getattr(e_from, "name", ent_id)
        etype = str(getattr(e_from, "type", "UNKNOWN"))
        trade = classify_trade(etype, getattr(e_from, "custom_properties", None))
        pos_from = _extract_position(e_from)
        vol_from = _calculate_volume(e_from)
        b_from = _extract_bounds(e_from)

        if ent_id in ents_to:
            matched_to_ids.add(ent_id)
            e_to = ents_to[ent_id]
            pos_to = _extract_position(e_to)
            vol_to = _calculate_volume(e_to)
            b_to = _extract_bounds(e_to)
            vol_delta = vol_to - vol_from

            # Calculate displacement
            disp_m = 0.0
            disp_vec = [0.0, 0.0, 0.0]
            if pos_from and pos_to:
                dx = pos_to[0] - pos_from[0]
                dy = pos_to[1] - pos_from[1]
                dz = pos_to[2] - pos_from[2]
                disp_m = math.hypot(dx, dy, dz)
                disp_vec = [dx, dy, dz]

            # Determine change category
            if disp_m > movement_tolerance_m:
                change_kind = TemporalChangeKind.MOVED
                details = f"Displaced {disp_m * 100:.1f} cm (dx={disp_vec[0]:.2f}m, dy={disp_vec[1]:.2f}m, dz={disp_vec[2]:.2f}m)"
            elif abs(vol_delta) > max(0.02, 0.08 * vol_from):
                change_kind = TemporalChangeKind.RENOVATED
                pct = (vol_delta / (vol_from or 1.0)) * 100.0
                details = f"Volume altered by {vol_delta:+.3f} m³ ({pct:+.1f}%)"
            else:
                change_kind = TemporalChangeKind.UNCHANGED
                details = f"Static anchor within {disp_m * 100:.1f} cm tolerance"
                static_residuals.append(disp_m)

            delta = TemporalEntityDelta(
                entity_id=ent_id,
                name=name,
                entity_type=etype,
                trade=trade,
                change_kind=change_kind,
                displacement_m=disp_m,
                displacement_vector=disp_vec,
                volume_from_m3=vol_from,
                volume_to_m3=vol_to,
                volume_delta_m3=vol_delta,
                bounds_from=b_from,
                bounds_to=b_to,
                confidence=float(getattr(e_to, "confidence", 1.0) or 1.0),
                details=details,
            )
            changes.append(delta)
            trade_counts[trade.value][change_kind.value] += 1

        else:
            # Entity was removed in to_world
            delta = TemporalEntityDelta(
                entity_id=ent_id,
                name=name,
                entity_type=etype,
                trade=trade,
                change_kind=TemporalChangeKind.REMOVED,
                displacement_m=0.0,
                displacement_vector=[0.0, 0.0, 0.0],
                volume_from_m3=vol_from,
                volume_to_m3=0.0,
                volume_delta_m3=-vol_from,
                bounds_from=b_from,
                bounds_to={},
                confidence=1.0,
                details="Element decommissioned or removed from site",
            )
            changes.append(delta)
            trade_counts[trade.value]["removed"] += 1

    # 2. Find newly added entities in to_world
    for ent_id, e_to in ents_to.items():
        if ent_id in matched_to_ids:
            continue

        name = getattr(e_to, "name", ent_id)
        etype = str(getattr(e_to, "type", "UNKNOWN"))
        trade = classify_trade(etype, getattr(e_to, "custom_properties", None))
        vol_to = _calculate_volume(e_to)
        b_to = _extract_bounds(e_to)

        delta = TemporalEntityDelta(
            entity_id=ent_id,
            name=name,
            entity_type=etype,
            trade=trade,
            change_kind=TemporalChangeKind.ADDED,
            displacement_m=0.0,
            displacement_vector=[0.0, 0.0, 0.0],
            volume_from_m3=0.0,
            volume_to_m3=vol_to,
            volume_delta_m3=vol_to,
            bounds_from={},
            bounds_to=b_to,
            confidence=float(getattr(e_to, "confidence", 1.0) or 1.0),
            details="Newly constructed or installed element",
        )
        changes.append(delta)
        trade_counts[trade.value]["added"] += 1

    # Aggregate metrics
    n_added = sum(1 for c in changes if c.change_kind == TemporalChangeKind.ADDED)
    n_removed = sum(1 for c in changes if c.change_kind == TemporalChangeKind.REMOVED)
    n_moved = sum(1 for c in changes if c.change_kind == TemporalChangeKind.MOVED)
    n_renovated = sum(1 for c in changes if c.change_kind == TemporalChangeKind.RENOVATED)
    n_unchanged = sum(1 for c in changes if c.change_kind == TemporalChangeKind.UNCHANGED)

    net_vol = sum(c.volume_delta_m3 for c in changes)
    mean_residual = float(np.mean(static_residuals)) if static_residuals else 0.0

    return TemporalSiteDeltaSummary(
        world_id=world_id,
        from_epoch_id=from_epoch_id,
        to_epoch_id=to_epoch_id,
        from_timestamp_iso=from_timestamp_iso,
        to_timestamp_iso=to_timestamp_iso,
        is_aligned=n_unchanged >= 2 or len(changes) <= 2,
        static_anchor_count=n_unchanged,
        residual_alignment_error_m=mean_residual,
        total_added=n_added,
        total_removed=n_removed,
        total_moved=n_moved,
        total_renovated=n_renovated,
        total_unchanged=n_unchanged,
        net_volume_change_m3=net_vol,
        trade_counts=trade_counts,
        changes=changes,
        spatial_clusters=[],
    )


def detect_spatial_change_clusters(
    points_from: np.ndarray | Sequence[Sequence[float]],
    points_to: np.ndarray | Sequence[Sequence[float]],
    voxel_size: float = 0.15,
    min_cluster_voxels: int = 5,
) -> list[SpatialChangeCluster]:
    """Detect volumetric point cloud change clusters via 3D spatial voxel grid differencing."""
    p_from = np.asarray(points_from, dtype=np.float32)
    p_to = np.asarray(points_to, dtype=np.float32)

    if len(p_from) == 0 or len(p_to) == 0:
        return []

    # Discretize points into voxel coordinates (ix, iy, iz)
    voxels_from = set(tuple(v) for v in np.floor(p_from / voxel_size).astype(np.int32))
    voxels_to = set(tuple(v) for v in np.floor(p_to / voxel_size).astype(np.int32))

    added_voxels = voxels_to - voxels_from
    removed_voxels = voxels_from - voxels_to

    clusters: list[SpatialChangeCluster] = []
    voxel_vol = voxel_size**3

    def _cluster_voxels(voxels: set[tuple[int, int, int]], kind: TemporalChangeKind) -> list[SpatialChangeCluster]:
        visited: set[tuple[int, int, int]] = set()
        out: list[SpatialChangeCluster] = []
        c_idx = 1

        for v in voxels:
            if v in visited:
                continue
            # BFS 26-neighborhood clustering
            component: list[tuple[int, int, int]] = []
            queue = [v]
            visited.add(v)

            while queue:
                curr = queue.pop(0)
                component.append(curr)
                # Check neighbors
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nb = (curr[0] + dx, curr[1] + dy, curr[2] + dz)
                            if nb in voxels and nb not in visited:
                                visited.add(nb)
                                queue.append(nb)

            if len(component) >= min_cluster_voxels:
                pts = np.array(component, dtype=np.float32) * voxel_size + (voxel_size / 2.0)
                centroid = pts.mean(axis=0).tolist()
                b_min = pts.min(axis=0).tolist()
                b_max = pts.max(axis=0).tolist()
                out.append(
                    SpatialChangeCluster(
                        cluster_id=f"{kind.value}_cluster_{c_idx}",
                        change_kind=kind,
                        centroid=centroid,
                        bounds_min=b_min,
                        bounds_max=b_max,
                        voxel_count=len(component),
                        volume_m3=len(component) * voxel_vol,
                    )
                )
                c_idx += 1
        return out

    clusters.extend(_cluster_voxels(added_voxels, TemporalChangeKind.ADDED))
    clusters.extend(_cluster_voxels(removed_voxels, TemporalChangeKind.REMOVED))

    return clusters
