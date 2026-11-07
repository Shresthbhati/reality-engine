"""Capture-session grouping (Master Development Plan Phase 2, Task 2.7).

Groups photos and evidence into capture sessions based on:
1. Temporal gaps: breaks capture when time between consecutive shots > max_time_gap_s
   (e.g., 30 minutes indicates a different capture session / break).
2. Spatial distance: breaks capture when distance between consecutive GPS locations > max_distance_m
   (e.g., 500 meters indicates a different physical site / building).

Deterministic, raise-free, and handles mixed evidence with missing EXIF/GPS honestly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS coordinates in meters (WGS-84 sphere)."""
    r = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c


@dataclass(frozen=True)
class SessionGroup:
    """One coherent capture session of related evidence items."""

    group_id: str
    items: Tuple[Any, ...]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    center_gps: Optional[Tuple[float, float]] = None

    @property
    def item_count(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "item_count": self.item_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "center_gps": list(self.center_gps) if self.center_gps else None,
            "item_ids": [
                item.get("id") if isinstance(item, dict) else getattr(item, "id", str(item))
                for item in self.items
            ],
        }


def _extract_time_and_gps(item: Any) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Helper to extract timestamp (epoch seconds) and GPS (lat, lon) from various evidence types."""
    if isinstance(item, dict):
        t = item.get("captured_at") or item.get("acquired_at")
        meta = item.get("metadata", {}) or {}
        sensor = item.get("sensor_metadata", {}) or {}
        gps = None
        if "latitude" in item and "longitude" in item and item["latitude"] is not None and item["longitude"] is not None:
            try:
                gps = (float(item["latitude"]), float(item["longitude"]))
            except (ValueError, TypeError):
                pass
        if gps is None and "latitude_deg" in sensor and "longitude_deg" in sensor:
            try:
                gps = (float(sensor["latitude_deg"]), float(sensor["longitude_deg"]))
            except (ValueError, TypeError):
                pass
        if gps is None and "gps" in meta and isinstance(meta["gps"], (tuple, list)) and len(meta["gps"]) >= 2:
            try:
                gps = (float(meta["gps"][0]), float(meta["gps"][1]))
            except (ValueError, TypeError):
                pass
        return t, gps

    # 1. Direct attributes (e.g. EvidenceItem, ImportedAsset)
    t = getattr(item, "captured_at", None)
    if t is None:
        t = getattr(item, "acquired_at", None)

    # 2. Metadata attributes
    meta: Dict[str, Any] = getattr(item, "metadata", {}) or {}
    sensor: Dict[str, Any] = getattr(item, "sensor_metadata", {}) or {}

    if t is None:
        t = meta.get("captured_at") or meta.get("acquired_at")

    gps: Optional[Tuple[float, float]] = None
    # Check sensor_metadata first
    if "latitude_deg" in sensor and "longitude_deg" in sensor:
        try:
            gps = (float(sensor["latitude_deg"]), float(sensor["longitude_deg"]))
        except (ValueError, TypeError):
            pass
    elif "gps" in meta and isinstance(meta["gps"], (tuple, list)) and len(meta["gps"]) >= 2:
        try:
            gps = (float(meta["gps"][0]), float(meta["gps"][1]))
        except (ValueError, TypeError):
            pass

    return t, gps


def group_evidence_by_session(
    items: Sequence[Any],
    max_time_gap_s: float = 1800.0,
    max_distance_m: float = 500.0,
    group_prefix: str = "session",
) -> List[SessionGroup]:
    """Group evidence items into capture sessions.

    Items are sorted by timestamp when available. A new session group begins when:
      - The time gap to the previous item exceeds `max_time_gap_s` (default 30 min)
      - The spatial distance to the previous GPS point exceeds `max_distance_m` (default 500m)

    Items without timestamps or GPS remain together in the default session.
    """
    if not items:
        return []

    # Pair items with their extracted properties
    annotated: List[Tuple[Any, Optional[float], Optional[Tuple[float, float]]]] = []
    for item in items:
        t, gps = _extract_time_and_gps(item)
        annotated.append((item, t, gps))

    # Stable sort by timestamp if any have timestamps
    has_timestamps = any(t is not None for _, t, _ in annotated)
    if has_timestamps:
        annotated.sort(key=lambda x: (x[1] is None, x[1] or 0.0))

    clusters: List[List[Tuple[Any, Optional[float], Optional[Tuple[float, float]]]]] = []
    current_cluster: List[Tuple[Any, Optional[float], Optional[Tuple[float, float]]]] = []

    last_time: Optional[float] = None
    last_gps: Optional[Tuple[float, float]] = None

    for item, t, gps in annotated:
        split = False
        if current_cluster:
            # Check time gap
            if t is not None and last_time is not None:
                if (t - last_time) > max_time_gap_s:
                    split = True
            # Check spatial gap
            if gps is not None and last_gps is not None:
                dist = haversine_distance_m(last_gps[0], last_gps[1], gps[0], gps[1])
                if dist > max_distance_m:
                    split = True

        if split:
            clusters.append(current_cluster)
            current_cluster = []

        current_cluster.append((item, t, gps))
        if t is not None:
            last_time = t
        if gps is not None:
            last_gps = gps

    if current_cluster:
        clusters.append(current_cluster)

    groups: List[SessionGroup] = []
    for idx, cluster in enumerate(clusters, start=1):
        c_items = tuple(x[0] for x in cluster)
        c_times = [x[1] for x in cluster if x[1] is not None]
        c_gps = [x[2] for x in cluster if x[2] is not None]

        start_t = min(c_times) if c_times else None
        end_t = max(c_times) if c_times else None

        center_gps = None
        if c_gps:
            mean_lat = sum(g[0] for g in c_gps) / len(c_gps)
            mean_lon = sum(g[1] for g in c_gps) / len(c_gps)
            center_gps = (round(mean_lat, 6), round(mean_lon, 6))

        groups.append(
            SessionGroup(
                group_id=f"{group_prefix}-{idx:03d}",
                items=c_items,
                start_time=start_t,
                end_time=end_t,
                center_gps=center_gps,
            )
        )

    return groups
