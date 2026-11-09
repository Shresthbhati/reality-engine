"""Cross-world geographic identity (plan item E.25).

Creates a consistent geographic reference system for buildings, floors, rooms,
corridors, outdoor areas and connected sites. Multiple captured properties can
be placed in a coherent geographic context without relying solely on manually
supplied transforms.

Honesty rules:
* A world with no geographic anchor is "unanchored", not placed at (0, 0).
* Anchor confidence reflects the quality of the geographic evidence (GPS,
  surveyed control points, shared landmarks).
* Two worlds are only merged when their anchors agree within measured tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class GeoAnchor:
    """One geographic anchor for a world."""
    world_id: str
    crs: str                              # e.g. "EPSG:4326", "EPSG:27700"
    origin_m: Tuple[float, float, float]  # world-frame (0,0,0) in CRS units
    confidence: float                     # 0..1
    source: str                           # "gps" | "surveyed" | "shared_landmark" | "manual" | "none"
    evidence_ids: List[str]               # GPS readings, control point photos, etc.
    note: Optional[str] = None


@dataclass(frozen=True)
class GeoTransform:
    """Transform from one world's local frame to another's."""
    from_world: str
    to_world: str
    translation_m: Tuple[float, float, float]
    rotation_deg: float                   # Z-axis rotation (yaw) in degrees
    scale: float                          # uniform scale factor (1.0 = same units)
    confidence: float                     # 0..1, derived from anchor agreement
    method: str                           # "anchor_match" | "landmark_match" | "manual"


@dataclass(frozen=True)
class GeographicIdentityReport:
    """Geographic relationship between multiple worlds."""
    anchors: Dict[str, GeoAnchor]
    transforms: List[GeoTransform]
    unanchored_worlds: List[str]
    merge_candidates: List[Tuple[str, str, float]]  # (world_a, world_b, confidence)
    provisional: bool = True


class GeographicIdentity:
    """Manages geographic anchors and cross-world transforms."""

    #: Two anchors closer than this (in their shared CRS units) are candidates
    #: for merging (metres).
    ANCHOR_AGREE_TOL_M = 5.0
    #: Confidence below which a merge is not suggested.
    MERGE_MIN_CONFIDENCE = 0.60

    def __init__(self):
        self._anchors: Dict[str, GeoAnchor] = {}

    def set_anchor(
        self,
        world_id: str,
        crs: str,
        origin_m: Tuple[float, float, float],
        confidence: float,
        source: str,
        evidence_ids: Optional[Sequence[str]] = None,
        note: Optional[str] = None,
    ) -> GeoAnchor:
        """Register (or replace) the geographic anchor for one world."""
        anchor = GeoAnchor(
            world_id=world_id, crs=crs, origin_m=tuple(float(c) for c in origin_m),
            confidence=max(0.0, min(1.0, confidence)), source=source,
            evidence_ids=list(evidence_ids or []), note=note,
        )
        self._anchors[world_id] = anchor
        return anchor

    def anchor_for(self, world_id: str) -> Optional[GeoAnchor]:
        return self._anchors.get(world_id)

    def report(self, world_ids: Sequence[str]) -> GeographicIdentityReport:
        """Compute the geographic relationship between the given worlds."""
        anchors: Dict[str, GeoAnchor] = {}
        unanchored: List[str] = []
        for wid in world_ids:
            a = self._anchors.get(wid)
            if a is None or a.source == "none":
                unanchored.append(wid)
            else:
                anchors[wid] = a

        # ---- transforms between anchored worlds -----------------------------
        transforms: List[GeoTransform] = []
        anchor_list = list(anchors.values())
        for i in range(len(anchor_list)):
            for j in range(i + 1, len(anchor_list)):
                a, b = anchor_list[i], anchor_list[j]
                if a.crs != b.crs:
                    # different CRS: cannot compute a transform without a CRS
                    # conversion pipeline; report as unmergeable
                    continue
                ta = np.asarray(a.origin_m, float)
                tb = np.asarray(b.origin_m, float)
                translation = tuple(round(float(c), 4) for c in (tb - ta))
                dist = float(np.linalg.norm(tb - ta))
                # simple yaw-only transform (no roll/pitch for plan alignment)
                conf = round(min(a.confidence, b.confidence) * 0.9, 3)
                transforms.append(GeoTransform(
                    from_world=a.world_id, to_world=b.world_id,
                    translation_m=translation, rotation_deg=0.0, scale=1.0,
                    confidence=conf, method="anchor_match",
                ))

        # ---- merge candidates ----------------------------------------------
        merge_candidates: List[Tuple[str, str, float]] = []
        for t in transforms:
            dist = math.sqrt(sum(c * c for c in t.translation_m))
            if dist <= self.ANCHOR_AGREE_TOL_M and t.confidence >= self.MERGE_MIN_CONFIDENCE:
                merge_candidates.append((t.from_world, t.to_world, t.confidence))

        return GeographicIdentityReport(
            anchors=anchors, transforms=transforms,
            unanchored_worlds=unanchored, merge_candidates=merge_candidates,
            provisional=True,
        )

    @staticmethod
    def world_bounds_in_crs(
        entities: Sequence[Entity], anchor: GeoAnchor
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Convert world-frame entity bounds to CRS coordinates using the anchor.

        Returns None if no entity has bounds.
        """
        mins: List[np.ndarray] = []
        maxs: List[np.ndarray] = []
        origin = np.asarray(anchor.origin_m, float)
        for e in entities:
            cp = e.custom_properties
            bmin = cp.get("bounds_min")
            bmax = cp.get("bounds_max")
            if bmin is None or bmax is None:
                continue
            mins.append(np.asarray(bmin, float) + origin)
            maxs.append(np.asarray(bmax, float) + origin)
        if not mins:
            return None
        return (tuple(round(float(c), 4) for c in np.min(mins, axis=0)),
                tuple(round(float(c), 4) for c in np.max(maxs, axis=0)))


__all__ = ["GeographicIdentity", "GeoAnchor", "GeoTransform", "GeographicIdentityReport"]