"""Team D — Room Arrangement Engine.

Analyzes the spatial arrangement of furniture within a room from
multi-image indoor perception data.

Capabilities:
  - Support-surface graph construction (what rests on what)
  - Occlusion-order reasoning between overlapping detections
  - Functional zone clustering (seating area, work area, sleeping area)
  - Object permanence tracking across frames
  - Spatial relationship typing (adjacent, on, in, below, etc.)
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("reality.furniture.room_arrangement_engine")


# ---------------------------------------------------------------------------
# Spatial relationship types
# ---------------------------------------------------------------------------
class SpatialRelationType(str):
    ON       = "on"           # object A rests on surface B
    IN       = "in"           # object A is inside container B
    BELOW    = "below"        # A is below B in image
    ABOVE    = "above"        # A is above B in image
    ADJACENT = "adjacent"     # A and B are side by side
    OCCLUDES = "occludes"     # A partially hides B from camera
    SUPPORTS = "supports"     # surface A supports object B


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FurnitureObservationRecord:
    """Minimal record of a furniture observation for arrangement analysis."""
    observation_id: str
    image_id: str
    label: str
    bbox_px: tuple[float, float, float, float]
    confidence: float
    domain: str = "furniture"   # "furniture" | "architecture"

    @property
    def area_px(self) -> float:
        x1, y1, x2, y2 = self.bbox_px
        return max(0.0, x2-x1) * max(0.0, y2-y1)

    @property
    def center_px(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_px
        return ((x1+x2)/2, (y1+y2)/2)

    @property
    def bottom_y(self) -> float:
        return self.bbox_px[3]

    @property
    def top_y(self) -> float:
        return self.bbox_px[1]


@dataclass(frozen=True)
class SpatialRelationship:
    """A typed spatial relationship between two observations."""
    relationship_id: str
    subject_id: str       # the "subject" observation
    object_id: str        # the "object" observation
    relation: str         # SpatialRelationType
    image_id: str
    confidence: float
    evidence: str         # human-readable explanation


@dataclass
class SupportSurfaceGraph:
    """Graph of support surfaces and the objects resting on them."""
    image_id: str
    nodes: dict[str, FurnitureObservationRecord]  # obs_id → record
    relationships: list[SpatialRelationship]
    surface_node_ids: list[str]   # nodes that are surfaces (table, floor, shelf)
    object_node_ids: list[str]    # nodes that are objects resting on surfaces


@dataclass(frozen=True)
class FunctionalZone:
    """A cluster of spatially co-located furniture forming a functional area."""
    zone_id: str
    image_id: str
    zone_type: str         # "seating" | "work" | "sleeping" | "dining" | "storage" | "mixed"
    observation_ids: tuple[str, ...]
    centroid_px: tuple[float, float]
    confidence: float
    dominant_labels: tuple[str, ...]


@dataclass(frozen=True)
class OcclusionOrderDecision:
    """Ordering decision for two spatially overlapping detections."""
    decision_id: str
    front_observation_id: str    # closer to camera
    back_observation_id: str     # occluded by front
    image_id: str
    overlap_iou: float
    reasoning: str


# ---------------------------------------------------------------------------
# Support surface detection
# ---------------------------------------------------------------------------
_SURFACE_LABELS_FURN: frozenset[str] = frozenset({
    "table", "desk", "coffee table", "dining table", "side table",
    "nightstand", "shelf", "bookshelf", "shelving unit",
    "ottoman", "bed", "floor",
})

_CONTAINER_LABELS: frozenset[str] = frozenset({
    "cabinet", "wardrobe", "dresser", "drawer", "box", "basket",
    "laundry basket", "trash can", "bin", "shelf",
})


def is_support_surface(label: str) -> bool:
    return label.lower() in _SURFACE_LABELS_FURN


def is_container(label: str) -> bool:
    return label.lower() in _CONTAINER_LABELS


def _box_overlap_fraction(a: tuple, b: tuple) -> float:
    """Fraction of box A that overlaps with box B."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    area_a = max(1.0, (a[2]-a[0])*(a[3]-a[1]))
    return inter / area_a


def _iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    if inter == 0.0: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / max(aa + ab - inter, 1e-6)


def build_support_surface_graph(
    observations: list[FurnitureObservationRecord],
    image_id: str,
    *,
    proximity_threshold_px: float = 30.0,
    min_overlap_for_on: float = 0.25,
) -> SupportSurfaceGraph:
    """Build a support-surface graph from furniture observations.

    Heuristic: an object A rests "on" surface B if A's bounding box
    horizontally overlaps B's by at least ``min_overlap_for_on`` and
    A's bottom edge is within ``proximity_threshold_px`` of B's top edge.

    Args:
        observations: Furniture and surface observations for one image.
        image_id: Image identifier.
        proximity_threshold_px: Vertical proximity for ON relationship.
        min_overlap_for_on: Minimum horizontal overlap fraction.

    Returns:
        ``SupportSurfaceGraph`` with typed relationships.
    """
    nodes = {obs.observation_id: obs for obs in observations}
    relationships: list[SpatialRelationship] = []
    surface_ids: list[str] = [
        obs.observation_id for obs in observations if is_support_surface(obs.label)
    ]
    object_ids: list[str] = [
        obs.observation_id for obs in observations if not is_support_surface(obs.label)
    ]

    for obj_id in object_ids:
        obj = nodes[obj_id]
        for surf_id in surface_ids:
            surf = nodes[surf_id]
            if obj_id == surf_id:
                continue
            # Vertical proximity check
            vert_gap = abs(obj.bottom_y - surf.top_y)
            if vert_gap > proximity_threshold_px:
                continue
            # Horizontal overlap check
            ox1, _, ox2, _ = obj.bbox_px
            sx1, _, sx2, _ = surf.bbox_px
            overlap_len = max(0.0, min(ox2, sx2) - max(ox1, sx1))
            obj_width = max(1.0, ox2 - ox1)
            if overlap_len / obj_width < min_overlap_for_on:
                continue
            conf = min(0.90, (1.0 - vert_gap / proximity_threshold_px) * 0.9)
            rid = hashlib.sha1(f"{obj_id}:on:{surf_id}".encode()).hexdigest()[:8]
            relationships.append(SpatialRelationship(
                relationship_id=f"rel_{rid}",
                subject_id=obj_id,
                object_id=surf_id,
                relation=SpatialRelationType.ON,
                image_id=image_id,
                confidence=conf,
                evidence=(
                    f"{obj.label} bottom_y={obj.bottom_y:.0f} near "
                    f"{surf.label} top_y={surf.top_y:.0f} (gap={vert_gap:.0f}px)"
                ),
            ))

    return SupportSurfaceGraph(
        image_id=image_id,
        nodes=nodes,
        relationships=relationships,
        surface_node_ids=surface_ids,
        object_node_ids=object_ids,
    )


# ---------------------------------------------------------------------------
# Occlusion-order reasoning
# ---------------------------------------------------------------------------
def infer_occlusion_order(
    observations: list[FurnitureObservationRecord],
    image_id: str,
    *,
    min_iou_for_occlusion: float = 0.10,
) -> list[OcclusionOrderDecision]:
    """Determine which of two overlapping objects is in front of the other.

    Uses bottom-edge-in-image heuristic: in indoor perspective views,
    objects lower in the image are generally closer to the camera.

    Args:
        observations: All furniture observations for the image.
        image_id: Image identifier.
        min_iou_for_occlusion: Minimum IoU to consider for occlusion analysis.

    Returns:
        List of ``OcclusionOrderDecision`` for overlapping pairs.
    """
    decisions: list[OcclusionOrderDecision] = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            a = observations[i]
            b = observations[j]
            iou = _iou(a.bbox_px, b.bbox_px)
            if iou < min_iou_for_occlusion:
                continue

            # Lower bottom-y = closer to bottom of image = typically closer to camera
            if a.bottom_y > b.bottom_y:
                front, back = a, b
            elif b.bottom_y > a.bottom_y:
                front, back = b, a
            else:
                # Same bottom — use larger object as background
                front, back = (b, a) if a.area_px > b.area_px else (a, b)

            did = hashlib.sha1(f"{front.observation_id}:{back.observation_id}".encode()).hexdigest()[:8]
            decisions.append(OcclusionOrderDecision(
                decision_id=f"occ_{did}",
                front_observation_id=front.observation_id,
                back_observation_id=back.observation_id,
                image_id=image_id,
                overlap_iou=iou,
                reasoning=(
                    f"{front.label} (bottom_y={front.bottom_y:.0f}) occludes "
                    f"{back.label} (bottom_y={back.bottom_y:.0f}), IoU={iou:.2f}"
                ),
            ))
    return decisions


# ---------------------------------------------------------------------------
# Functional zone clustering
# ---------------------------------------------------------------------------
_ZONE_SIGNATURES: dict[str, frozenset[str]] = {
    "seating":  frozenset(["chair", "sofa", "armchair", "coffee table", "ottoman"]),
    "work":     frozenset(["desk", "office chair", "monitor", "computer", "keyboard", "printer"]),
    "sleeping": frozenset(["bed", "nightstand", "wardrobe", "lamp", "dresser"]),
    "dining":   frozenset(["dining table", "dining chair", "table", "chair"]),
    "storage":  frozenset(["cabinet", "wardrobe", "bookshelf", "shelving", "dresser"]),
}


def cluster_into_functional_zones(
    observations: list[FurnitureObservationRecord],
    image_id: str,
    *,
    proximity_threshold_px: float = 150.0,
) -> list[FunctionalZone]:
    """Cluster spatially proximate furniture into functional zones.

    Args:
        observations: Furniture observations for one image.
        image_id: Image identifier.
        proximity_threshold_px: Max center-to-center distance to cluster.

    Returns:
        List of ``FunctionalZone`` objects.
    """
    if not observations:
        return []

    # Cluster by spatial proximity
    clusters: list[list[FurnitureObservationRecord]] = []
    assigned = [False] * len(observations)

    for i, obs_i in enumerate(observations):
        if assigned[i]:
            continue
        cluster = [obs_i]
        assigned[i] = True
        for j, obs_j in enumerate(observations):
            if assigned[j] or j == i:
                continue
            cx_i, cy_i = obs_i.center_px
            cx_j, cy_j = obs_j.center_px
            dist = math.sqrt((cx_i-cx_j)**2 + (cy_i-cy_j)**2)
            if dist <= proximity_threshold_px:
                cluster.append(obs_j)
                assigned[j] = True
        clusters.append(cluster)

    zones: list[FunctionalZone] = []
    for idx, cluster in enumerate(clusters):
        label_set = frozenset(obs.label.lower() for obs in cluster)

        # Match against zone signatures
        best_zone = "mixed"
        best_overlap = 0
        for zone_name, signature in _ZONE_SIGNATURES.items():
            overlap = len(label_set & signature)
            if overlap > best_overlap:
                best_overlap = overlap
                best_zone = zone_name

        confidence = min(0.90, 0.3 + 0.15 * best_overlap + 0.05 * len(cluster))
        cx = sum(obs.center_px[0] for obs in cluster) / len(cluster)
        cy = sum(obs.center_px[1] for obs in cluster) / len(cluster)
        label_counts: dict[str, int] = defaultdict(int)
        for obs in cluster:
            label_counts[obs.label] += 1
        dominant = sorted(label_counts, key=label_counts.__getitem__, reverse=True)[:3]

        zid = hashlib.sha1(f"{image_id}:zone:{idx}".encode()).hexdigest()[:8]
        zones.append(FunctionalZone(
            zone_id=f"zone_{zid}",
            image_id=image_id,
            zone_type=best_zone,
            observation_ids=tuple(obs.observation_id for obs in cluster),
            centroid_px=(cx, cy),
            confidence=confidence,
            dominant_labels=tuple(dominant),
        ))

    return zones
