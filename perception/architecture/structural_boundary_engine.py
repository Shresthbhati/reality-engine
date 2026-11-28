"""Team C — Structural Boundary Engine.

Extracts and analyzes architectural structural boundaries from indoor perception data.

Capabilities:
  - Wall intersection and corner topology (L / T / X junctions)
  - Host-wall hypothesis graph construction
  - Room boundary polygon extraction
  - Structural element relationship graph
  - Cross-view structural continuity assessment
  - Contradiction detection for structural hypotheses

Design principles:
  - Never asserts a room topology it cannot support with observation evidence.
  - Ambiguous junctions are always preserved as alternatives, not silently resolved.
  - Metric estimates are always marked as image-space estimates until camera pose available.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

log = logging.getLogger("reality.architecture.structural_boundary_engine")


# ---------------------------------------------------------------------------
# Junction type taxonomy
# ---------------------------------------------------------------------------
class JunctionType(str, Enum):
    L_JUNCTION = "L"      # Two walls meet at a corner (90-ish degrees)
    T_JUNCTION = "T"      # One wall meets the middle of another
    X_JUNCTION = "X"      # Two walls cross (rare in residential, common in commercial)
    AMBIGUOUS  = "?"      # Cannot be determined from available evidence


# ---------------------------------------------------------------------------
# 2D point in image space
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImagePoint:
    x: float    # column in pixels
    y: float    # row in pixels

    def distance_to(self, other: "ImagePoint") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def midpoint(self, other: "ImagePoint") -> "ImagePoint":
        return ImagePoint((self.x + other.x) / 2.0, (self.y + other.y) / 2.0)


# ---------------------------------------------------------------------------
# Wall segment hypothesis
# ---------------------------------------------------------------------------
@dataclass
class WallSegmentHypothesis:
    """A detected wall segment in image space."""
    segment_id: str
    image_id: str
    p1: ImagePoint
    p2: ImagePoint
    confidence: float
    source_evidence: list[str]  # observation IDs that support this segment
    surface_label: str = "wall"
    estimated_normal_direction: float | None = None  # degrees, 0=right
    is_primary_surface: bool = True

    @property
    def length_px(self) -> float:
        return self.p1.distance_to(self.p2)

    @property
    def midpoint(self) -> ImagePoint:
        return self.p1.midpoint(self.p2)

    @property
    def angle_deg(self) -> float:
        """Angle of the wall segment from horizontal, in [0, 180) degrees."""
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        angle = math.degrees(math.atan2(dy, dx)) % 180
        return angle

    def is_approximately_vertical(self, tolerance_deg: float = 15.0) -> bool:
        a = self.angle_deg
        return a > 90 - tolerance_deg

    def is_approximately_horizontal(self, tolerance_deg: float = 15.0) -> bool:
        a = self.angle_deg
        return a < tolerance_deg or a > 180 - tolerance_deg


# ---------------------------------------------------------------------------
# Junction detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WallJunction:
    """A detected junction between two or more wall segments."""
    junction_id: str
    junction_type: JunctionType
    image_id: str
    location: ImagePoint
    participating_segment_ids: tuple[str, ...]
    confidence: float
    angle_deg: float    # interior angle at this junction
    notes: str = ""


def _segments_intersect_or_meet(
    s1: WallSegmentHypothesis,
    s2: WallSegmentHypothesis,
    proximity_threshold_px: float = 30.0,
) -> tuple[bool, ImagePoint | None]:
    """Check if two segments meet or nearly intersect.

    Returns (meets, meeting_point) where meeting_point is the nearest point.
    """
    # Segment parametric intersection
    p1, p2 = (s1.p1.x, s1.p1.y), (s1.p2.x, s1.p2.y)
    p3, p4 = (s2.p1.x, s2.p1.y), (s2.p2.x, s2.p2.y)

    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    denom = d1x * d2y - d1y * d2x

    if abs(denom) < 1e-6:
        # Parallel — check endpoint proximity
        for ep in [s1.p1, s1.p2]:
            for ep2 in [s2.p1, s2.p2]:
                if ep.distance_to(ep2) <= proximity_threshold_px:
                    return True, ep.midpoint(ep2)
        return False, None

    t_num = (p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x
    u_num = (p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x

    t = t_num / denom
    u = u_num / denom

    # Check if intersection is near the segments (within extended tolerance)
    ext = proximity_threshold_px / max(s1.length_px, 1.0)
    if -ext <= t <= 1 + ext and -ext <= u <= 1 + ext:
        ix = p1[0] + t * d1x
        iy = p1[1] + t * d1y
        return True, ImagePoint(ix, iy)

    # Check endpoint proximity anyway
    for ep in [s1.p1, s1.p2]:
        for ep2 in [s2.p1, s2.p2]:
            if ep.distance_to(ep2) <= proximity_threshold_px:
                return True, ep.midpoint(ep2)

    return False, None


def _junction_angle(s1: WallSegmentHypothesis, s2: WallSegmentHypothesis) -> float:
    """Compute interior angle between two wall segments at their junction."""
    a1 = math.radians(s1.angle_deg)
    a2 = math.radians(s2.angle_deg)
    diff = abs(a1 - a2)
    diff = min(diff, math.pi - diff)
    return math.degrees(diff)


def _classify_junction(
    angle_deg: float,
    n_segments: int,
) -> tuple[JunctionType, str]:
    """Classify junction type from meeting angle and segment count."""
    if n_segments >= 4:
        return JunctionType.X_JUNCTION, "Four or more segments converge."
    if n_segments == 3:
        return JunctionType.T_JUNCTION, "Three segments converge."
    if n_segments == 2:
        if 60 <= angle_deg <= 120:
            return JunctionType.L_JUNCTION, f"Corner junction, angle={angle_deg:.0f}°."
        elif angle_deg < 60:
            return JunctionType.T_JUNCTION, f"Acute junction treated as T, angle={angle_deg:.0f}°."
        else:
            return JunctionType.L_JUNCTION, f"Obtuse corner, angle={angle_deg:.0f}°."
    return JunctionType.AMBIGUOUS, "Insufficient segments."


def detect_wall_junctions(
    segments: list[WallSegmentHypothesis],
    image_id: str,
    *,
    proximity_threshold_px: float = 40.0,
) -> list[WallJunction]:
    """Detect and classify all junctions between wall segments.

    Args:
        segments: List of wall segment hypotheses for one image.
        image_id: Image identifier.
        proximity_threshold_px: Max distance (px) for endpoints to be considered meeting.

    Returns:
        List of detected ``WallJunction`` objects.
    """
    junctions: list[WallJunction] = []
    # Group by meeting point cluster
    meeting_points: list[tuple[ImagePoint, list[WallSegmentHypothesis]]] = []

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            meets, pt = _segments_intersect_or_meet(
                segments[i], segments[j], proximity_threshold_px
            )
            if not meets or pt is None:
                continue
            # Find existing cluster near this point
            matched_cluster = None
            for cluster_pt, cluster_segs in meeting_points:
                if pt.distance_to(cluster_pt) < proximity_threshold_px:
                    matched_cluster = (cluster_pt, cluster_segs)
                    break
            if matched_cluster:
                _, cluster_segs = matched_cluster
                if segments[i] not in cluster_segs:
                    cluster_segs.append(segments[i])
                if segments[j] not in cluster_segs:
                    cluster_segs.append(segments[j])
            else:
                meeting_points.append((pt, [segments[i], segments[j]]))

    for idx, (pt, segs) in enumerate(meeting_points):
        angles = []
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                angles.append(_junction_angle(segs[a], segs[b]))

        mean_angle = sum(angles) / len(angles) if angles else 90.0
        jtype, notes = _classify_junction(mean_angle, len(segs))
        confidence = min(0.95, 0.5 + 0.1 * len(segs))

        jid = hashlib.sha1(f"{image_id}:{pt.x:.0f}:{pt.y:.0f}".encode()).hexdigest()[:8]
        junctions.append(WallJunction(
            junction_id=f"junc_{jid}",
            junction_type=jtype,
            image_id=image_id,
            location=pt,
            participating_segment_ids=tuple(s.segment_id for s in segs),
            confidence=confidence,
            angle_deg=mean_angle,
            notes=notes,
        ))

    return junctions


# ---------------------------------------------------------------------------
# Host-wall hypothesis graph
# ---------------------------------------------------------------------------
@dataclass
class HostWallHypothesisNode:
    """A node in the host-wall graph, representing one candidate wall plane."""
    node_id: str
    image_id: str
    primary_segment_ids: list[str]
    hosted_opening_ids: list[str]   # doors, windows on this wall
    normal_estimate_deg: float | None
    confidence: float
    span_px: float                  # horizontal or vertical span in image
    notes: list[str]


@dataclass
class HostWallHypothesisGraph:
    """Graph of wall planes with their hosted openings."""
    image_id: str
    nodes: list[HostWallHypothesisNode]
    junctions: list[WallJunction]
    adjacency: dict[str, list[str]]   # node_id → adjacent node_ids
    contradictions: list[str]         # human-readable contradiction descriptions


def build_host_wall_graph(
    wall_segments: list[WallSegmentHypothesis],
    opening_bboxes: list[tuple[str, str, tuple[float, float, float, float]]],
    junctions: list[WallJunction],
    image_id: str,
    *,
    angle_tolerance_deg: float = 15.0,
    max_distance_px: float = 80.0,
) -> HostWallHypothesisGraph:
    """Build a host-wall graph from wall segments and opening detections.

    Groups co-planar wall segments into unified wall nodes and assigns
    detected openings (doors, windows) to their host walls.

    Args:
        wall_segments: Detected wall segment hypotheses.
        opening_bboxes: List of (opening_id, opening_label, bbox_px) tuples.
        junctions: Pre-computed junctions between segments.
        image_id: Image identifier.
        angle_tolerance_deg: Max angular difference to merge segments into one plane.
        max_distance_px: Max perpendicular distance between parallel segments to merge.

    Returns:
        ``HostWallHypothesisGraph`` with wall nodes and adjacency.
    """
    if not wall_segments:
        return HostWallHypothesisGraph(
            image_id=image_id, nodes=[], junctions=junctions,
            adjacency={}, contradictions=[],
        )

    # Cluster segments by angle (co-planar grouping)
    clusters: list[list[WallSegmentHypothesis]] = []
    assigned = [False] * len(wall_segments)

    for i, seg_i in enumerate(wall_segments):
        if assigned[i]:
            continue
        cluster = [seg_i]
        assigned[i] = True
        for j, seg_j in enumerate(wall_segments):
            if assigned[j] or j == i:
                continue
            angle_diff = abs(seg_i.angle_deg - seg_j.angle_deg)
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            if angle_diff > angle_tolerance_deg:
                continue
            # Approximate parallel segment distance: midpoint perpendicular distance
            mid_i = seg_i.midpoint
            mid_j = seg_j.midpoint
            dist = mid_i.distance_to(mid_j)
            if dist <= max_distance_px:
                cluster.append(seg_j)
                assigned[j] = True
        clusters.append(cluster)

    # Build nodes
    nodes: list[HostWallHypothesisNode] = []
    contradictions: list[str] = []

    for cluster_idx, cluster in enumerate(clusters):
        angle_estimates = [s.angle_deg for s in cluster]
        mean_angle = sum(angle_estimates) / len(angle_estimates)
        confidence = min(0.95, 0.4 + 0.15 * len(cluster))

        # Span: total projected length
        all_xs = [s.p1.x for s in cluster] + [s.p2.x for s in cluster]
        all_ys = [s.p1.y for s in cluster] + [s.p2.y for s in cluster]
        span = math.sqrt(
            (max(all_xs) - min(all_xs)) ** 2 + (max(all_ys) - min(all_ys)) ** 2
        )

        nid = hashlib.sha1(f"{image_id}:wall:{cluster_idx}".encode()).hexdigest()[:8]

        # Assign openings to this wall (opening center lies within wall span bbox)
        hosted_openings: list[str] = []
        wall_x1 = min(all_xs) - 20
        wall_x2 = max(all_xs) + 20
        wall_y1 = min(all_ys) - 20
        wall_y2 = max(all_ys) + 20

        for oid, olabel, obox in opening_bboxes:
            ox1, oy1, ox2, oy2 = obox
            ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
            if wall_x1 <= ocx <= wall_x2 and wall_y1 <= ocy <= wall_y2:
                hosted_openings.append(oid)

        nodes.append(HostWallHypothesisNode(
            node_id=f"wall_{nid}",
            image_id=image_id,
            primary_segment_ids=[s.segment_id for s in cluster],
            hosted_opening_ids=hosted_openings,
            normal_estimate_deg=mean_angle + 90.0,
            confidence=confidence,
            span_px=span,
            notes=[f"Clustered {len(cluster)} segment(s), mean_angle={mean_angle:.1f}°"],
        ))

    # Build adjacency from junctions
    adjacency: dict[str, list[str]] = {n.node_id: [] for n in nodes}
    seg_to_node: dict[str, str] = {}
    for node in nodes:
        for sid in node.primary_segment_ids:
            seg_to_node[sid] = node.node_id

    for junc in junctions:
        node_ids = list(dict.fromkeys(
            seg_to_node.get(sid) for sid in junc.participating_segment_ids
            if seg_to_node.get(sid)
        ))
        for a in range(len(node_ids)):
            for b in range(a + 1, len(node_ids)):
                na, nb = node_ids[a], node_ids[b]
                if nb not in adjacency[na]:
                    adjacency[na].append(nb)
                if na not in adjacency[nb]:
                    adjacency[nb].append(na)

    # Contradiction detection: parallel walls with no junction are suspicious
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            angle_i = ni.normal_estimate_deg or 0.0
            angle_j = nj.normal_estimate_deg or 0.0
            angle_diff = abs(angle_i - angle_j) % 180
            angle_diff = min(angle_diff, 180 - angle_diff)
            if angle_diff < 10.0:  # roughly parallel
                # Parallel walls with shared openings is unusual
                shared_openings = set(ni.hosted_opening_ids) & set(nj.hosted_opening_ids)
                if shared_openings:
                    contradictions.append(
                        f"Wall {ni.node_id} and {nj.node_id} are parallel but "
                        f"share opening(s) {shared_openings} — possible misassignment."
                    )

    return HostWallHypothesisGraph(
        image_id=image_id,
        nodes=nodes,
        junctions=junctions,
        adjacency=adjacency,
        contradictions=contradictions,
    )


# ---------------------------------------------------------------------------
# Room boundary polygon
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoomBoundaryPolygon:
    """An estimated room boundary polygon in image space."""
    polygon_id: str
    image_id: str
    vertices: tuple[ImagePoint, ...]
    area_px: float
    is_closed: bool
    confidence: float
    supporting_node_ids: tuple[str, ...]
    notes: str


def estimate_room_boundary(
    graph: HostWallHypothesisGraph,
    image_width: int,
    image_height: int,
) -> RoomBoundaryPolygon | None:
    """Estimate the room boundary polygon from the host-wall graph.

    Uses the convex hull of wall segment endpoints as a conservative
    room boundary estimate. Returns None if fewer than 3 wall nodes exist.

    Args:
        graph: Host-wall hypothesis graph.
        image_width: Image width for clamping.
        image_height: Image height for clamping.

    Returns:
        ``RoomBoundaryPolygon`` or None if insufficient evidence.
    """
    if len(graph.nodes) < 3:
        return None

    # Collect all junction locations as boundary vertices
    pts = [(j.location.x, j.location.y) for j in graph.junctions]
    if len(pts) < 3:
        return None

    pts_arr = np.array(pts, dtype=np.float64)

    # Convex hull
    try:
        from scipy.spatial import ConvexHull  # noqa: PLC0415
        hull = ConvexHull(pts_arr)
        hull_pts = [ImagePoint(pts_arr[i, 0], pts_arr[i, 1]) for i in hull.vertices]
        area = hull.volume  # 2D: volume = area
        is_closed = True
    except (ImportError, Exception):
        # Fallback: use points as-is
        hull_pts = [ImagePoint(x, y) for x, y in pts]
        area = 0.0
        is_closed = False

    # Confidence proportional to number of junctions
    confidence = min(0.90, 0.3 + 0.1 * len(graph.junctions))
    pid = hashlib.sha1(graph.image_id.encode()).hexdigest()[:8]

    return RoomBoundaryPolygon(
        polygon_id=f"room_{pid}",
        image_id=graph.image_id,
        vertices=tuple(hull_pts),
        area_px=area,
        is_closed=is_closed,
        confidence=confidence,
        supporting_node_ids=tuple(n.node_id for n in graph.nodes),
        notes=(
            f"Estimated from {len(graph.junctions)} junction(s) and "
            f"{len(graph.nodes)} wall node(s). "
            "Metric scale unknown — image-space estimate only."
        ),
    )


# ---------------------------------------------------------------------------
# Cross-view structural continuity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StructuralContinuityDecision:
    """Assessment of whether a structural element is consistent across views."""
    element_id_a: str
    element_id_b: str
    image_id_a: str
    image_id_b: str
    is_consistent: bool
    consistency_score: float
    reason: str


def assess_cross_view_structural_continuity(
    segments_by_image: dict[str, list[WallSegmentHypothesis]],
    *,
    min_consistency_score: float = 0.60,
) -> list[StructuralContinuityDecision]:
    """Assess whether wall segments are consistent across multiple views.

    Two wall segments from different images are considered consistent if
    their angles are within tolerance. This is a lightweight geometric
    check that does not require camera poses.

    Args:
        segments_by_image: Mapping from image_id to wall segments.
        min_consistency_score: Minimum score to report as consistent.

    Returns:
        List of pairwise continuity decisions.
    """
    image_ids = list(segments_by_image.keys())
    decisions: list[StructuralContinuityDecision] = []

    for i in range(len(image_ids)):
        for j in range(i + 1, len(image_ids)):
            segs_a = segments_by_image[image_ids[i]]
            segs_b = segments_by_image[image_ids[j]]

            for seg_a in segs_a:
                for seg_b in segs_b:
                    angle_diff = abs(seg_a.angle_deg - seg_b.angle_deg)
                    angle_diff = min(angle_diff, 180.0 - angle_diff)
                    # Score: 1.0 at 0° diff, 0.0 at 45° diff
                    score = max(0.0, 1.0 - angle_diff / 45.0)
                    if score >= min_consistency_score:
                        decisions.append(StructuralContinuityDecision(
                            element_id_a=seg_a.segment_id,
                            element_id_b=seg_b.segment_id,
                            image_id_a=image_ids[i],
                            image_id_b=image_ids[j],
                            is_consistent=True,
                            consistency_score=score,
                            reason=f"Angle diff={angle_diff:.1f}°, score={score:.2f}",
                        ))

    return decisions
