"""Object-level occlusion reasoning and object permanence across views.

Builds on the existing point-level visibility primitives
(``engine.multiengine.visibility_reprojection``) to reason about occlusion
between *objects* in a scene:

* For each camera view, every pair of objects is tested: the nearer object
  (smaller depth at their overlapping image region) occludes the farther one.
* Occlusion relations form a directed ``OcclusionGraph`` (A -occludes-> B
  from camera C). Chains (A occludes B occludes C) are computed transitively.
* From the occlusion graph we derive per-view *occluded regions* of each
  object and an *object permanence* judgement: an object counts as present in
  a view when it is either visible or occluded by another object (it persists
  even while hidden).

All functions are pure and deterministic: the same inputs always produce the
same graph. Depth evidence is treated honestly -- when a depth map is
missing or relative (not metric), occlusion can only be inferred from
2D box overlap and is flagged as such, never claimed as metric fact.

This module is KiloCode-owned spatial intelligence (WORKSTREAM 2). It does
not replace ``world_ir/relations.py`` (box-derived spatial relations) or
``engine.multiengine.visibility_reprojection.py`` (point-level visibility);
it consumes box + depth evidence and produces object-level occlusion and
permanence judgements that those layers can consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "OcclusionRelation",
    "OcclusionGraph",
    "OccludedRegion",
    "ObjectPermanence",
    "compute_object_occlusions",
    "occluded_region_of",
    "object_permanence_in_view",
    "transitive_occlusion_chain",
]


# --------------------------------------------------------------------------- data model

@dataclass(frozen=True)
class OcclusionRelation:
    """One object occluding another, observed from one camera.

    ``occluder_id`` blocks (part of) ``occluded_id`` in ``camera_id``'s view.
    ``overlap_px`` is the area (in source-image pixels) of the occluder's
    box that overlaps the occluded box. ``depth_separation_m`` is the
    difference in measured depth between the two objects at the overlap
    region, or None when only 2D overlap is available (relative-depth /
    no-depth evidence -- honest but weaker).
    """

    occluder_id: str
    occluded_id: str
    camera_id: str
    overlap_px: float
    depth_separation_m: Optional[float]
    confidence: float
    metric: bool  # True only when depth_separation_m is physically calibrated


@dataclass(frozen=True)
class OccludedRegion:
    """The part of one object hidden behind another, in image pixels."""

    object_id: str
    hidden_by: str
    camera_id: str
    hidden_box_px: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    hidden_fraction: float  # hidden area / object box area, [0, 1]


@dataclass(frozen=True)
class OcclusionGraph:
    """Directed object->object occlusion graph for one scene/view-set.

    ``relations`` holds every observed (occluder, occluded, camera) triple.
    ``by_camera`` indexes relations per camera. ``adjacency`` maps each
    object to the set of objects it directly occludes (union over cameras).
    """

    relations: tuple[OcclusionRelation, ...]
    by_camera: dict[str, tuple[OcclusionRelation, ...]] = field(default_factory=dict)
    adjacency: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def directly_occludes(self, object_id: str) -> tuple[str, ...]:
        return self.adjacency.get(object_id, ())

    def is_occluded_by(self, object_id: str, camera_id: str) -> tuple[str, ...]:
        return tuple(
            r.occluder_id
            for r in self.by_camera.get(camera_id, ())
            if r.occluded_id == object_id
        )


@dataclass(frozen=True)
class ObjectPermanence:
    """Whether an object is judged present in one camera view.

    ``present`` is True when the object is visible OR occluded by another
    object in that view (it persists even while hidden). ``visible`` is True
    only when some part of it is not hidden. ``occluded_by`` lists the objects
    hiding it in this view. ``confidence`` reflects the strength of the
    evidence (lower when only 2D overlap, no depth, is available).
    """

    object_id: str
    camera_id: str
    present: bool
    visible: bool
    occluded_by: tuple[str, ...]
    confidence: float
    reason: str


# --------------------------------------------------------------------------- geometry helpers

def _box_area(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return (0.0, 0.0, 0.0, 0.0)
    return (x0, y0, x1, y1)


def _depth_at_box(
    depth_map: Optional[np.ndarray],
    box: tuple[float, float, float, float],
) -> Optional[float]:
    """Median depth inside a box region, or None when no depth evidence."""
    if depth_map is None or depth_map.size == 0:
        return None
    h, w = depth_map.shape
    x0 = int(math.floor(max(0.0, box[0]))); y0 = int(math.floor(max(0.0, box[1])))
    x1 = int(math.ceil(min(float(w), box[2])));  y1 = int(math.ceil(min(float(h), box[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    region = depth_map[y0:y1, x0:x1]
    region = region[np.isfinite(region) & (region > 0.0)]
    if region.size == 0:
        return None
    return float(np.median(region))


# --------------------------------------------------------------------------- core computation

def compute_object_occlusions(
    *,
    objects: dict[str, tuple[float, float, float, float]],
    camera_id: str,
    depth_map: Optional[np.ndarray] = None,
    depth_metric: bool = False,
    iou_threshold: float = 0.02,
    max_occlusion_confidence: float = 0.97,
) -> OcclusionGraph:
    """Determine pairwise object occlusion from one camera view.

    For every pair of objects whose boxes overlap by at least
    ``iou_threshold`` (intersection-over-union), the nearer object
    (smaller median depth in the overlap region) is taken to occlude the
    farther one. When no depth evidence is available the pair is still
    recorded as a *potential* occlusion (2D overlap only) with lower
    confidence and ``metric=False``.

    Parameters
    ----------
    objects:
        ``object_id -> (x0, y0, x1, y1)`` source-image pixel boxes.
    camera_id:
        The camera that produced this view (stored on every relation).
    depth_map:
        Optional metric depth image aligned to ``objects``' image frame.
    depth_metric:
        True when ``depth_map`` values are physically calibrated metres.
    iou_threshold:
        Minimum IoU for two boxes to be considered overlapping.
    max_occlusion_confidence:
        Ceiling on the confidence of any single-view occlusion relation.

    Returns
    -------
    OcclusionGraph
        Directed graph of ``occluder -> occluded`` relations for this view.
    """
    ids = sorted(objects)
    boxes = {oid: objects[oid] for oid in ids}
    relations: list[OcclusionRelation] = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            box_a, box_b = boxes[a], boxes[b]
            inter = _intersection(box_a, box_b)
            inter_area = _box_area(inter)
            if inter_area <= 0.0:
                continue
            union_area = _box_area(box_a) + _box_area(box_b) - inter_area
            if union_area <= 0.0 or (inter_area / union_area) < iou_threshold:
                continue

            # Depth evidence at the overlap region decides which is nearer.
            overlap_depth_a = _depth_at_box(depth_map, inter) if depth_map is not None else None
            overlap_depth_b = overlap_depth_a  # same overlap region -> same depth sample
            # Per-object median depth (full box) as a tie-breaker / strength signal.
            depth_a = _depth_at_box(depth_map, box_a) if depth_map is not None else None
            depth_b = _depth_at_box(depth_map, box_b) if depth_map is not None else None

            # Confidence from overlap fraction + depth separation certainty.
            overlap_fraction = inter_area / min(_box_area(box_a), _box_area(box_b))
            conf = min(max_occlusion_confidence, 0.5 + 0.45 * overlap_fraction)

            if depth_a is not None and depth_b is not None and abs(depth_a - depth_b) > 1e-4:
                # Metric depth available: nearer object occludes farther.
                nearer, farther = (a, b) if depth_a < depth_b else (b, a)
                sep = abs(depth_a - depth_b)
                metric_conf = min(max_occlusion_confidence, conf * (1.0 + min(sep, 1.0)))
                relations.append(OcclusionRelation(
                    occluder_id=nearer, occluded_id=farther, camera_id=camera_id,
                    overlap_px=inter_area, depth_separation_m=sep,
                    confidence=round(metric_conf, 3), metric=depth_metric,
                ))
            elif depth_map is not None:
                # Depth present but objects at indistinguishable depth: weak 2D-only.
                # No definite nearer object; record a potential occlusion in both
                # directions at low confidence so the caller can see the ambiguity.
                for occluder, occluded in ((a, b), (b, a)):
                    relations.append(OcclusionRelation(
                        occluder_id=occluder, occluded_id=occluded, camera_id=camera_id,
                        overlap_px=inter_area, depth_separation_m=None,
                        confidence=round(conf * 0.5, 3), metric=False,
                    ))
            else:
                # No depth: 2D overlap only, potential occlusion, low confidence.
                for occluder, occluded in ((a, b), (b, a)):
                    relations.append(OcclusionRelation(
                        occluder_id=occluder, occluded_id=occluded, camera_id=camera_id,
                        overlap_px=inter_area, depth_separation_m=None,
                        confidence=round(conf * 0.4, 3), metric=False,
                    ))

    # Build adjacency: object -> objects it directly occludes (metric relations win).
    adjacency: dict[str, set[str]] = {oid: set() for oid in ids}
    by_camera: dict[str, list[OcclusionRelation]] = {camera_id: []}
    for r in relations:
        by_camera[camera_id].append(r)
        if r.metric:
            adjacency[r.occluder_id].add(r.occluded_id)

    return OcclusionGraph(
        relations=tuple(relations),
        by_camera={k: tuple(v) for k, v in by_camera.items()},
        adjacency={k: tuple(sorted(v)) for k, v in adjacency.items()},
    )


def merge_graphs(*graphs: OcclusionGraph) -> OcclusionGraph:
    """Merge several per-camera occlusion graphs into one scene graph.

    Union of relations, adjacency, and per-camera indexes. Duplicate
    (occluder, occluded, camera) triples are collapsed.
    """
    seen: set[tuple[str, str, str]] = set()
    relations: list[OcclusionRelation] = []
    by_camera: dict[str, list[OcclusionRelation]] = {}
    adjacency: dict[str, set[str]] = {}

    for g in graphs:
        for r in g.relations:
            key = (r.occluder_id, r.occluded_id, r.camera_id)
            if key in seen:
                continue
            seen.add(key)
            relations.append(r)
            by_camera.setdefault(r.camera_id, []).append(r)
            if r.metric:
                adjacency.setdefault(r.occluder_id, set()).add(r.occluded_id)

    return OcclusionGraph(
        relations=tuple(relations),
        by_camera={k: tuple(v) for k, v in by_camera.items()},
        adjacency={k: tuple(sorted(v)) for k, v in adjacency.items()},
    )


def transitive_occlusion_chain(graph: OcclusionGraph, start: str) -> tuple[str, ...]:
    """Objects reachable from ``start`` by following ``directly_occludes`` edges.

    Returns the closure (excluding ``start`` itself) in deterministic order.
    A chain A->B->C means A (transitively) occludes C through B.
    """
    seen: set[str] = set()
    frontier = list(graph.directly_occludes(start))
    order: list[str] = []
    while frontier:
        node = frontier.pop(0)
        if node in seen or node == start:
            continue
        seen.add(node)
        order.append(node)
        frontier.extend(graph.directly_occludes(node))
    return tuple(order)


# --------------------------------------------------------------------------- derived judgements

def occluded_region_of(
    graph: OcclusionGraph,
    object_id: str,
    boxes: dict[str, tuple[float, float, float, float]],
) -> list[OccludedRegion]:
    """Image regions of ``object_id`` hidden behind other objects (per camera).

    For each direct occluder recorded in ``graph`` for ``object_id``, the
    hidden region is the intersection of the object's box with the occluder's
    box (the occluder is assumed to be in front). ``hidden_fraction`` is the
    hidden area over the object's box area.
    """
    regions: list[OccludedRegion] = []
    box = boxes.get(object_id)
    if box is None:
        return regions
    box_area = _box_area(box)
    if box_area <= 0.0:
        return regions
    for r in graph.relations:
        if r.occluded_id != object_id:
            continue
        occluder_box = boxes.get(r.occluder_id)
        if occluder_box is None:
            continue
        hidden = _intersection(box, occluder_box)
        hidden_area = _box_area(hidden)
        if hidden_area <= 0.0:
            continue
        regions.append(OccludedRegion(
            object_id=object_id,
            hidden_by=r.occluder_id,
            camera_id=r.camera_id,
            hidden_box_px=hidden,
            hidden_fraction=round(hidden_area / box_area, 4),
        ))
    regions.sort(key=lambda x: (-x.hidden_fraction, x.hidden_by, x.camera_id))
    return regions


def object_permanence_in_view(
    graph: OcclusionGraph,
    object_id: str,
    camera_id: str,
    visible_boxes: dict[str, tuple[float, float, float, float]],
) -> ObjectPermanence:
    """Judge whether ``object_id`` is present in ``camera_id``'s view.

    An object is *present* when it is visible in this view OR occluded by
    another object (it persists even while hidden). It is *visible* only
    when some part of its box is not hidden by any occluder.
    """
    relations = graph.by_camera.get(camera_id, ())
    occluders = tuple(
        r.occluder_id for r in relations if r.occluded_id == object_id
    )
    if object_id not in visible_boxes:
        # No box at all in this view: not present (no evidence).
        return ObjectPermanence(
            object_id=object_id, camera_id=camera_id, present=False,
            visible=False, occluded_by=(), confidence=0.0,
            reason="no box observed in this view",
        )

    hidden_fractions = [
        region.hidden_fraction for region in occluded_region_of(graph, object_id, visible_boxes)
        if region.camera_id == camera_id
    ]
    total_hidden = min(1.0, sum(hidden_fractions)) if hidden_fractions else 0.0
    visible = total_hidden < 1.0
    present = visible or len(occluders) > 0

    if visible:
        confidence = 0.9 if not occluders else 0.7
        reason = "visible" if not occluders else "partially visible (occluded by others)"
    else:
        confidence = 0.6 if occluders else 0.0
        reason = "fully occluded -- present but hidden" if occluders else "not visible, no occluder"

    return ObjectPermanence(
        object_id=object_id, camera_id=camera_id, present=present,
        visible=visible, occluded_by=occluders, confidence=round(confidence, 3),
        reason=reason,
    )
