"""Observed-vs-Inferred Spatial Scene Graph Builder for Perception Observations.

Constructs entity-to-entity relationships directly from multi-engine
``ObservationRecord`` outputs (and optional depth maps), strictly distinguishing:
- ``provenance="observed"``: directly supported by same-view bounding-box/mask
  containment, overlap, vertical ordering, and (when available) depth proximity.
- ``provenance="inferred"``: structural/contextual priors (e.g. an isolated wall
  switch without a co-detected wall polygon) that must never be mistaken for
  direct geometric observation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from perception.engine_evidence import ObservationRecord
from provenance import Provenance
from world_ir.schema_v1 import Relationship, RelationshipKind

EdgeProvenance = Literal["observed", "inferred"]

_HOST_LABELS = frozenset({
    "wall", "wall_panel", "partition", "door", "window", "cabinet",
    "switchboard", "distribution_panel", "stair_assembly", "ceiling", "floor",
    "table", "desk", "counter", "shelf", "building_facade",
})

_COMPONENT_TO_HOST: dict[str, tuple[str, str]] = {
    "switch": ("attached_to", "wall"),
    "electrical_socket": ("attached_to", "wall"),
    "socket": ("attached_to", "wall"),
    "plug_point": ("attached_to", "wall"),
    "switchboard": ("attached_to", "wall"),
    "junction_box": ("attached_to", "wall"),
    "handle": ("part_of", "door"),
    "door_handle": ("part_of", "door"),
    "hinge": ("part_of", "door"),
    "lock": ("part_of", "door"),
    "door_frame": ("part_of", "door"),
    "window_frame": ("part_of", "window"),
    "ceiling_fan": ("attached_to", "ceiling"),
    "light_fixture": ("attached_to", "ceiling"),
    "smoke_detector": ("attached_to", "ceiling"),
    "sprinkler": ("attached_to", "ceiling"),
}


@dataclass(frozen=True)
class SceneGraphEdge:
    """Directed relationship between two observed or inferred scene entities."""

    source_id: str
    target_id: str
    kind: str
    provenance: EdgeProvenance
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneGraphReport:
    """Scene graph built from perception observations."""

    edges: tuple[SceneGraphEdge, ...]
    observed_count: int
    inferred_count: int


def _bbox_metrics(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Return (iou, a_in_b_containment, b_in_a_containment)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    union = max(1e-6, area_a + area_b - inter)
    return inter / union, inter / area_a, inter / area_b


def _median_depth_for_obs(
    obs: ObservationRecord,
    depth_map: np.ndarray | None,
) -> float | None:
    if depth_map is None or obs.bbox_px is None or depth_map.ndim != 2:
        return None
    dh, dw = depth_map.shape
    x1, y1, x2, y2 = obs.bbox_px
    ix1 = max(0, min(dw - 1, int(round(x1))))
    iy1 = max(0, min(dh - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(dw, int(round(x2))))
    iy2 = max(iy1 + 1, min(dh, int(round(y2))))
    crop = depth_map[iy1:iy2, ix1:ix2]
    finite = crop[np.isfinite(crop)]
    return float(np.median(finite)) if finite.size > 0 else None


def _short_label(obs: ObservationRecord) -> str:
    ont = str(obs.metadata.get("ontology_path") or obs.label or "").lower()
    return ont.split("/")[-1].split(".")[-1]


def build_scene_graph_from_observations(
    observations: Sequence[ObservationRecord],
    depth_maps: dict[str, np.ndarray] | None = None,
    *,
    include_inferred_priors: bool = True,
    containment_threshold: float = 0.70,
) -> SceneGraphReport:
    """Build observed-vs-inferred scene graph edges from perception observations."""
    edges: list[SceneGraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    by_image: dict[str, list[ObservationRecord]] = {}
    for obs in observations:
        if obs.bbox_px is None or len(obs.bbox_px) != 4:
            continue
        by_image.setdefault(obs.source_image_id, []).append(obs)

    hosts_found: set[str] = set()

    for img_id, group in by_image.items():
        dmap = depth_maps.get(img_id) if depth_maps else None
        for i, a in enumerate(group):
            la = _short_label(a)
            da = _median_depth_for_obs(a, dmap)
            for j, b in enumerate(group):
                if i == j:
                    continue
                lb = _short_label(b)
                if lb not in _HOST_LABELS and la != lb:
                    continue
                iou, a_in_b, _b_in_a = _bbox_metrics(a.bbox_px, b.bbox_px)  # type: ignore[arg-type]
                if a_in_b < containment_threshold or iou > 0.92:
                    continue

                db = _median_depth_for_obs(b, dmap)
                depth_compatible = True
                depth_delta = None
                if da is not None and db is not None:
                    depth_delta = abs(da - db)
                    denom = max(1e-3, max(abs(da), abs(db)))
                    depth_compatible = (depth_delta / denom) <= 0.35

                if not depth_compatible:
                    continue

                kind = "part_of" if lb in {"door", "window", "switchboard", "cabinet"} else "attached_to"
                if lb in {"floor", "table", "desk", "counter", "shelf"} and la not in {"floor", "wall"}:
                    kind = "rests_on"
                key = (a.evidence_id, b.evidence_id, kind)
                if key in seen:
                    continue
                seen.add(key)
                hosts_found.add(a.evidence_id)
                conf = round(min(0.95, 0.55 + 0.35 * a_in_b), 4)
                edges.append(SceneGraphEdge(
                    source_id=a.evidence_id,
                    target_id=b.evidence_id,
                    kind=kind,
                    provenance="observed",
                    confidence=conf,
                    evidence={
                        "image_id": img_id,
                        "containment": round(a_in_b, 4),
                        "iou": round(iou, 4),
                        "depth_delta": round(depth_delta, 4) if depth_delta is not None else None,
                    },
                ))

    if include_inferred_priors:
        for obs in observations:
            if obs.evidence_id in hosts_found:
                continue
            la = _short_label(obs)
            prior = _COMPONENT_TO_HOST.get(la)
            if prior is None:
                continue
            kind, expected_host = prior
            target_id = f"inferred:{obs.source_image_id}:{expected_host}"
            key = (obs.evidence_id, target_id, kind)
            if key in seen:
                continue
            seen.add(key)
            edges.append(SceneGraphEdge(
                source_id=obs.evidence_id,
                target_id=target_id,
                kind=kind,
                provenance="inferred",
                confidence=0.35,
                evidence={
                    "image_id": obs.source_image_id,
                    "reason": f"no co-detected {expected_host} host containing {la}",
                },
            ))

    obs_count = sum(1 for e in edges if e.provenance == "observed")
    inf_count = len(edges) - obs_count
    return SceneGraphReport(
        edges=tuple(edges),
        observed_count=obs_count,
        inferred_count=inf_count,
    )


def to_world_ir_relationships(edges: Sequence[SceneGraphEdge]) -> list[Relationship]:
    """Convert SceneGraphEdges into WorldIR Relationships with explicit Provenance."""
    kind_map = {
        "attached_to": RelationshipKind.ATTACHED_TO,
        "rests_on": RelationshipKind.RESTS_ON,
        "part_of": RelationshipKind.PART_OF,
        "contains": RelationshipKind.CONTAINS,
        "supports": RelationshipKind.SUPPORTS,
    }
    out: list[Relationship] = []
    for e in edges:
        rkind = kind_map.get(e.kind, RelationshipKind.ADJACENT_TO)
        prov = Provenance.OBSERVED if e.provenance == "observed" else Provenance.INFERRED
        out.append(Relationship(
            kind=rkind,
            target_id=e.target_id,
            confidence=e.confidence,
            provenance=prov,
            metadata={"source_id": e.source_id, **e.evidence},
        ))
    return out
