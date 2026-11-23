"""Image-level detections: group what DIFFERENT engines saw of the SAME object in ONE image.

Cross-view association (``association.associate``) forbids two members from one image in one entity -- correct for two different
physical objects. But under the maximum-fidelity policy several detectors each report the same chair in the same photo; treating each
report as a separate object would triple every entity. So detections are grouped first, then groups are associated across views.

Rules:
* observations are grouped only within one source image, by box IoU >= ``iou_min`` (a stated parameter, not a tuned default);
* an engine never joins two of its OWN detections (two boxes from one engine are two objects, however much they overlap) -- with ONE narrow
  exception, ``DUPLICATE_BOX_IOU``: one engine emitting the SAME box (IoU >= 0.95) twice with DIFFERENT labels, neither the parent/child part
  of the other, is one physical object whose class the engine itself is unsure of (a multi-label emission), not two objects. Measured on real
  photographs: a wall plate boxed identically as "socket" and "switchboard plate" became two entities, one never lifted to 3D. Two identical
  boxes with the SAME label are still two reports and are not merged here;
* labels are NOT used to group: two engines that box the same object but disagree on its class are one detection with a class
  conflict, which canonical fusion then reports -- hiding that disagreement by splitting the group would be the wrong fix;
* strongest overlap first, ties by observation id, so the result is independent of input order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from engine.multiengine.observation import EngineObservation


#: one engine, one image, near-identical box, different label: the same object with a class conflict (see the module doc)
DUPLICATE_BOX_IOU = 0.95


def _related(a: EngineObservation, b: EngineObservation) -> bool:
    """True when one box is a recorded PART (child/parent) of the other: those are two objects even at identical extents."""
    pa, pb = a.predicted or {}, b.predicted or {}
    ida, idb = pa.get("pipeline_box_id"), pb.get("pipeline_box_id")
    return bool((ida and pb.get("parent_pipeline_box_id") == ida) or (idb and pa.get("parent_pipeline_box_id") == idb)
                or (idb and idb in (pa.get("child_box_ids") or ())) or (ida and ida in (pb.get("child_box_ids") or ())))


@dataclass(frozen=True)
class DetectionGroup:
    group_id: str
    source_id: str
    representative: str  # observation id used to stand for the group (largest box area; ties by id)
    members: tuple[str, ...]  # observation ids, sorted
    engines: tuple[str, ...]  # distinct engines that reported it


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def group_detections(observations: Sequence[EngineObservation], *, iou_min: float) -> list[DetectionGroup]:
    if not 0.0 < iou_min <= 1.0:
        raise ValueError("iou_min must be in (0, 1]")
    obs = [o for o in observations if o.image_region is not None]
    parent = {o.observation_id: o.observation_id for o in obs}
    engines: dict[str, set[str]] = {o.observation_id: {o.engine_id} for o in obs}
    by_id = {o.observation_id: o for o in obs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    duplicates = set()  # same-engine multi-label emissions of one box: allowed to join despite the one-detection-per-engine rule
    for i, a in enumerate(obs):
        for b in obs[i + 1 :]:
            if a.source_id != b.source_id:
                continue
            iou = box_iou(a.image_region, b.image_region)
            if a.engine_id == b.engine_id:
                la, lb = (a.predicted or {}).get("label"), (b.predicted or {}).get("label")
                if iou >= DUPLICATE_BOX_IOU and la and lb and la != lb and not _related(a, b):
                    key = tuple(sorted((a.observation_id, b.observation_id)))
                    duplicates.add(key)
                    pairs.append((-iou, *key))
                continue
            if iou >= iou_min:
                pairs.append((-iou, *sorted((a.observation_id, b.observation_id))))
    for _, x, y in sorted(pairs):
        rx, ry = find(x), find(y)
        if rx == ry or (engines[rx] & engines[ry] and (x, y) not in duplicates):  # an engine never contributes two detections to one group
            continue
        parent[ry] = rx
        engines[rx] |= engines[ry]
    groups: dict[str, list[str]] = {}
    for oid in by_id:
        groups.setdefault(find(oid), []).append(oid)
    out = []
    for members in groups.values():
        ms = tuple(sorted(members))

        def area(i: str) -> float:
            r = by_id[i].image_region
            return (r[2] - r[0]) * (r[3] - r[1])

        rep = max(ms, key=lambda i: (area(i), i))
        gid = "det_" + hashlib.sha256("|".join(ms).encode()).hexdigest()[:20]
        out.append(DetectionGroup(gid, by_id[rep].source_id, rep, ms, tuple(sorted({by_id[i].engine_id for i in ms}))))
    return sorted(out, key=lambda g: g.members)
