"""Entity completeness: what a region should contain, what was found, and why "done" cannot be claimed.

Completeness is never 100% merely because everything detected was reconstructed: expected categories may have no
detection, parts of the region may be occluded or never inspected, detections may be unresolved. The report states each
gap; ``complete_claim`` is True only when every one of them is empty AND the whole region was inspected unoccluded --
which for a real capture should essentially never happen.
Expectations are PRIORS about typical rooms, labelled as such; they are not measurements.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: region kind -> {class id: (min expected, max expected or None)}  (typical priors, not measurements)
EXPECTED: Dict[str, Dict[str, Tuple[int, Optional[int]]]] = {
    "room": {"envelope.wall": (3, None), "envelope.floor": (1, 1), "envelope.ceiling": (0, 1), "opening.door": (1, None),
             "electrical.switchboard": (1, None), "electrical.light_fitting": (1, None)},
    "room.bathroom": {"envelope.wall": (3, None), "envelope.floor": (1, 1), "opening.door": (1, 1),
                      "plumbing.toilet": (0, None), "plumbing.sink": (1, None), "plumbing.tap": (1, None),
                      "electrical.switchboard": (1, None)},
    "room.kitchen": {"envelope.wall": (3, None), "envelope.floor": (1, 1), "plumbing.sink": (1, None),
                     "electrical.switchboard": (1, None)},
    "building.exterior": {"exterior.roof": (1, None), "opening.door": (1, None), "opening.window": (0, None)},
    "street": {"road.carriageway": (1, None), "road.footpath": (0, None), "road.kerb": (0, None)},
}


def completeness_report(region_id: str, region_kind: str, entities: Sequence[Mapping], *,
                        inspected_fraction: float, occluded_fraction: float = 0.0,
                        uninspected_note: str = "", expected: Optional[Mapping[str, Tuple[int, Optional[int]]]] = None
                        ) -> dict:
    """``entities``: dicts with ``class_id`` and ``status`` in {"reconstructed", "partial", "detected_only", "unresolved"}."""
    exp = dict(expected if expected is not None else EXPECTED.get(region_kind, EXPECTED.get(region_kind.split(".")[0], {})))
    by_class: Dict[str, Dict[str, int]] = {}
    for e in entities:
        c = by_class.setdefault(e["class_id"], {"detected": 0, "reconstructed": 0, "partial": 0, "unresolved": 0})
        c["detected"] += 1
        st = e.get("status", "detected_only")
        if st in ("reconstructed", "partial", "unresolved"):
            c[st] += 1
    rows = []
    gaps: List[str] = []
    for cid, (lo, hi) in sorted(exp.items()):
        c = by_class.get(cid, {"detected": 0, "reconstructed": 0, "partial": 0, "unresolved": 0})
        status = ("missing_evidence" if c["detected"] < lo else "above_expected" if hi is not None and c["detected"] > hi
                  else "as_expected")
        if status == "missing_evidence":
            gaps.append(f"expected >= {lo} x {cid}, found {c['detected']}")
        rows.append({"class_id": cid, "expected": [lo, hi], "expected_basis": "typical-room prior, not measured", **c,
                     "status": status})
    unexpected = sorted(c for c in by_class if c not in exp)
    unresolved = sum(c["unresolved"] for c in by_class.values())
    partial = sum(c["partial"] for c in by_class.values())
    if unresolved:
        gaps.append(f"{unresolved} unresolved detection(s)")
    if partial:
        gaps.append(f"{partial} partially reconstructed entit(ies)")
    if occluded_fraction > 0:
        gaps.append(f"{occluded_fraction:.0%} of the region is occluded")
    if inspected_fraction < 1.0:
        gaps.append(f"{1 - inspected_fraction:.0%} of the region was never inspected"
                    + (f" ({uninspected_note})" if uninspected_note else ""))
    n_det = sum(c["detected"] for c in by_class.values())
    n_rec = sum(c["reconstructed"] for c in by_class.values())
    coverage = max(0.0, min(1.0, inspected_fraction * (1.0 - occluded_fraction)))
    expected_met = (sum(1 for r in rows if r["status"] != "missing_evidence") / len(rows)) if rows else None
    # an upper-bound estimate from coverage and expected categories, never a promise: capped below 1
    est = None if expected_met is None else round(min(0.95, coverage * expected_met * (n_rec / n_det if n_det else 0.0)), 3)
    return {
        "region_id": region_id, "region_kind": region_kind, "expected": rows, "unexpected_classes": unexpected,
        "detected": n_det, "reconstructed": n_rec, "unresolved": unresolved,
        "inspected_fraction": inspected_fraction, "occluded_fraction": occluded_fraction,
        "evidence_coverage": round(coverage, 3), "completeness_estimate": est,
        "completeness_note": "an upper-bound estimate from coverage and expected categories; never a claim of 100%",
        "gaps": gaps, "complete_claim": not gaps and coverage >= 1.0 and bool(rows),
    }
