"""The per-image entity inventory: the contract a run must produce, and the evaluator that scores it against annotations.

A record is one observation of one entity in one image. Mandatory fields are never silently dropped: a field that was not
measured is ``None`` AND listed in ``unavailable`` with the reason. Ground truth uses the same shape plus a physical
``instance_id`` (the same chair in two photographs shares it); predictions carry the engine's canonical ``entity_id``.

Detection is scored at IoU >= 0.5 with one-to-one matching. Nothing here knows any image: annotations are input.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks.reality_benchmark.registry import ENTITY_GROUPS, METRIC_BY_ID
from benchmarks.reality_benchmark.scorer import evaluate_metric, wilson_ci

IOU_MATCH = 0.5
#: mandatory inventory fields
REQUIRED = ("image_id", "entity_id", "category", "subtype", "bbox")
#: fields that may be None only when ``unavailable`` says why
MEASURABLE = ("instance_count", "mask", "position_3d", "dimensions", "orientation", "material", "visibility", "evidence")
VISIBILITY = ("visible", "partial", "occluded")
_DOOR = {"door", "door_frame", "handle", "hinge", "lock"}
_WINDOW = {"window", "window_frame", "grill", "latch"}
_PLANT = {"plant", "leaf", "pot", "stem", "vegetation", "tree"}


def validate_inventory(records: Sequence[dict], *, truth: bool = False) -> List[str]:
    """Errors in an inventory (empty = valid). Truth needs ``instance_id``; predictions need a confidence in [0, 1]."""
    errs: List[str] = []
    for i, r in enumerate(records):
        tag = f"[{i}] {r.get('image_id')}/{r.get('entity_id')}"
        for k in REQUIRED:
            if r.get(k) in (None, "", []):
                errs.append(f"{tag}: missing {k}")
        b = r.get("bbox")
        if b is not None and not (len(b) == 4 and b[2] > b[0] and b[3] > b[1]):
            errs.append(f"{tag}: bbox must be [x0, y0, x1, y1] with x1 > x0 and y1 > y0")
        if truth and not r.get("instance_id"):
            errs.append(f"{tag}: ground truth needs an instance_id")
        if not truth:
            c = r.get("confidence")
            if c is None or not (0.0 <= c <= 1.0):
                errs.append(f"{tag}: prediction needs a confidence in [0, 1]")
        v = r.get("visibility")
        vis = v.get("state") if isinstance(v, dict) else v
        if vis is not None and vis not in VISIBILITY:
            errs.append(f"{tag}: visibility must be one of {VISIBILITY}")
        for k in MEASURABLE:                      # never silently omitted: None must come with a stated reason
            if r.get(k) is None and k not in (r.get("unavailable") or {}):
                errs.append(f"{tag}: {k} is None without an 'unavailable' reason")
    return errs


def fine_group(rec: dict) -> Optional[str]:
    """The fine-grained entity group (benchmark category E) of a record, or None when it belongs to none."""
    cat, sub = (rec.get("category") or "").lower(), (rec.get("subtype") or "").lower()
    if cat in ENTITY_GROUPS:
        return cat
    if sub in _DOOR:
        return "doors"
    if sub in _WINDOW:
        return "windows"
    if sub in _PLANT:
        return "vegetation"
    return None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _match(pred: Sequence[dict], truth: Sequence[dict], thr: float = IOU_MATCH, same_subtype: bool = True) -> List[Tuple[int, int]]:
    """One-to-one greedy matching per image by descending IoU (and equal subtype when asked)."""
    by_img: Dict[str, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for i, p in enumerate(pred):
        by_img[p["image_id"]][0].append(i)
    for j, t in enumerate(truth):
        by_img[t["image_id"]][1].append(j)
    out: List[Tuple[int, int]] = []
    for pi, ti in by_img.values():
        pairs = sorted(((iou(pred[i]["bbox"], truth[j]["bbox"]), i, j) for i in pi for j in ti
                        if not same_subtype or pred[i]["subtype"] == truth[j]["subtype"]), reverse=True)
        used_p, used_t = set(), set()
        for v, i, j in pairs:
            if v >= thr and i not in used_p and j not in used_t:
                used_p.add(i)
                used_t.add(j)
                out.append((i, j))
    return out


def average_precision(pred: Sequence[dict], truth: Sequence[dict], thr: float) -> Optional[float]:
    """Mean over subtypes (that have ground truth) of all-point-interpolated AP at one IoU threshold."""
    aps = []
    for sub in sorted({t["subtype"] for t in truth}):
        t_sub = [t for t in truth if t["subtype"] == sub]
        p_sub = sorted((p for p in pred if p["subtype"] == sub), key=lambda p: -p["confidence"])
        taken, tp = set(), []
        for p in p_sub:
            best, bj = 0.0, None
            for j, t in enumerate(t_sub):
                if t["image_id"] == p["image_id"] and j not in taken:
                    v = iou(p["bbox"], t["bbox"])
                    if v > best:
                        best, bj = v, j
            if bj is not None and best >= thr:
                taken.add(bj)
                tp.append(1)
            else:
                tp.append(0)
        ctp, cfp, rec, prec = 0, 0, [], []
        for x in tp:
            ctp, cfp = ctp + x, cfp + (1 - x)
            rec.append(ctp / len(t_sub))
            prec.append(ctp / (ctp + cfp))
        ap, prev_r = 0.0, 0.0
        for k, r in enumerate(rec):                 # precision envelope, then area under it
            ap += (r - prev_r) * max(prec[k:])
            prev_r = r
        aps.append(ap)
    return sum(aps) / len(aps) if aps else None


def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else 100.0 * num / den


def evaluate_inventory(pred: Sequence[dict], truth: Sequence[dict]) -> dict:
    """Measured values (percent) and sample counts. A value is None when it cannot be computed from this ground truth."""
    errs = validate_inventory(truth, truth=True) + validate_inventory(pred)
    if errs:
        raise ValueError("invalid inventory: " + "; ".join(errs[:5]))
    pred, truth = list(pred), list(truth)
    tp_pairs = _match(pred, truth)
    loc_pairs = _match(pred, truth, same_subtype=False)
    tp = len(tp_pairs)
    precision, recall = _rate(tp, len(pred)), _rate(tp, len(truth))
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    right = sum(pred[i]["subtype"] == truth[j]["subtype"] for i, j in loc_pairs)

    def ap_pct(thr):
        v = average_precision(pred, truth, thr)
        return None if v is None else 100.0 * v

    maps = [ap_pct(0.5 + 0.05 * k) for k in range(10)]
    # identity: map each predicted canonical id to the truth instances it matched
    ent_to_inst: Dict[str, set] = defaultdict(set)
    inst_to_ent: Dict[str, set] = defaultdict(set)
    inst_views: Dict[str, set] = defaultdict(set)
    for t in truth:
        inst_views[t["instance_id"]].add(t["image_id"])
    matched_views: Dict[str, set] = defaultdict(set)
    for i, j in tp_pairs:
        e, inst = pred[i]["entity_id"], truth[j]["instance_id"]
        ent_to_inst[e].add(inst)
        inst_to_ent[inst].add(e)
        matched_views[inst].add(pred[i]["image_id"])
    pred_entities = {p["entity_id"] for p in pred}
    split = sum(len(es) - 1 for es in inst_to_ent.values())               # extra canonical ids for one physical object
    eligible = [inst for inst, v in inst_views.items() if len(v) >= 2 and matched_views[inst] == v]
    consistent = sum(len(inst_to_ent[inst]) == 1 for inst in eligible)
    merged = sorted(e for e, s in ent_to_inst.items() if len(s) > 1)        # one canonical id for two different objects
    # fine-grained groups: precision/recall/count accuracy, each reported on its own
    groups: Dict[str, dict] = {}
    for g in sorted({fine_group(t) for t in truth} - {None}):
        gp = [p for p in pred if fine_group(p) == g]
        gt = [t for t in truth if fine_group(t) == g]
        sub_pred, sub_truth = defaultdict(int), defaultdict(int)
        for p in gp:
            sub_pred[(p["image_id"], p["subtype"])] += 1
        for t in gt:
            sub_truth[(t["image_id"], t["subtype"])] += 1
        g_tp = sum(1 for i, j in tp_pairs if fine_group(pred[i]) == g)
        groups[g] = {"precision": _rate(g_tp, len(gp)), "recall": _rate(g_tp, len(gt)), "n_pred": len(gp), "n_truth": len(gt),
                     "instance_accuracy": _rate(sum(sub_pred.get(k, 0) == v for k, v in sub_truth.items()), len(sub_truth)),
                     "n_count_cells": len(sub_truth)}
    return {
        "n_pred": len(pred), "n_truth": len(truth), "true_positives": tp,
        "precision": precision, "recall": recall, "f1": f1, "map50": maps[0],
        "map50_95": None if any(m is None for m in maps) else sum(maps) / len(maps),
        "classification_accuracy": _rate(right, len(loc_pairs)), "n_localised": len(loc_pairs),
        "duplicate_object_rate": _rate(split, len(pred_entities)), "n_pred_entities": len(pred_entities),
        "cross_view_entity_matching": _rate(consistent, len(eligible)), "n_cross_view": len(eligible),
        "merged_entity_ids": merged, "groups": groups,
    }


def inventory_metric_results(pred: Sequence[dict], truth: Sequence[dict]) -> List[dict]:
    """Benchmark ``metric_results`` for everything an inventory can establish. Mask, 3D, material and latency metrics
    are NOT produced here: they stay NOT_RUN until their own evidence exists."""
    m = evaluate_inventory(pred, truth)
    out: List[dict] = []

    def add(mid: str, value: Optional[float], n: int, successes: Optional[int] = None, breakdown: Optional[dict] = None):
        if value is None:
            return
        ci = None
        if successes is not None:
            ci = tuple(None if v is None else 100 * v for v in wilson_ci(successes, n))
        out.append(evaluate_metric(METRIC_BY_ID[mid], value, n, ci=ci, breakdown=breakdown))

    def count(rate, n):
        return None if rate is None else round(rate * n / 100)

    add("A.precision", m["precision"], m["n_pred"], m["true_positives"])
    add("A.recall", m["recall"], m["n_truth"], m["true_positives"])
    add("A.f1", m["f1"], m["n_truth"])
    add("A.map50", m["map50"], m["n_truth"])
    add("A.map50_95", m["map50_95"], m["n_truth"])
    add("A.classification_accuracy", m["classification_accuracy"], m["n_localised"],
        count(m["classification_accuracy"], m["n_localised"]))
    add("C.duplicate_object_rate", m["duplicate_object_rate"], m["n_pred_entities"])
    add("C.cross_view_entity_matching", m["cross_view_entity_matching"], m["n_cross_view"],
        count(m["cross_view_entity_matching"], m["n_cross_view"]), {"merged_entity_ids": m["merged_entity_ids"]})
    for g, v in m["groups"].items():
        add(f"E.{g}.precision", v["precision"], v["n_pred"], count(v["precision"], v["n_pred"]))
        add(f"E.{g}.recall", v["recall"], v["n_truth"], count(v["recall"], v["n_truth"]))
        add(f"E.{g}.instance_accuracy", v["instance_accuracy"], v["n_count_cells"], count(v["instance_accuracy"], v["n_count_cells"]))
    return out
