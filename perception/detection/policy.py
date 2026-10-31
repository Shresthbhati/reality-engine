"""Structure-only taxonomy: keep what the world is made of, drop what moves."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

_KEEP_ALIASES: Dict[str, Tuple[str, ...]] = {
    "building": ("house", "skyscraper", "tower", "shop", "storefront", "facade"),
    "wall": (),
    "door": ("doorway",),
    "window": (),
    "gate": (),
    "barrier": ("barricade", "police barricade", "jersey barrier", "road barrier", "crash barrier",
                "bollard", "fence", "railing", "guardrail", "traffic barrier", "barrier gate"),
    "pole": ("utility pole", "electric pole"),
    "lamp post": ("street light", "street lamp", "streetlight", "light pole", "lamppost"),
    "traffic light": ("traffic signal", "stoplight"),
    "sign": ("stop sign", "street sign", "road sign", "signboard", "billboard"),
    "bin": ("dustbin", "trash can", "garbage can", "waste bin", "trash bin"),
    "post box": ("letter box", "mailbox", "postbox"),
    "bench": (),
    "fire hydrant": ("hydrant",),
    "stairs": ("steps", "step", "staircase", "stair"),
    "bridge": ("footbridge", "overpass"),
    "station": ("metro station", "metro entrance", "subway entrance", "railway station"),
    "bus shelter": ("bus stop shelter", "bus stop"),
    "kiosk": (),
    "tree": ("palm tree",),
    "sofa": ("couch",),
    "chair": ("armchair", "stool"),
    "table": ("dining table", "coffee table"),
    "bed": (),
    "desk": (),
    "cabinet": ("shelf", "bookcase", "bookshelf", "cupboard", "drawer"),
    "wardrobe": ("closet",),
    "tv": ("television", "monitor"),
    "lamp": ("table lamp", "floor lamp"),
    "sink": (),
    "toilet": (),
    "bathtub": ("bath", "tub"),
    "refrigerator": ("fridge",),
    "stove": ("oven", "cooktop"),
    "curtain": ("blind",),
}
_DROP_ALIASES: Dict[str, Tuple[str, ...]] = {
    "person": ("people", "pedestrian", "crowd", "child", "man", "woman", "boy", "girl", "human",
               "rider", "cyclist", "worker"),
    "car": ("taxi", "cab", "suv", "jeep"),
    "bus": (),
    "truck": ("lorry", "van", "pickup"),
    "motorcycle": ("scooter", "motorbike", "moped"),
    "bicycle": ("bike",),
    "rickshaw": ("auto rickshaw", "cart", "tuk tuk", "autorickshaw", "handcart"),
    "animal": ("dog", "cow", "cat", "bird", "horse", "goat", "sheep", "monkey", "pigeon", "elephant"),
    "foliage": ("undergrowth", "grass", "bush", "shrub", "hedge", "weed", "potted plant", "plant",
                "flower pot", "flowerpot", "flower", "vegetation", "leaf"),
}

KEEP = frozenset(_KEEP_ALIASES)
DROP = frozenset(_DROP_ALIASES)
BARRIER = "barrier"

_ALIAS: Dict[str, str] = {}
for _table in (_KEEP_ALIASES, _DROP_ALIASES):
    for _canon, _aliases in _table.items():
        _ALIAS[_canon] = _canon
        for _a in _aliases:
            _ALIAS[_a] = _canon

STRUCTURE_QUERIES: List[str] = [
    "building", "wall", "door", "window", "gate", "barricade", "police barricade", "jersey barrier",
    "bollard", "fence", "railing", "pole", "street light", "traffic light", "sign", "dustbin",
    "post box", "bench", "fire hydrant", "stairs", "bridge", "metro entrance", "bus stop shelter",
    "kiosk", "tree", "sofa", "chair", "table", "bed", "desk", "cabinet", "bookcase", "tv", "lamp",
    "sink", "toilet", "bathtub", "refrigerator", "stove", "wardrobe", "curtain",
]
TRANSIENT_QUERIES: List[str] = [
    "person", "pedestrian", "crowd", "child", "car", "taxi", "bus", "truck", "van", "motorcycle",
    "scooter", "bicycle", "rickshaw", "auto-rickshaw", "cart", "dog", "cow", "cat", "bird", "horse",
    "bush", "grass", "undergrowth", "flower pot",
]

_IRREGULAR = {"people": "people", "men": "man", "women": "woman", "children": "child",
              "shelves": "shelf", "leaves": "leaf", "buses": "bus", "bushes": "bush",
              "glasses": "glasses", "couches": "couch", "benches": "bench", "wardrobes": "wardrobe"}


def _singular(w: str) -> str:
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def normalize(label: str) -> str:
    s = re.sub(r"^(a|an|the|photo of|picture of)\s+", "", str(label).lower().strip())
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    words = s.split()
    if not words:
        return ""
    words[-1] = _singular(words[-1])
    return " ".join(words)


def canonical(label: str):
    """Canonical class name, or None if the label is not in the taxonomy."""
    n = normalize(label)
    if n in _ALIAS:
        return _ALIAS[n]
    words = n.split()
    if any(w in ("barricade", "barrier", "bollard") for w in words):
        return BARRIER  # a barricade must never be dropped, whatever qualifies it
    if len(words) > 1 and words[-1] in _ALIAS:
        return _ALIAS[words[-1]]
    return None


def classify(label: str) -> str:
    c = canonical(label)
    if c is None:
        return "unknown"
    return "keep" if c in KEEP else "drop"


def filter_detections(dets: Iterable) -> Tuple[list, list, list]:
    """Split into (kept, dropped, unknown). Unknown labels are reported, never silently kept."""
    kept, dropped, unknown = [], [], []
    for d in dets:
        {"keep": kept, "drop": dropped, "unknown": unknown}[classify(d.label)].append(d)
    return kept, dropped, unknown


def keep_ratio(kept: Sequence, dropped: Sequence, unknown: Sequence = ()) -> dict:
    n = len(kept) + len(dropped) + len(unknown)
    return {"total": n, "kept": len(kept), "dropped": len(dropped), "unknown": len(unknown),
            "keep_ratio": (len(kept) / n) if n else None,
            "unknown_labels": sorted({d.label for d in unknown})}
