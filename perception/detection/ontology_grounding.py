"""Ontology grounding for production open-vocabulary detection.

The production multi-engine detection adapter (``perception/detection_adapter.py``)
originally prompted Grounding DINO with a hand-written 25-label tuple. That tuple
had two defects:

* it contained **no small classes at all** (no sockets, switches, door handles,
  hinges, vents, taps, valves), so the 27 ``small: true`` classes in the canonical
  ontology (``config/ontology.yaml``) could never be detected -- not because the
  model failed, but because they were never asked for; and
* detector labels were never resolved against the ontology, so ``category`` was a
  hard-coded two-way string test and the class identity downstream saw was
  whatever free text the model emitted.

This module fixes both without adding a second perception pipeline or a new
dependency. It is a thin, pure layer over the existing canonical ontology
(``perception.ontology``) and the existing ontology bridge
(``perception.detection.extensible_prompter``):

* :func:`production_queries` builds a *bounded* prompt list (Grounding DINO runs
  one forward pass per chunk of queries, so an unbounded list is not usable) that
  reserves a share of the budget for small classes, guaranteeing they are
  queried.
* :func:`ground_label` maps a raw detector label to a canonical ontology node
  and returns provenance (path, group, small flag, world type) or an explicit,
  honest ``resolved=False`` fallback -- an unknown class is never invented.
* :func:`tile_windows` / :func:`nms` support tiled small-object inference: a
  4000 px photograph downscaled to the detector's input makes a 40 px socket
  sub-pixel, which is *why* small fixtures disappear between capture and
  detection. Tiling keeps small regions at native resolution.

Everything here is deterministic and testable without a GPU or model weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from perception.ontology import FALLBACK_PATH, get_default_ontology

#: Ontology groups that describe the built structure rather than loose objects.
#: ``category`` keeps the two-value vocabulary downstream already consumes
#: ("architecture" | "object"); the finer group is surfaced separately as
#: ``Grounding.group`` so nothing that reads ``category`` breaks.
ARCHITECTURE_GROUPS: frozenset[str] = frozenset(
    {
        "architecture",
        "envelope",
        "structural",
        "opening",
        "circulation",
        "building",
        "exterior",
        "road",
        "street",
        "signage",
        "utility",
    }
)

#: Groups treated as belonging to the outdoor domain (used only for filtering).
OUTDOOR_GROUPS: frozenset[str] = frozenset(
    {"road", "street", "vegetation", "exterior", "signage", "utility", "building"}
)

#: Groups treated as belonging to the indoor domain (used only for filtering).
INDOOR_GROUPS: frozenset[str] = frozenset(
    {
        "furniture",
        "decor",
        "electrical",
        "hvac",
        "plumbing",
        "fixtures",
        "fixture",
        "room",
        "finish",
        "architecture",
        "envelope",
        "opening",
        "circulation",
    }
)

#: Default share of the query budget reserved for small classes. Without a
#: reservation the small classes are always crowded out by large furniture and
#: structure, which is exactly the failure this module exists to remove.
DEFAULT_SMALL_SHARE: float = 0.4

#: Last-resort query list, used only when the ontology cannot be loaded so that
#: detection degrades to its historical coverage instead of failing outright.
FALLBACK_QUERIES: tuple[str, ...] = (
    "wall", "floor", "ceiling", "door", "door frame", "window", "window frame",
    "opening", "column", "beam", "stairs", "railing",
    "table", "chair", "sofa", "bed", "cabinet", "wardrobe", "shelf",
    "desk", "sink", "toilet", "refrigerator", "lamp",
)


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = " ".join(str(item).strip().lower().split())
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _domain_allows(group: str, domain: str) -> bool:
    d = (domain or "all").strip().lower()
    if d == "indoor":
        return group not in OUTDOOR_GROUPS
    if d == "outdoor":
        return group not in INDOOR_GROUPS
    return True


def production_queries(
    *,
    domain: str = "all",
    budget: int = 40,
    small_share: float = DEFAULT_SMALL_SHARE,
    extra: Sequence[str] = (),
) -> list[str]:
    """Bounded, deduplicated detector prompts that always include small classes.

    ``budget`` counts *prompts*, not classes: Grounding DINO is invoked once per
    chunk of ``chunk_size`` queries, so keeping the prompt count bounded keeps the
    number of forward passes bounded.

    Ordering is deterministic (structure first, then a reserved block of small
    classes, then the rest), so the same inputs always produce the same prompt
    list -- reproducibility matters when comparing detector runs.

    Args:
        domain: ``"indoor"``, ``"outdoor"`` or ``"all"``.
        budget: maximum number of prompts returned (>= 1).
        small_share: fraction of ``budget`` reserved for small classes.
        extra: caller-supplied prompts prepended to the result.

    Returns:
        A list of lowercase prompts, at most ``budget`` long. Falls back to
        :data:`FALLBACK_QUERIES` when the ontology is empty or unloadable.
    """
    budget = max(1, int(budget))
    extra_clean = _dedupe(extra)

    try:
        onto = get_default_ontology()
    except Exception:  # noqa: BLE001 -- degrade to historical coverage, never crash detection
        return _dedupe([*extra_clean, *FALLBACK_QUERIES])[:budget]

    nodes = [
        n
        for n in onto.nodes.values()
        if n.prompts
        and not n.path.startswith("fallback")
        and _domain_allows(n.group, domain)
    ]
    if not nodes:
        return _dedupe([*extra_clean, *FALLBACK_QUERIES])[:budget]

    small = sorted((n for n in nodes if n.small), key=lambda n: n.path)
    large = sorted(
        (n for n in nodes if not n.small),
        key=lambda n: (n.group not in ARCHITECTURE_GROUPS, n.path),
    )

    small_budget = int(round(budget * max(0.0, min(1.0, small_share))))
    if small and small_budget == 0:
        small_budget = 1  # a small class always gets at least one slot
    small_budget = min(small_budget, len(small))

    large_budget = max(0, budget - small_budget)
    picked = large[:large_budget] + small[:small_budget]

    # Primary prompt for every picked node first, then one variant each, so a
    # tight budget spends its slots on breadth (more classes) before depth
    # (more aliases for one class).
    prompts: list[str] = []
    prompts.extend(n.prompts[0] for n in picked)
    prompts.extend(n.prompts[1] for n in picked if len(n.prompts) > 1)

    return _dedupe([*extra_clean, *prompts])[:budget]


@dataclass(frozen=True)
class Grounding:
    """Canonical ontology resolution of one raw detector label."""

    raw_label: str
    canonical_label: str
    ontology_path: str
    group: str
    category: str  # "architecture" | "object" (unchanged downstream vocabulary)
    is_small: bool
    min_pixels: int
    world_type: str
    resolved: bool
    taxonomy_source: str  # "canonical_ontology" | "prompter_target" | "fallback"

    def to_metadata(self) -> dict[str, object]:
        """Provenance stamped into an observation's ``metadata``."""
        return {
            "ontology_path": self.ontology_path,
            "ontology_label": self.canonical_label,
            "ontology_group": self.group,
            "ontology_world_type": self.world_type,
            "is_small": self.is_small,
            "min_pixels": self.min_pixels,
            "label_resolved": self.resolved,
            "taxonomy_source": self.taxonomy_source,
        }


def _fallback_grounding(raw_label: str) -> Grounding:
    return Grounding(
        raw_label=raw_label,
        canonical_label=raw_label or "object",
        ontology_path=FALLBACK_PATH,
        group="fallback",
        category="object",
        is_small=False,
        min_pixels=0,
        world_type="unknown",
        resolved=False,
        taxonomy_source="fallback",
    )


def ground_label(raw_label: str) -> Grounding:
    """Resolve a raw detector label to a canonical ontology node.

    Resolution order (most authoritative first): exact canonical path/alias, the
    ontology's own alias/name/prompt lookup, then the open-vocabulary prompter's
    richer alias index. A label that resolves to none of these returns an
    explicit ``resolved=False`` grounding pointing at ``fallback/unknown_object``
    -- the raw label is preserved, but no class is invented.
    """
    raw = str(raw_label or "").strip()
    key = " ".join(raw.lower().split())
    if not key:
        return _fallback_grounding(raw)

    try:
        onto = get_default_ontology()
    except Exception:  # noqa: BLE001
        return _fallback_grounding(raw)

    node = onto.get(key) or onto.resolve_or_unknown(key)
    source = "canonical_ontology"

    if node.path.startswith("fallback"):
        # The canonical ontology's own lookup missed; the prompter indexes aliases
        # and a second (universal) class list, and maps its hits back to a path.
        # Only accept a target that actually exists in the canonical ontology so
        # every observation's path is resolvable by ``Ontology.get``.
        try:
            from perception.detection.extensible_prompter import get_extensible_prompter

            target = get_extensible_prompter().resolve_prediction(raw)
        except Exception:  # noqa: BLE001 -- prompter is optional
            target = None

        if target is not None:
            candidate = onto.get(target.path)
            if candidate is not None:
                node, source = candidate, "prompter_target"
            else:
                candidate = onto.resolve_or_unknown(target.canonical_label)
                if not candidate.path.startswith("fallback"):
                    node, source = candidate, "prompter_target"

    if node.path.startswith("fallback"):
        return _fallback_grounding(raw)

    return Grounding(
        raw_label=raw,
        canonical_label=node.label,
        ontology_path=node.path,
        group=node.group,
        category="architecture" if node.group in ARCHITECTURE_GROUPS else "object",
        is_small=node.small,
        min_pixels=node.min_pixels,
        world_type=node.world_type,
        resolved=True,
        taxonomy_source=source,
    )


def tile_windows(
    width: int, height: int, scale: int = 2, overlap: float = 0.2
) -> list[tuple[int, int, int, int]]:
    """Split an image into ``scale x scale`` overlapping tiles in pixel space.

    The first tile starts at 0 and the last ends at ``width``/``height``, so the
    union of the tiles always covers the whole image; ``overlap`` is the
    fractional overlap between adjacent tiles (clamped to ``[0, 0.9)``).

    Returns:
        ``(x0, y0, x1, y1)`` boxes, row-major. ``scale <= 1`` returns the whole
        image as a single window.
    """
    w = max(0, int(width))
    h = max(0, int(height))
    n = max(1, int(scale))
    if n <= 1 or w <= 0 or h <= 0:
        return [(0, 0, w, h)]

    ov = max(0.0, min(0.9, float(overlap)))
    tw = min(w, int(math.ceil(w / n * (1.0 + ov))))
    th = min(h, int(math.ceil(h / n * (1.0 + ov))))

    def _starts(total: int, tile: int) -> list[int]:
        if n == 1 or tile >= total:
            return [0]
        step = (total - tile) / (n - 1)
        return [max(0, min(int(round(i * step)), total - tile)) for i in range(n)]

    windows: list[tuple[int, int, int, int]] = []
    for y0 in _starts(h, th):
        for x0 in _starts(w, tw):
            windows.append((x0, y0, x0 + tw, y0 + th))
    return windows


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
) -> list[int]:
    """Greedy non-maximum suppression; returns kept indices, highest score first.

    Used to collapse the duplicate boxes that tiled inference produces where
    tiles overlap. Class-agnostic on purpose: a genuine class disagreement over
    the same pixels is left to the multi-engine fusion layer, which reports it as
    a competing hypothesis rather than silently dropping one side here.
    """
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    kept: list[int] = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) <= iou_threshold for j in kept):
            kept.append(i)
    return kept


__all__ = [
    "ARCHITECTURE_GROUPS",
    "DEFAULT_SMALL_SHARE",
    "FALLBACK_QUERIES",
    "Grounding",
    "ground_label",
    "nms",
    "production_queries",
    "tile_windows",
]
