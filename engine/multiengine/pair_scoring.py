"""Cross-view pair scores from geometric and appearance evidence.

Adapts the repository's own two-view primitives -- ``perception.instances.epipolar.epipolar_residual_px`` (calibrated epipolar
geometry from registered ``PinholeCamera`` poses) and ``perception.instances.appearance`` (colour-histogram intersection) -- to the
multiengine ``Candidate`` / ``PairScore`` contract. It adds no new geometry.

What a score means, and does not:
* A pair is scored ONLY when both detections come from different images that have a registered camera. No camera -> no score (the
  pair is simply not offered); nothing is guessed.
* Geometry is a GATE: the symmetric epipolar residual (the larger of A->B and B->A, in pixels) must be <= ``epipolar_tolerance_px``.
  A point constraint is necessary, not sufficient -- every object along the same epipolar line satisfies it -- so the appearance gate
  and the association layer's same-image exclusion carry the rest. ``basis`` states this.
* Appearance is a GATE when both histograms exist (histogram intersection >= ``appearance_min``); when either is unavailable the pair is
  scored on geometry alone and ``basis["appearance"]`` is ``None`` -- never a fabricated similarity.
* The numeric score only ORDERS merges (strongest first); it is the product of geometric closeness ``1 / (1 + residual / tol)`` and
  the appearance similarity (1.0 when absent). It is not a probability.
* ``epipolar_tolerance_px`` and ``appearance_min`` have NO defaults: the repository's own constant is "not tuned against a real
  dataset", so the caller must state the value it is using, and it is recorded on every pair.
Refused pairs are returned with score 0 and the reason in ``basis``, so the association ledger shows WHY a geometrically or visually
inconsistent pair was not merged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from engine.multiengine.association import Candidate, PairScore
from engine.multiengine.evidence_matrix import EvidenceMatrix
from perception.instances.appearance import (
    AppearanceDescriptor,
    appearance_similarity,
    compute_color_histogram,
)
from perception.instances.epipolar import DEFAULT_EPIPOLAR_TOLERANCE_PX, epipolar_residual_px  # noqa: F401  (re-exported for apps.api)
from world_ir.camera import PinholeCamera

AppearanceFn = Callable[[Candidate, EvidenceMatrix], "AppearanceDescriptor | None"]

LIMITATION = (
    "epipolar geometry is a necessary, not sufficient, condition for identity: other objects on the same epipolar line also pass"
)


@dataclass(frozen=True)
class ScorerConfig:
    epipolar_tolerance_px: float
    appearance_min: float

    def __post_init__(self) -> None:
        if not self.epipolar_tolerance_px > 0:
            raise ValueError("epipolar_tolerance_px must be > 0")
        if not 0.0 <= self.appearance_min <= 1.0:
            raise ValueError("appearance_min must be in [0, 1]")


def box_centre_px(matrix: EvidenceMatrix, oid: str, camera: PinholeCamera) -> tuple[tuple[float, float], str] | None:
    """Detection-box centre in the CAMERA's pixel grid. Resolution-independent when the engine gave a normalised box; otherwise the
    source-pixel box is used and the basis says the camera resolution is ASSUMED to equal the image's."""
    o = matrix.store.get(oid)
    nb = o.predicted.get("box_normalized")
    if isinstance(nb, (list, tuple)) and len(nb) == 4:
        cx, cy = (nb[0] + nb[2]) / 2.0, (nb[1] + nb[3]) / 2.0
        return (cx * camera.intrinsics.width, cy * camera.intrinsics.height), "normalised box scaled to the camera's pixel grid"
    if o.image_region is not None:
        x0, y0, x1, y1 = o.image_region
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0), "source-pixel box (camera resolution ASSUMED equal to the image's)"
    return None


def appearance_from_images(load_rgb: Callable[[str], Any], *, max_side: int = 48) -> AppearanceFn:
    """Appearance descriptors from the original photographs: crop the detection box, downscale to <= ``max_side`` px, build a normalised
    RGB histogram. No mask is used (the box includes background) -- stated, not hidden. ``load_rgb(source_id)`` returns a PIL image."""

    def fn(c: Candidate, matrix: EvidenceMatrix) -> AppearanceDescriptor | None:
        o = matrix.store.get(c.observation_id)
        if o.image_region is None:
            return None
        try:
            img = load_rgb(c.source_id).convert("RGB")
            x0, y0, x1, y1 = (int(round(v)) for v in o.image_region)
            crop = img.crop((max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)))
            if crop.width < 2 or crop.height < 2:
                return None
            scale = min(1.0, max_side / max(crop.width, crop.height))
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
            px = crop.load()
            rows = [[px[x, y][:3] for x in range(crop.width)] for y in range(crop.height)]
            return compute_color_histogram(rows, [[True] * crop.width for _ in range(crop.height)])
        except (OSError, ValueError, KeyError):
            return None  # an unreadable image yields NO descriptor, never a fabricated one

    return fn


def make_pair_scorer(
    cameras: Mapping[str, PinholeCamera], config: ScorerConfig, *, appearance: AppearanceFn | None = None
):
    """A ``pair_scorer`` for ``run_evidence_flow``: ``cameras`` maps source id -> registered camera."""
    tol = config.epipolar_tolerance_px

    def scorer(cands: Sequence[Candidate], matrix: EvidenceMatrix) -> Iterable[PairScore]:
        app_cache: dict[str, AppearanceDescriptor | None] = {}

        def desc(c: Candidate) -> AppearanceDescriptor | None:
            if appearance is None:
                return None
            if c.observation_id not in app_cache:
                app_cache[c.observation_id] = appearance(c, matrix)
            return app_cache[c.observation_id]

        out: list[PairScore] = []
        for i, a in enumerate(cands):
            for b in cands[i + 1 :]:
                if a.source_id == b.source_id:
                    continue  # same image: the association layer refuses this itself; no score is offered
                ca, cb = cameras.get(a.source_id), cameras.get(b.source_id)
                if ca is None or cb is None:
                    continue  # no registered camera for one of the images: geometry cannot be evaluated, so no score
                pa, pb = box_centre_px(matrix, a.observation_id, ca), box_centre_px(matrix, b.observation_id, cb)
                if pa is None or pb is None:
                    continue
                r = max(epipolar_residual_px(ca, pa[0], cb, pb[0]), epipolar_residual_px(cb, pb[0], ca, pa[0]))
                basis: dict[str, Any] = {
                    "method": "symmetric epipolar residual (px) + colour-histogram intersection",
                    "epipolar_residual_px": None if r == float("inf") else round(r, 4), "epipolar_tolerance_px": tol,
                    "appearance_min": config.appearance_min, "pixel_basis": [pa[1], pb[1]], "limitation": LIMITATION,
                }
                if not r <= tol:
                    why = f"epipolar residual {basis['epipolar_residual_px']} px exceeds {tol} px"
                    out.append(PairScore(a.observation_id, b.observation_id, 0.0, {**basis, "refused": why}))
                    continue
                da, db = desc(a), desc(b)
                comparable = da is not None and db is not None and da.bins_per_channel == db.bins_per_channel
                sim = appearance_similarity(da, db) if comparable else None
                basis["appearance"] = None if sim is None else round(sim, 4)
                if sim is not None and sim < config.appearance_min:
                    why = f"appearance {basis['appearance']} below minimum {config.appearance_min}"
                    out.append(PairScore(a.observation_id, b.observation_id, 0.0, {**basis, "refused": why}))
                    continue
                score = (1.0 / (1.0 + r / tol)) * (1.0 if sim is None else sim)
                out.append(PairScore(a.observation_id, b.observation_id, score, basis))
        return out

    return scorer
